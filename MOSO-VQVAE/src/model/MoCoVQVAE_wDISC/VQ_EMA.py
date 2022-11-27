import torch.nn as nn
import torch.nn.functional as F
import torch

import numpy as np
import torch as torch
import torch.nn as nn
import ipdb
from einops import repeat, rearrange

class ExponentialMovingAverage(nn.Module):
    """Maintains an exponential moving average for a value.

      Note this module uses debiasing by default. If you don't want this please use
      an alternative implementation.

      This module keeps track of a hidden exponential moving average that is
      initialized as a vector of zeros which is then normalized to give the average.
      This gives us a moving average which isn't biased towards either zero or the
      initial value. Reference (https://arxiv.org/pdf/1412.6980.pdf)

      Initially:

          hidden_0 = 0

      Then iteratively:

          hidden_i = (hidden_{i-1} - value) * (1 - decay)
          average_i = hidden_i / (1 - decay^i)

      Attributes:
        average: Variable holding average. Note that this is None until the first
          value is passed.
      """

    def __init__(self, decay, name = None):
        """Creates a debiased moving average module.

            Args:
              decay: The decay to use. Note values close to 1 result in a slow decay
                whereas values close to 0 result in faster decay, tracking the input
                values more closely.
              name: Name of the module.
            """
        super(ExponentialMovingAverage, self).__init__()
        self._decay = decay
        self._counter = nn.Parameter(torch.zeros(1),
                                     requires_grad=False)

        self._hidden = None
        self.average = None
        self._time_cnt = None

    def initilize(self, target, dtype):
        self._hidden = nn.Parameter(torch.zeros_like(target, dtype=dtype),
                                    requires_grad=False)
        self.average = nn.Parameter(torch.zeros_like(target, dtype=dtype),
                                    requires_grad=False)

    def update(self, value):
        assert self._hidden is not None

        self._counter.data[0] = self._counter.data[0] + 1
        value_ = torch.as_tensor(value).detach()

        # 源代码中_hidden的更新方式跟公式不同，此处与源代码一致
        self._hidden.data = self._hidden.data - (self._hidden.data - value_) * (1. - self._decay)
        self.average.data = self._hidden.data / (1. - pow(self._decay, self._counter.data[0]))
        return self.average.data

    def reinitialize(self, embeddings_cnt, embeddings):
        # embeddings_cnt: [num_embeddings]
        if torch.sum(embeddings_cnt == 0) > 0 and self._counter % 10 == 0:
            zeros = (embeddings_cnt == 0).to(torch.int32)
            cid = np.random.choice(range(len(zeros)), p=(zeros/torch.sum(zeros)).cpu().numpy())
            tar_hidden = torch.mean(self._hidden.transpose(0,1)[embeddings_cnt!=0], dim=0)
            tar_average = torch.mean(self.average.transpose(0,1)[embeddings_cnt!=0], dim=0)
            tar_embedding = torch.mean(embeddings.transpose(0,1)[embeddings_cnt!=0], dim=0)

            self._hidden.data[:, cid] = tar_hidden
            self.average.data[:, cid] = tar_average
            embeddings[:, cid] = tar_embedding
        return embeddings

    def reinitialize2(self, embeddings_cnt, embeddings):
        if self._time_cnt is None:
            self._time_cnt = torch.zeros_like(embeddings_cnt)
        # count how long does each embedding not inparticipate
        self._time_cnt[embeddings_cnt==0] += 1
        self._time_cnt[embeddings_cnt!=0] = 0

        # If an entry has not be count for 1W iters, then involve it in reinitialize
        embeddings_cnt[self._time_cnt >= 10000] = 0

        # embeddings_cnt: [num_embeddings]
        if torch.sum(embeddings_cnt == 0) > 0 and self._counter % 10 == 0:
            # random select a count==0 embedding, and reinitizalize it
            zeros = (embeddings_cnt == 0).to(torch.int32)
            cid = np.random.choice(range(len(zeros)), p=(zeros/torch.sum(zeros)).cpu().numpy())
            tar_hidden = torch.mean(self._hidden.transpose(0,1)[embeddings_cnt!=0], dim=0)
            tar_average = torch.mean(self.average.transpose(0,1)[embeddings_cnt!=0], dim=0)
            tar_embedding = torch.mean(embeddings.transpose(0,1)[embeddings_cnt!=0], dim=0)

            self._hidden.data[:, cid] = tar_hidden
            self.average.data[:, cid] = tar_average
            embeddings[:, cid] = tar_embedding

        return embeddings, embeddings_cnt

    def value(self):
        return self.average

class VectorQuantizerEMA(nn.Module):
    """Sonnet module representing the VQ-VAE layer.

      Implements a slightly modified version of the algorithm presented in
      'Neural Discrete Representation Learning' by van den Oord et al.
      https://arxiv.org/abs/1711.00937

      The difference between VectorQuantizerEMA and VectorQuantizer is that
      this module uses exponential moving averages to update the embedding vectors
      instead of an auxiliary loss. This has the advantage that the embedding
      updates are independent of the choice of optimizer (SGD, RMSProp, Adam, K-Fac,
      ...) used for the encoder, decoder and other parts of the architecture. For
      most experiments the EMA version trains faster than the non-EMA version.

      Input any tensor to be quantized. Last dimension will be used as space in
      which to quantize. All other dimensions will be flattened and will be seen
      as different examples to quantize.

      The output tensor will have the same shape as the input.

      For example a tensor with shape [16, 32, 32, 64] will be reshaped into
      [16384, 64] and all 16384 vectors (each of 64 dimensions)  will be quantized
      independently.

      Attributes:
        embedding_dim: integer representing the dimensionality of the tensors in the
          quantized space. Inputs to the modules must be in this format as well.
        num_embeddings: integer, the number of vectors in the quantized space.
        commitment_cost: scalar which controls the weighting of the loss terms (see
          equation 4 in the paper).
        decay: float, decay for the moving averages.
        epsilon: small float constant to avoid numerical instability.
      """

    def __init__(self, embedding_dim, num_embeddings, commitment_cost,
                 decay, if_augcb=False, epsilon=1e-5, dtype=torch.float32):
        """Initializes a VQ-VAE EMA module.

           Args:
             embedding_dim: integer representing the dimensionality of the tensors in
               the quantized space. Inputs to the modules must be in this format as
               well.
             num_embeddings: integer, the number of vectors in the quantized space.
             commitment_cost: scalar which controls the weighting of the loss terms
               (see equation 4 in the paper - this variable is Beta).
             decay: float between 0 and 1, controls the speed of the Exponential Moving
               Averages.
             epsilon: small constant to aid numerical stability, default 1e-5.
             dtype: dtype for the embeddings variable, defaults to tf.float32.
             name: name of the module.
           """

        super(VectorQuantizerEMA, self).__init__()
        assert (decay >= 0 and decay <= 1)
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.epsilon = epsilon
        self.if_augcb = if_augcb
        self.decay = decay

        embedding_shape = [embedding_dim, num_embeddings]

        self.embeddings = nn.Parameter(torch.rand(embedding_shape, dtype=dtype) * 0.01,
                                     requires_grad=False)
        self._embeddings_cnt = nn.Parameter(torch.zeros([num_embeddings], dtype=torch.int32),
                                            requires_grad=False)
        zeros = torch.zeros(num_embeddings, requires_grad=False)

        self.ema_cluster_size = ExponentialMovingAverage(decay)
        self.ema_cluster_size.initilize(zeros, dtype)

        self.ema_dw = ExponentialMovingAverage(decay)
        self.ema_dw.initilize(self.embeddings, dtype)

    def forward(self, z, is_training):
        """Connects the module to some inputs.

        Args:
          inputs: Tensor, final dimension must be equal to embedding_dim. All other
            leading dimensions will be flattened and treated as a large batch.
          is_training: boolean, whether this connection is to training data. When
            this is set to False, the internal moving average statistics will not be
            updated.

        Returns:
          dict containing the following keys and values:
            quantize: Tensor containing the quantized version of the input.
            loss: Tensor containing the loss to optimize.
            perplexity: Tensor containing the perplexity of the encodings.
            encodings: Tensor containing the discrete encodings, ie which element
            of the quantized space each input element was mapped to.
            encoding_indices: Tensor containing the discrete encoding indices, ie
            which element of the quantized space each input element was mapped to.
        """
        inputs = z.detach().permute(0, 2, 3, 1)
        with torch.no_grad():
            assert inputs.shape[-1] == self.embedding_dim

            flat_inputs = torch.reshape(inputs, [-1, self.embedding_dim])
            distances = (torch.sum(flat_inputs**2, 1, keepdim=True) -
                         2 * torch.matmul(flat_inputs, self.embeddings) +
                         torch.sum(self.embeddings**2, 0, keepdim=True))

            encoding_indices = torch.argmax(-distances, 1)
            encodings = F.one_hot(encoding_indices
                                  , self.num_embeddings).type_as(distances)
            current_cnt = torch.sum(torch.reshape(encodings, [-1, self.num_embeddings]), dim=0)
            self._embeddings_cnt += current_cnt.to(torch.int32)

            quantized = self.quantize(encodings)
            quantized = torch.reshape(quantized, inputs.shape).permute(0, 3, 1, 2)
            encoding_indices = torch.reshape(encoding_indices, inputs.shape[:-1])
        e_latent_loss = torch.mean((quantized.detach() - z)**2)

        if is_training:
            with torch.no_grad():
                dw = torch.matmul(flat_inputs.transpose(0, 1), encodings)
                updated_ema_dw = self.ema_dw.update(dw)

                updated_ema_cluster_size = self.ema_cluster_size.update(
                                            torch.sum(encodings, dim=0))
                n = torch.sum(updated_ema_cluster_size)
                updated_ema_cluster_size = ((updated_ema_cluster_size + self.epsilon) /
                                            (n + self.num_embeddings * self.epsilon) * n)

                self.embeddings.data = updated_ema_dw / \
                                  torch.reshape(updated_ema_cluster_size, [1, -1])

                if self.if_augcb == 1:  # AugCB
                    self.embeddings.data = self.ema_dw.reinitialize(self._embeddings_cnt,
                                                                    self.embeddings.data)
                elif self.if_augcb == 2:
                    embed, embed_cnt = self.ema_dw.reinitialize2(self._embeddings_cnt,
                                                                    self.embeddings.data)
                    self.embeddings.data = embed
                    self._embeddings_cnt.data = embed_cnt

            loss = self.commitment_cost * e_latent_loss

        else:
            loss = self.commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = z + (quantized - z).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs *
                                          torch.log(avg_probs + 1e-10)))

        return {
            'loss': loss,
            'quantize': quantized,
            'perplexity': perplexity,
            'encodings': encodings,
            'encoding_indices': encoding_indices,
        }


    def quantize(self, encodings):
        """Returns embedding tensor for a batch of indices."""
        embeddings = torch.transpose(self.embeddings, 0, 1)
        quantized = torch.matmul(encodings, embeddings)
        return quantized

    def quantize_code(self, tokens):
        """tokens: [B, T, H, W]"""
        B, T, H, W = tokens.shape
        encoding_indices = rearrange(tokens, 'B T H W -> (B T H W)')
        encodings = F.one_hot(encoding_indices
                              , self.num_embeddings).to(torch.float32)
        quantized = self.quantize(encodings)
        quantized = rearrange(quantized, '(B T H W) C -> (B T) C H W', B=B, T=T, H=H, W=W)
        return quantized


def printkey(keys):
    for item in keys:
        print(item)

if __name__ == '__main__':
    model = VectorQuantizerEMA(128, 64, 0.25, 0.99, False)
    printkey(model.state_dict().keys())
    # print(model.state_dict())
    # print(list(model.children()))
