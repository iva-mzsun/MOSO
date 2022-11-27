import sys
import time
import torch
import torch.distributed as dist

class Timer():
    def __init__(self):
        self.stime = dict({})
        self.etime = dict({})
        self.pad = dict({})

    def start(self, name, pad=None):
        # dist.barrier()
        self.stime[name] = time.time()
        self.pad[name] = pad if pad else 0

    def convert(self, time, mode='h'):
        '''mode: 'h'-hour, 'm'-minute, 's'-second'''
        if mode == 'h':
            time = time / 3600
            ret = "{:.2f}h".format(time)
        elif mode == 'm':
            time = time / 60
            ret = "{:.2f}m".format(time)
        else:
            ret = "{:.2f}s".format(time)
        return ret

    def end(self, name, mode='h'):
        # dist.barrier()
        self.etime[name] = time.time()
        total_time = self.etime[name] - self.stime[name]
        return self.convert(total_time, mode)

    def duration(self, name, past, total, mode='h'):
        '''
            past: num of done iterations (minus start_step, namely pad)
            total： num of total iterations (minus start_step, namely_pad)
            mode: 'h'-hour, 'm'-minute, 's'-second
        '''
        past -= self.pad[name]
        total -= self.pad[name]

        # dist.barrier()
        past_time = time.time() - self.stime[name]
        total_time = (past_time / past) * total
        left_time = (past_time / past) * (total - past)
        return self.convert(left_time, mode), self.convert(total_time, mode)



def get_timer():
    if not hasattr(get_timer, 'main'):
        get_timer.main = Timer()
    return get_timer.main
