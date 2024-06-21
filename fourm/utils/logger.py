# Copyright 2024 EPFL and Apple Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------
# Based on DETR code base
# https://github.com/facebookresearch/detr
# --------------------------------------------------------
import datetime
import logging
import time
from collections import defaultdict, deque

import torch
import torch.distributed as dist

try:
    import wandb
except:
    pass

from .dist import is_dist_avail_and_initialized


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, iter_len=None, header=None):
        iter_len = iter_len if iter_len is not None else len(iterable)
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(iter_len))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == iter_len - 1:
                if iter_len > 0:
                    eta_seconds = iter_time.global_avg * (iter_len - i)
                    eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                else:
                    eta_string = '?'
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, iter_len if iter_len > 0 else '?', eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, iter_len if iter_len > 0 else '?', eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        time_per_iter_str = '{:.4f}'.format(total_time / iter_len) if iter_len > 0 else '?'
        print('{} Total time: {} ({} s / it)'.format(
            header, total_time_str, time_per_iter_str))


class WandbLogger(object):
    def __init__(self, args):
        wandb.init(
            config=args,
            entity=args.wandb_entity,
            project=args.wandb_project,
            group=getattr(args, 'wandb_group', None),
            name=getattr(args, 'wandb_run_name', None),
            tags=getattr(args, 'wandb_tags', None),
            mode=getattr(args, 'wandb_mode', 'online'),
        )

    @staticmethod
    def wandb_safe_log(*args, **kwargs):
        try:
            wandb.log(*args, **kwargs)
        except (wandb.CommError, BrokenPipeError):
            logging.error('wandb logging failed, skipping...')

    def set_step(self, step=None):
        if step is not None:
            self.step = step
        else:
            self.step += 1

    def update(self, metrics):
        log_dict = dict()
        for k, v in metrics.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            log_dict[k] = v

        self.wandb_safe_log(log_dict, step=self.step)

    def flush(self):
        pass

    def finish(self):
        try:
            wandb.finish()
        except (wandb.CommError, BrokenPipeError):
            logging.error('wandb failed to finish')