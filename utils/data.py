import torch
from torch.utils.data import Dataset, DataLoader, random_split, DistributedSampler
from torch import nn
from glob import glob
import random
import os
import numpy as np


class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        # if record_stream() doesn't work, another option is to make sure device inputs are created
        # on the main stream.
        # self.next_input_gpu = torch.empty_like(self.next_input, device='cuda')
        # self.next_target_gpu = torch.empty_like(self.next_target, device='cuda')
        # Need to make sure the memory allocated for next_* is not still in use by the main stream
        # at the time we start copying to next_*:
        # self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            # more code for the alternative if record_stream() doesn't work:
            # copy_ will record the use of the pinned source tensor in this side stream.
            # self.next_input_gpu.copy_(self.next_input, non_blocking=True)
            # self.next_target_gpu.copy_(self.next_target, non_blocking=True)
            # self.next_input = self.next_input_gpu
            # self.next_target = self.next_target_gpu

            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            self.next_input = self.next_input.float()
            self.next_target = self.next_target.float()

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        if input is not None:
            input.record_stream(torch.cuda.current_stream())
        if target is not None:
            target.record_stream(torch.cuda.current_stream())
        self.preload()
        return input, target


class PATDataset(Dataset):
    def __init__(self, path) -> None:
        super().__init__()
        paths = glob(os.path.join(path, '*.npz'))
        self.data = []
        for path in paths:
            idx = np.load(path)
            self.data.append([idx['sinogram'], idx['gt']])

    def __getitem__(self, index):
        sinogram, gt = self.data[index]
        sinogram = sinogram.astype(np.float32)
        gt = gt[None, ...].astype(np.float32)

        return sinogram, gt
    
    def __len__(self):
        return len(self.data)


def load_data(args):
    dataset = PATDataset(path=args.data.data_dir)
    train_set, test_set = random_split(
        dataset, [0.9, 0.1]
    )
    train_sampler = DistributedSampler(train_set)
    test_sampler = DistributedSampler(test_set)
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=args.training.batch_size,
        sampler=train_sampler,
        num_workers=args.data.num_workers,
        prefetch_factor=args.data.prefetch_factor,
        pin_memory=True
    )
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=args.training.batch_size,
        sampler=test_sampler,
        num_workers=args.data.num_workers,
        prefetch_factor=args.data.prefetch_factor,
        pin_memory=True
    )
    return train_loader, train_sampler, test_loader, test_sampler


@torch.no_grad()
def random_mask(x, axis=2, n_keep=32):
    mask = torch.zeros_like(x).cpu()
    mask_axis = random.sample(range(x.shape[axis]), n_keep)
    mask_shape = [slice(None)] * x.dim()
    mask_shape[axis] = mask_axis
    mask[tuple(mask_shape)] = 1.

    return mask.cuda()

@torch.no_grad()
def uniform_mask(x, axis=2, n_keep=32):
    mask = torch.zeros_like(x).cpu()
    mask_axis = max_gap_interval(x.shape[axis], n_keep)
    mask_shape = [slice(None)] * x.dim()
    mask_shape[axis] = mask_axis
    mask[tuple(mask_shape)] = 1.

    return mask.cuda()

@torch.no_grad()
def limited_view(x, axis=2, n_keep=32):
    mask = torch.zeros_like(x).cpu()
    start = random.randint(0, x.shape[axis]-1)
    mask_axis = [i % x.shape[axis] for i in range(start, start + n_keep)]
    mask_shape = [slice(None)] * x.dim()
    mask_shape[axis] = mask_axis
    mask[tuple(mask_shape)] = 1.

    return mask.cuda()

@torch.no_grad()
def get_mask_fn(args):
    mask_type = args.data.mask
    n_keep=args.data.num_known
    def mask_fn(x, axis=-1):
        if mask_type == 'uniform_mask':
            mask = uniform_mask(x, axis, n_keep)
        elif mask_type == 'random_mask':
            mask = random_mask(x, axis, n_keep)
        elif mask_type == 'limited_view':
            mask = limited_view(x, axis, n_keep)
        else:
            raise ValueError(f'Sampling pattern {mask_type} unsupported!')

        x_masked = mask * x

        return x_masked
    
    return mask_fn
    
def max_gap_interval(n, n_keep):
    step = n / (n_keep + 1)
    result = [int(round(step * i)) for i in range(1, n_keep + 1)]
    return result
