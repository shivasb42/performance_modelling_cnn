#!/usr/bin/env python3
"""
main.py - PyTorch ImageNet training (modified for short benchmark runs)

Features added/changed from original:
- --max-iters to stop training after N iterations
- conditional pin_memory based on device (only True for CUDA)
- benchmarking helpers: estimate FLOPs (ptflops optional), params bytes,
  activation bytes (approx via hooks)
- train() returns timing stats (avg batch time, throughput)
- after one measured run, writes a JSON file with benchmark results
- sets OMP/MKL/PyTorch thread limits in main()
"""

import argparse
import json
import os
import random
import shutil
import time
import warnings
from enum import Enum

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Subset

# optional FLOPs estimator
try:
    from ptflops import get_model_complexity_info
    HAVE_PT = True
except Exception:
    HAVE_PT = False

# model list
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


# -------------------------
# Helper functions for benchmarking
# -------------------------
def bytes_for_params(model):
    total = 0
    for p in model.parameters():
        total += p.numel() * p.element_size()
    return total

def collect_activation_sizes(model, device):
    """
    Run a single forward and register hooks on leaf modules to sum output sizes.
    This provides an approximate activation bytes count for one sample.
    """
    activation_bytes = 0
    handles = []
    def hook_fn(module, inp, out):
        nonlocal activation_bytes
        try:
            if isinstance(out, torch.Tensor):
                activation_bytes += out.numel() * out.element_size()
            elif isinstance(out, (list, tuple)):
                for t in out:
                    if isinstance(t, torch.Tensor):
                        activation_bytes += t.numel() * t.element_size()
        except Exception:
            # ignore hooking errors on weird modules
            pass

    for m in model.modules():
        # only register on leaf modules (no children)
        if len(list(m.children())) == 0:
            try:
                handles.append(m.register_forward_hook(hook_fn))
            except Exception:
                pass

    try:
        x = torch.randn(1, 3, 224, 224, device=device)
        with torch.no_grad():
            model(x)
    except Exception:
        # fallback: if device can't run or input size incompatible, return 0
        activation_bytes = 0

    for h in handles:
        try:
            h.remove()
        except Exception:
            pass

    return activation_bytes

def estimate_flops_per_image(model):
    """
    Return estimated FLOPs per image (FP32) using ptflops if available.
    Returns None if ptflops not installed.
    """
    if not HAVE_PT:
        return None
    macs, params = get_model_complexity_info(model, (3,224,224), as_strings=False, print_per_layer_stat=False, verbose=False)
    flops = float(macs) * 2.0  # MACs -> FLOPs
    return flops


# -------------------------
# Arg parser (mostly original)
# -------------------------
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR', nargs='?', default='imagenet',
                    help='path to dataset (default: imagenet)')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--no-accel', action='store_true',
                    help='disables accelerator')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--dummy', action='store_true', help="use fake data to benchmark")
parser.add_argument('--max-iters', default=0, type=int,
                    help='stop training after this many iterations (0 = disabled)')
parser.add_argument('--device', default=None, type=str,
                    help='force device string: "cpu", "cuda", or "mps" (overrides auto detect)')
parser.add_argument('--bench-out', default=None, type=str,
                    help='optional output filename for benchmark JSON (defaults to generated name)')


best_acc1 = 0


def main():
    args = parser.parse_args()

    # Limit threads for responsiveness on laptops
    os.environ.setdefault('OMP_NUM_THREADS', '2')
    os.environ.setdefault('MKL_NUM_THREADS', '2')
    try:
        torch.set_num_threads(2)
    except Exception:
        pass

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ.get("WORLD_SIZE", -1))

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    # device selection
    if args.device:
        dev_str = args.device
        if dev_str == 'cuda' and not torch.cuda.is_available():
            print("Requested device cuda but torch.cuda not available. Falling back to cpu.")
            device = torch.device('cpu')
        else:
            device = torch.device(dev_str)
        # wrap for compatibility with rest of code using torch.accelerator
        # set args.no_accel accordingly if cpu chosen
        if device.type == 'cpu':
            args.no_accel = True
    else:
        use_accel = not args.no_accel and torch.accelerator.is_available()
        if use_accel:
            device = torch.accelerator.current_accelerator()
        else:
            device = torch.device("cpu")

    # print device for info
    print(f"Using device: {device}")

    # determine gpus per node (used if multiprocessing distributed)
    if isinstance(device, torch.device) and device.type == 'cuda':
        ngpus_per_node = torch.cuda.device_count()
    elif hasattr(device, 'type') and device.type == 'cuda':
        ngpus_per_node = torch.cuda.device_count()
    elif hasattr(device, 'type') and device.type == 'mps':
        ngpus_per_node = 1
    elif hasattr(device, 'type') and device.type == 'cpu':
        ngpus_per_node = 1
    else:
        ngpus_per_node = 1

    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(None, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    # decide device again here for clarity (use torch.accelerator if available)
    use_accel = not args.no_accel and torch.accelerator.is_available()
    if args.device:
        # if user forced device via CLI
        if args.device == 'cuda':
            device = torch.device('cuda')
        elif args.device == 'mps':
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        if use_accel:
            device = torch.accelerator.current_accelerator()
        else:
            device = torch.device("cpu")

    # distributed init if requested
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ.get("RANK", "0"))
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

    # move model to device / handle DataParallel cases
    if not (hasattr(device, 'type') and device.type != 'cuda') and str(device).startswith('cuda'):
        # CUDA case: try to use DataParallel / DistributedDataParallel as original
        if args.distributed:
            if args.gpu is not None:
                torch.cuda.set_device(args.gpu)
                model.cuda()
                args.batch_size = int(args.batch_size / ngpus_per_node)
                args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            else:
                model.cuda()
                model = torch.nn.parallel.DistributedDataParallel(model)
        else:
            if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
                model.features = torch.nn.DataParallel(model.features)
                model.cuda()
            else:
                model = torch.nn.DataParallel(model).cuda()
    else:
        # CPU or MPS or other accelerator
        try:
            model.to(device)
        except Exception:
            # fallback to cpu
            device = torch.device("cpu")
            model.to(device)

    # -------------------------
    # QUICK BENCH / COMPLEXITY ESTIMATES (per-image / per-batch)
    # -------------------------
    flops_forward_per_image = None
    try:
        flops_forward_per_image = estimate_flops_per_image(model)
    except Exception:
        flops_forward_per_image = None

    params_bytes = bytes_for_params(model)
    try:
        activation_bytes = collect_activation_sizes(model, device)
    except Exception:
        activation_bytes = 0

    input_bytes_per_sample = 3 * 224 * 224 * 4  # float32
    flops_forward_per_batch = (flops_forward_per_image or 0.0) * args.batch_size
    flops_train_per_batch_est = flops_forward_per_batch * 3.0  # forward+back+opt heuristic
    dram_bytes_per_batch_est = args.batch_size * (input_bytes_per_sample + activation_bytes) + params_bytes

    args._bench_est = {
        'flops_forward_per_image': flops_forward_per_image,
        'flops_forward_per_batch': flops_forward_per_batch,
        'flops_train_per_batch_est': flops_train_per_batch_est,
        'params_bytes': params_bytes,
        'activation_bytes_one_sample': activation_bytes,
        'dram_bytes_per_batch_est': dram_bytes_per_batch_est,
    }

    print("=== BENCH ESTIMATES ===")
    print(f"flops_forward_per_image: {flops_forward_per_image}")
    print(f"flops_forward_per_batch (batch {args.batch_size}): {flops_forward_per_batch}")
    print(f"approx training FLOPs per batch: {flops_train_per_batch_est}")
    print(f"params_bytes: {params_bytes}, activation_bytes(one sample): {activation_bytes}")
    print(f"dram_bytes_per_batch_est: {dram_bytes_per_batch_est}")
    print("========================")

    # define loss function (criterion), optimizer, and learning rate scheduler
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                loc = f'{device.type}:{args.gpu}'
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint.get('epoch', 0)
            best_acc1 = checkpoint.get('best_acc1', 0)
            try:
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                scheduler.load_state_dict(checkpoint['scheduler'])
            except Exception:
                print("Warning: checkpoint couldn't be fully loaded (possible device mismatch).")
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint.get('epoch', 'N/A')))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # Data loading code
    if args.dummy:
        print("=> Dummy data is used!")
        train_dataset = datasets.FakeData(1281167, (3, 224, 224), 1000, transforms.ToTensor())
        val_dataset = datasets.FakeData(50000, (3, 224, 224), 1000, transforms.ToTensor())
    else:
        traindir = os.path.join(args.data, 'train')
        valdir = os.path.join(args.data, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

        val_dataset = datasets.ImageFolder(
            valdir,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=True)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=(hasattr(device, 'type') and device.type == 'cuda'), sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=(hasattr(device, 'type') and device.type == 'cuda'), sampler=val_sampler)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch (collect timing stats)
        stats = train(train_loader, model, criterion, optimizer, epoch, device, args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args)

        # collate bench results and save JSON
        bench = getattr(args, '_bench_est', {})
        out = {
            'arch': args.arch,
            'device': getattr(device, 'type', str(device)),
            'batch_size': args.batch_size,
            'avg_batch_time_s': stats.get('avg_batch_time_s'),
            'throughput_imgs_per_s': stats.get('throughput_imgs_per_s'),
            'measured_batches': stats.get('measured_batches'),
            'flops_forward_per_image': bench.get('flops_forward_per_image'),
            'flops_forward_per_batch': bench.get('flops_forward_per_batch'),
            'flops_train_per_batch_est': bench.get('flops_train_per_batch_est'),
            'dram_bytes_per_batch_est': bench.get('dram_bytes_per_batch_est'),
            'params_bytes': bench.get('params_bytes'),
            'activation_bytes_one_sample': bench.get('activation_bytes_one_sample'),
        }

        if out['avg_batch_time_s'] and out['flops_train_per_batch_est']:
            out['GFLOP_per_s_est'] = (out['flops_train_per_batch_est'] / out['avg_batch_time_s']) / 1e9
            out['arithmetic_intensity_Flop_per_byte'] = (out['flops_train_per_batch_est'] / out['dram_bytes_per_batch_est']) if out['dram_bytes_per_batch_est'] else None
        else:
            out['GFLOP_per_s_est'] = None
            out['arithmetic_intensity_Flop_per_byte'] = None

        # choose filename
        fname = args.bench_out if args.bench_out else f'benchmark_result_{args.arch}_{getattr(device, "type", str(device))}_b{args.batch_size}.json'
        try:
            with open(fname, 'w') as f:
                json.dump(out, f, indent=2)
            print(f"Saved benchmark results to {fname}")
        except Exception as e:
            print(f"Failed to save benchmark results to {fname}: {e}")

        scheduler.step()

        # remember best acc@1 and save checkpoint if desired
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }, is_best)


def train(train_loader, model, criterion, optimizer, epoch, device, args):
    """
    Training loop — collects per-batch forward+backward times for benchmarking.
    Returns dict with avg batch time and throughput.
    """
    use_accel = not args.no_accel and torch.accelerator.is_available()

    batch_time = AverageMeter('Time', use_accel, ':6.3f', Summary.NONE)
    data_time = AverageMeter('Data', use_accel, ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', use_accel, ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', use_accel, ':6.2f', Summary.NONE)
    top5 = AverageMeter('Acc@5', use_accel, ':6.2f', Summary.NONE)
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    model.train()

    measured_times = []
    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        if getattr(args, 'max_iters', 0) and (i + 1) > args.max_iters:
            break

        # measure data loading time
        data_time.update(time.time() - end)

        # move to device
        if hasattr(device, 'type') and device.type == 'cuda' and not getattr(args, 'no_accel', False):
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
        else:
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

        # timed block
        t0 = time.time()
        output = model(images)
        loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        t1 = time.time()

        # measure elapsed and record
        measured_times.append(t1 - t0)
        batch_time.update(t1 - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i + 1)

    stats = {}
    if len(measured_times) > 0:
        avg_batch_time = float(sum(measured_times) / len(measured_times))
        throughput = args.batch_size / avg_batch_time
        stats['avg_batch_time_s'] = avg_batch_time
        stats['throughput_imgs_per_s'] = throughput
        stats['measured_batches'] = len(measured_times)
    else:
        stats['avg_batch_time_s'] = None
        stats['throughput_imgs_per_s'] = None
        stats['measured_batches'] = 0

    return stats


def validate(val_loader, model, criterion, args):
    use_accel = not args.no_accel and torch.accelerator.is_available()

    def run_validate(loader, base_progress=0):
        if use_accel:
            device = torch.accelerator.current_accelerator()
        else:
            device = torch.device("cpu")

        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(loader):
                i = base_progress + i
                if use_accel:
                    if args.gpu is not None and device.type == 'cuda':
                        torch.accelerator.set_device_index(args.gpu)
                        images = images.cuda(args.gpu, non_blocking=True)
                        target = target.cuda(args.gpu, non_blocking=True)
                    else:
                        images = images.to(device)
                        target = target.to(device)
                else:
                    images = images.to(device)
                    target = target.to(device)

                output = model(images)
                loss = criterion(output, target)

                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                # aggregate losses and accuracies for display
                # reuse AverageMeter objects local to this function
                try:
                    losses.update(loss.item(), images.size(0))
                    top1.update(acc1[0], images.size(0))
                    top5.update(acc5[0], images.size(0))
                except Exception:
                    # these will be created below if not present (safe guard)
                    pass

                # measure elapsed time
                try:
                    batch_time.update(time.time() - end)
                except Exception:
                    pass
                end = time.time()

                if i % args.print_freq == 0:
                    try:
                        progress.display(i + 1)
                    except Exception:
                        pass

    # create meters used in display inside run_validate
    use_accel = not args.no_accel and torch.accelerator.is_available()
    batch_time = AverageMeter('Time', use_accel, ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', use_accel, ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', use_accel, ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', use_accel, ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader) + (args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset))),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    model.eval()
    run_validate(val_loader)

    if args.distributed:
        top1.all_reduce()
        top5.all_reduce()

    if args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset)):
        aux_val_dataset = Subset(val_loader.dataset,
                                 range(len(val_loader.sampler) * args.world_size, len(val_loader.dataset)))
        aux_val_loader = torch.utils.data.DataLoader(
            aux_val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=(hasattr(torch, 'cuda') and torch.cuda.is_available()))
        run_validate(aux_val_loader, len(val_loader))

    progress.display_summary()
    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        try:
            shutil.copyfile(filename, 'model_best.pth.tar')
        except Exception:
            pass


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, use_accel, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.use_accel = use_accel
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        if self.use_accel:
            device = torch.accelerator.current_accelerator()
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
