import argparse
import os
import random
import time
import warnings
import copy
import numpy as np

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms


from utils import AverageMeter, ProgressMeter, ToTensorAndNormalize, accuracy, parse_gpus
from report_acc_regime import init_acc_regime, update_acc_regime
from loss import BinaryCrossEntropyWithLogits
from checkpoint import save_checkpoint, load_checkpoint
from thop import profile
from networks import create_net


parser = argparse.ArgumentParser(description='PyTorch Self-supervised Learning for Abstract Visual Reasoning')

# dataset settings
parser.add_argument('--dataset-dir', default='datasets/',
                    help='path to dataset')
parser.add_argument('--dataset-name', default='RAVEN-FAIR',
                    help='dataset name')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 128), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-ba', '--batch-accum', default=1, type=int,
                    metavar='N',
                    help='number of mini-batches to accumulate gradient over before updating (default: 1)')
parser.add_argument('--image-size', default=96, type=int,
                    help='image size')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 16)')


# network settings
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    help='model architecture (default: resnet18)')
parser.add_argument('--num-extra-stages', default=1, type=int,
                    help='number of extra subconeptconv layers')
parser.add_argument('--imagenet-pretrained', action='store_true',
                    help='whether use ImageNet pretrained weights')
parser.add_argument('--classifier-hidreduce', default=4, type=int,
                    help='projection dimensions before the last classifier')
parser.add_argument('--block-drop', default=0.0, type=float,
                    help="dropout within each block")
parser.add_argument('--classifier-drop', default=0.0, type=float,
                    help="dropout within classifier block")
parser.add_argument('--in-channels', default=1, type=int,
                    help="input image channels")
parser.add_argument('--ou-channels', default=1, type=int,
                    help="classifier output channels (number of classes in supervised learning)")


# training settings
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

# others settings
parser.add_argument("--ckpt", default="./ckpts/", 
                    help="folder to output checkpoints")
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default="0",
                    help='GPU id to use.')
parser.add_argument('-p', '--print-freq', default=50, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument("--fp16", action='store_true',
                    help="whether to use fp16 for training")
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on test set')
parser.add_argument('--show-detail', action='store_true',
                    help="whether to show detail accuracy on all sub-types")
parser.add_argument('--unsupervised-training', action='store_true',
                    help='whether to train a model in an unsupervised manner')


# seed the sampling process for reproducibility
# https://pytorch.org/docs/stable/notes/randomness.html
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_data_loader(args, data_split = 'train', transform = None):

    if 'RAVEN' in args.dataset_name:
        from data import RAVEN as create_dataset
    # elif 'PGM' in args.dataset_name:
    elif 'PGM' in args.dataset_dir:
        from data import PGM as create_dataset
    elif 'VAD' in args.dataset_dir:
        from data import VAD as create_dataset
        

    dataset = create_dataset(
        args.dataset_dir, 
        data_split = data_split, 
        image_size = args.image_size, 
        transform = transform
    )

    if args.seed is not None:
        g = torch.Generator()
        g.manual_seed(args.seed)

        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=(data_split == "train"),
            num_workers=args.workers, pin_memory=False, sampler=None,
            generator=g, worker_init_fn=seed_worker, persistent_workers=True
        )
    else:
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=(data_split == "train"),
            num_workers=args.workers, pin_memory=False, sampler=None,
            persistent_workers=True
        )

    return data_loader



best_acc = 0

def main():
    args = parser.parse_args()

    args.dataset_dir = os.path.join(args.dataset_dir, args.dataset_name)

    args.ckpt += args.dataset_name
    args.ckpt += "-" + args.arch
    args.ckpt += "-es" + str(args.num_extra_stages)

    if args.block_drop > 0.0 or args.classifier_drop > 0.0:
        args.ckpt += "-b" + str(args.block_drop) + "c" + str(args.classifier_drop)

    args.ckpt += "-imsz" + str(args.image_size)
    args.ckpt += "-wd" + str(args.weight_decay)
    args.ckpt += "-ep" + str(args.epochs)

    if args.unsupervised_training:
        args.ckpt += "-ul"

    args.gpu = parse_gpus(args.gpu)
    if args.gpu is not None:
        args.device = torch.device("cuda:{}".format(args.gpu[0]))
    else:
        args.device = torch.device("cpu")

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
        args.ckpt += '-seed' + str(args.seed)

    else:
        cudnn.deterministic = False
        cudnn.benchmark = True

    if not os.path.isdir(args.ckpt):
        os.makedirs(args.ckpt)

    main_worker(args)


def main_worker(args):

    global best_acc

    # create model
    model = create_net(args)

    log_path = os.path.join(args.ckpt, "log.txt")

    if os.path.exists(log_path):
        log_file = open(log_path, mode="a")
    else:
        log_file = open(log_path, mode="w")
    
    for key, value in vars(args).items():
        log_file.write('{0}: {1}\n'.format(key, value))

    args.log_file = log_file

    model_flops = copy.deepcopy(model)

    if args.dataset_name == "IQMath":
        x = torch.randn(2, 3, args.image_size, args.image_size)
    else:
        x = torch.randn(2, 16, args.image_size, args.image_size)
    flops, params = profile(model_flops, inputs=(x,))
    del model_flops

    print("model [%s] - params: %.6fM" % (args.arch, params / 1e6))
    print("model [%s] - FLOPs: %.6fG" % (args.arch, flops / 1e9))
        
    args.log_file.write("--------------------------------------------------\n")
    args.log_file.write("Network - " + args.arch + "\n")
    args.log_file.write("Params - %.6fM" % (params / 1e6) + "\n")
    args.log_file.write("FLOPs - %.6fG" % (flops / 1e9) + "\n")


    if args.evaluate == False:
        print(model)

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    else:
        torch.cuda.set_device(args.device)
        model = model.to(args.gpu[0])
        model = torch.nn.DataParallel(model, args.gpu) 

    # define loss function (criterion) and optimizer
    criterion = BinaryCrossEntropyWithLogits().cuda(args.gpu)

    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr = args.lr, 
        weight_decay = args.weight_decay
    )

    if args.resume:
        model, optimizer, best_acc, start_epoch = load_checkpoint(args, model, optimizer)
        args.start_epoch = start_epoch


    # --------------------------------------------------------------------------------------------------------------
    # Create data loader

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        ToTensorAndNormalize()
    ])
    
    test_transform = transforms.Compose([ToTensorAndNormalize()])

    train_loader = get_data_loader(args, data_split='train', transform=train_transform)
    test_loader  = get_data_loader(args, data_split='test',  transform=test_transform)
    
    args.log_file.write("Number of training samples: %d\n" % len(train_loader.dataset))
    args.log_file.write("Number of testing samples: %d\n" % len(test_loader.dataset))

    args.log_file.write("--------------------------------------------------\n")
    args.log_file.close()


    if args.evaluate:
        acc = validate(test_loader, model, criterion, args, valid_set="Test")
        return

    if args.fp16:
        args.scaler = torch.cuda.amp.GradScaler()


    best_epoch = 0

    for epoch in range(args.start_epoch, args.epochs):
        
        args.log_file = open(log_path, mode="a")

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)
        acc = validate(test_loader, model, criterion, args, valid_set="Test")
        is_best = acc > best_acc
        best_acc = max(acc, best_acc)

        save_checkpoint({
            "epoch": epoch + 1,
            "arch": args.arch,
            "state_dict": model.state_dict(),
            "best_acc": best_acc,
            "optimizer" : optimizer.state_dict(),
            }, is_best, epoch, save_path=args.ckpt)

        if is_best:
            best_epoch = epoch

        epoch_msg = ("----------- Best Acc at [{}]: Test {:.3f} -----------".format(best_epoch, best_acc))
        print(epoch_msg)

        args.log_file.write(epoch_msg + "\n")
        args.log_file.close()


def train(data_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc', ':6.2f')

    progress = ProgressMeter(
        len(data_loader),
        [batch_time, data_time, losses, top1],
        prefix = "Epoch: [{}]".format(epoch))


    param_groups = optimizer.param_groups[0]
    curr_lr = param_groups["lr"]

    # switch to train mode
    model.train()
    accum_track = 0
    end = time.time()
    for i, (images, target, meta_target, structure_encoded, data_file) in enumerate(data_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.to(args.device, non_blocking=True)
            target = target.to(args.device, non_blocking=True)

        # compute output
        if args.fp16:
            with torch.cuda.amp.autocast():
                if args.unsupervised_training:
                    output, labels = model(images)
                else:
                    output = model(images)
                    labels = target

                loss = criterion(output, labels)
                losses.update(loss.item(), images.size(0))
                
            args.scaler.scale(loss).backward()

            accum_track += 1
            if(accum_track == args.batch_accum):
                args.scaler.step(optimizer)
                args.scaler.update()
                accum_track = 0
                optimizer.zero_grad()
        else:
            if args.unsupervised_training:
                output, labels = model(images)
            else:
                output = model(images)
                labels = target
                
            loss = criterion(output, labels)
            losses.update(loss.item(), images.size(0))
            # compute gradient and do SGD step
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # measure accuracy and record loss
        acc1 = accuracy(output, labels)
        top1.update(acc1[0][0], images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 or i == len(data_loader) - 1:
            epoch_msg = progress.get_message(i)
            epoch_msg += ("\tLr  {:.4f}".format(curr_lr))
            args.log_file.write(epoch_msg + "\n")
            print(epoch_msg)


def validate(data_loader, model, criterion, args, valid_set='Valid'):
    
    if 'RAVEN' in args.dataset_name:
        acc_regime = init_acc_regime(args.dataset_name)
    else:
        acc_regime = None

    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc', ':6.2f')

    progress = ProgressMeter(
        len(data_loader),
        [batch_time, losses, top1],
        prefix = valid_set + ': ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target, meta_target, structure_encoded, data_file) in enumerate(data_loader):
           
            if args.gpu is not None:
                images = images.to(args.device, non_blocking=True)
                target = target.to(args.device, non_blocking=True)

            # compute outputs
            output = model(images)
            loss = criterion(output, target)
            losses.update(loss.item(), images.size(0))

            # measure accuracy and record loss
            acc1 = accuracy(output, target)
            
            top1.update(acc1[0][0], images.size(0))

            if acc_regime is not None:
                update_acc_regime(args.dataset_name, acc_regime, output, target, structure_encoded, data_file)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0 or i == len(data_loader) - 1:
                epoch_msg = progress.get_message(i)
                print(epoch_msg)

        if acc_regime is not None:
            for key in acc_regime.keys():
                if acc_regime[key] is not None:
                    if acc_regime[key][1] > 0:
                        acc_regime[key] = float(acc_regime[key][0]) / acc_regime[key][1] * 100
                    else:
                        acc_regime[key] = None

            mean_acc = 0
            for key, val in acc_regime.items():
                mean_acc += val
            mean_acc /= len(acc_regime.keys())
        else:
            mean_acc = top1.avg

        epoch_msg = '----------- {valid_set} Acc {mean_acc:.3f} -----------'.format(valid_set=valid_set, mean_acc=mean_acc)

        print(epoch_msg)
        
        if args.evaluate == False:
            args.log_file.write(epoch_msg + "\n")


    return top1.avg


if __name__ == '__main__':
    main()