import argparse
import datetime
import glob
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import webdataset as wds

from util.crop import center_crop_arr
import util.misc as misc

import copy
from engine_jit import train_one_epoch, evaluate

from denoiser import Denoiser
from denoiser_cot import DenoiserCoT
from denoiser_repa import DenoiserRepa


def get_args_parser():
    parser = argparse.ArgumentParser('JiT', add_help=False)

    # architecture
    parser.add_argument('--model', default='JiT-B/16', type=str, metavar='MODEL',
                        help='Name of the model to train')
    parser.add_argument('--img_size', default=256, type=int, help='Image size')
    parser.add_argument('--attn_dropout', type=float, default=0.0, help='Attention dropout rate')
    parser.add_argument('--proj_dropout', type=float, default=0.0, help='Projection dropout rate')

    # training
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='Epochs to warm up LR')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size per GPU (effective batch size = batch_size * # GPUs)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='Learning rate (absolute)')
    parser.add_argument('--blr', type=float, default=5e-5, metavar='LR',
                        help='Base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='Minimum LR for cyclic schedulers that hit 0')
    parser.add_argument('--lr_schedule', type=str, default='constant',
                        help='Learning rate schedule')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='Weight decay (default: 0.0)')
    parser.add_argument('--ema_decay1', type=float, default=0.9999,
                        help='The first ema to track. Use the first ema for sampling by default.')
    parser.add_argument('--ema_decay2', type=float, default=0.9998,
                        help='The second ema to track')
    parser.add_argument('--P_mean', default=-0.8, type=float)
    parser.add_argument('--P_std', default=0.8, type=float)
    parser.add_argument('--D_mean', default=-0.8, type=float)
    parser.add_argument('--D_std', default=0.8, type=float)
    parser.add_argument('--dino_pixel_offset', default=0.0, type=float)
    parser.add_argument('--dino_pixel_shift', default=1.0, type=float)
    parser.add_argument('--noise_scale', default=1.0, type=float)
    parser.add_argument('--t_eps', default=5e-2, type=float)
    parser.add_argument('--label_drop_prob', default=0.1, type=float)

    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='Starting epoch')
    parser.add_argument('--num_workers', default=12, type=int)
    parser.add_argument('--prefetch_factor', default=6, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for faster GPU transfers')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # dino
    parser.add_argument('--latent_model', default='dino', type=str)  # dino, mocov3
    parser.add_argument('--dino_max_t', default=1.0, type=float)
    parser.add_argument('--dino_weight', default=1.0, type=float)
    parser.add_argument('--sample_mode', default='default', type=str)  # dino_first, 
    parser.add_argument('--choose_dino_p', default=0.0, type=float)  # dino_first, 
    parser.add_argument('--bottleneck_dim_dino', default=128, type=int)
    parser.add_argument('--dino_in_channels', default=768, type=int)
    parser.add_argument('--dh_depth', default=0, type=int)
    parser.add_argument('--dh_hidden_size', default=2048, type=int)
    parser.add_argument('--mask_p', default=0.0, type=float)
    parser.add_argument('--override_guidance', action='store_true')


    # sampling
    parser.add_argument('--sampling_method', default='heun', type=str,
                        help='ODE samping method')
    parser.add_argument('--num_sampling_steps', default=50, type=int,
                        help='Sampling steps')
    parser.add_argument('--cfg', default=1.0, type=float,
                        help='Classifier-free guidance factor')
    parser.add_argument('--cfg_dino', default=1.0, type=float)
    parser.add_argument('--interval_min', default=0.0, type=float,
                        help='CFG interval min')
    parser.add_argument('--interval_max', default=1.0, type=float,
                        help='CFG interval max')
    parser.add_argument('--interval_min_dino', default=0.0, type=float)
    parser.add_argument('--interval_max_dino', default=1.0, type=float)
    parser.add_argument('--num_images', default=50000, type=int,
                        help='Number of images to generate')
    parser.add_argument('--eval_freq', type=int, default=40,
                        help='Frequency (in epochs) for evaluation')
    parser.add_argument('--online_eval', action='store_true')
    parser.add_argument('--evaluate_gen', action='store_true')
    parser.add_argument('--keep_images', action='store_true')
    parser.add_argument('--gen_bsz', type=int, default=256,
                        help='Generation batch size')
    parser.add_argument('--autoguidance_ckpt', default='', type=str)
    parser.add_argument('--autoguidance_ema', default='1', type=str) # 'none', '1', '2'
    parser.add_argument('--generation_ema', default='1', type=str) # 'none', '1', '2'
    parser.add_argument('--t_eps_inference', default=0.05, type=float)
    parser.add_argument('--gen_shift_pixel', default=1.0, type=float)
    parser.add_argument('--gen_shift_dino', default=1.0, type=float)
    parser.add_argument('--guidance_method', default='cfg', type=str, help='cfg autoguidance cfg_interval') 
    
    # dataset
    parser.add_argument('--data_path', default='/path/to/ImageNet_2012_webdataset', type=str,
                        help='Path to the WebDataset shards')
    parser.add_argument('--dataset_size', default=1281167, type=int, help='Total number of images in dataset')
    parser.add_argument('--class_num', default=1000, type=int)

    # checkpointing
    parser.add_argument('--output_dir', default='./output_dir',
                        help='Directory to save outputs (empty for no saving)')
    parser.add_argument('--resume', default='',
                        help='Folder that contains checkpoint to resume from')
    parser.add_argument('--save_last_freq', type=int, default=5,
                        help='Frequency (in epochs) to save checkpoints')
    parser.add_argument('--log_freq', default=100, type=int)
    parser.add_argument('--device', default='cuda',
                        help='Device to use for training/testing')
    parser.add_argument('--checkpoint_keep_freq', default=100, type=int)


    # distributed training
    parser.add_argument('--world_size', default=1, type=int,
                        help='Number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='URL used to set up distributed training')

    return parser


def main(args):
    misc.init_distributed_mode(args)
    print('Job directory:', os.path.dirname(os.path.realpath(__file__)))
    print("Arguments:\n{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # Set seeds for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    # Set up TensorBoard logging (only on main process)
    if global_rank == 0 and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.output_dir)
    else:
        log_writer = None

    # Data augmentation transforms
    transform_train = transforms.Compose([
        transforms.Lambda(lambda img: center_crop_arr(img, args.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.PILToTensor()
    ])

    # WebDataset Pipeline
    
    train_paths_pattern = os.path.join(args.data_path, 'train', '*.tar')
    train_paths = glob.glob(train_paths_pattern)
    dataset_train = (
        wds.WebDataset(train_paths, shardshuffle=len(train_paths), nodesplitter=wds.split_by_node, workersplitter=wds.split_by_worker)
        .repeat()
        .shuffle(1000)
        .decode("pil")
        .to_tuple("jpg;png;jpeg", "cls")
        .map_tuple(transform_train, lambda x: x)
        .batched(args.batch_size, partial=False)
    )

    # Compute number of batches per epoch for the loader
    batches_per_epoch = args.dataset_size // (args.batch_size * num_tasks)

    data_loader_train = wds.WebLoader(
        dataset_train,
        batch_size=None,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        prefetch_factor=args.prefetch_factor,
    ).with_epoch(batches_per_epoch).with_length(batches_per_epoch)

    print(f"WebDataset loaded from {args.data_path}")

    torch._dynamo.config.cache_size_limit = 128
    torch._dynamo.config.optimize_ddp = False

    # Create denoiser
    if "CoT" in args.model:
        model = DenoiserCoT(args)
    elif "Repa" in args.model:
        model = DenoiserRepa(args)
    else:
        model = Denoiser(args)

    print("Model =", model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of trainable parameters: {:.6f}M".format(n_params / 1e6))

    model.to(device)

    eff_batch_size = args.batch_size * misc.get_world_size()
    if args.lr is None:  # only base_lr (blr) is specified
        args.lr = args.blr * eff_batch_size / 256

    print("Base lr: {:.2e}".format(args.lr * 256 / eff_batch_size))
    print("Actual lr: {:.2e}".format(args.lr))
    print("Effective batch size: %d" % eff_batch_size)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    model_without_ddp = model.module

    # Set up optimizer with weight decay adjustment for bias and norm layers
    param_groups = misc.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)

    # Resume from checkpoint if provided
    checkpoint_path = os.path.join(args.resume, "checkpoint-last.pth") if args.resume else None
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if args.autoguidance_ckpt:
            autoguidance_checkpoint = torch.load(args.autoguidance_ckpt, map_location='cpu', weights_only=False)
            ag_key = {'none': 'model', '1': 'model_ema1', '2': 'model_ema2'}[args.autoguidance_ema]
            for model_key in ['model', 'model_ema1', 'model_ema2']:
                checkpoint[model_key].update({'ag_' + k:v for k,v in autoguidance_checkpoint[ag_key].items() if k.startswith('net.')})

        model_without_ddp.load_state_dict(checkpoint['model'], strict=False)

        ema_state_dict1 = checkpoint['model_ema1']
        ema_state_dict2 = checkpoint['model_ema2']
        model_without_ddp.ema_params1 = [ema_state_dict1[name].cuda() for name, _ in model_without_ddp.named_parameters()]
        model_without_ddp.ema_params2 = [ema_state_dict2[name].cuda() for name, _ in model_without_ddp.named_parameters()]
        print("Resumed checkpoint from", args.resume)

        if 'optimizer' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
            print("Loaded optimizer & scaler state!")
        del checkpoint
    else:
        model_without_ddp.ema_params1 = copy.deepcopy(list(model_without_ddp.parameters()))
        model_without_ddp.ema_params2 = copy.deepcopy(list(model_without_ddp.parameters()))
        print("Training from scratch")

    # Evaluate generation
    if args.evaluate_gen:
        print("Evaluating checkpoint at {} epoch".format(args.start_epoch))
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            with torch.no_grad():
                evaluate(model_without_ddp, args, 1, batch_size=args.gen_bsz, log_writer=log_writer)
        return

    # Training loop
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        train_one_epoch(model, model_without_ddp, data_loader_train, optimizer, device, epoch, log_writer=log_writer, args=args)

        # Save checkpoint periodically
        if epoch % args.save_last_freq == 0 or epoch + 1 == args.epochs:
            misc.save_model(
                args=args,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                epoch=epoch,
                epoch_name="last"
            )

        if epoch % args.checkpoint_keep_freq == 0 and epoch > 0:
            misc.save_model(
                args=args,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                epoch=epoch
            )

        # Perform online evaluation at specified intervals
        if args.online_eval and (epoch % args.eval_freq == 0 or epoch + 1 == args.epochs):
            torch.cuda.empty_cache()
            with torch.no_grad():
                evaluate(model_without_ddp, args, epoch, batch_size=args.gen_bsz, log_writer=log_writer)
            torch.cuda.empty_cache()

        if misc.is_main_process() and log_writer is not None:
            log_writer.flush()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time:', total_time_str)


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)