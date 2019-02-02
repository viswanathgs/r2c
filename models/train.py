"""
Training script. Should be pretty adaptable to whatever.
"""
import argparse
import os
import shutil
import signal
import time

# Setting for distributed training.
import torch.multiprocessing as mp
if __name__ == '__main__':
    mp.set_start_method('forkserver')

import numpy as np
import pandas as pd
import torch
from allennlp.common.params import Params
from allennlp.training.learning_rate_schedulers import LearningRateScheduler
from allennlp.training.optimizers import Optimizer
import torch.distributed as dist
from torch.nn import DataParallel
from torch.nn.modules import BatchNorm2d
from torch.nn.parallel import DistributedDataParallel
import torch.utils.data.distributed
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from dataloaders.vcr import VCR, VCRLoader
from utils.pytorch_misc import time_batch, save_checkpoint, clip_grad_norm, \
    restore_checkpoint, print_para, restore_best_checkpoint

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', level=logging.DEBUG)

# This is needed to make the imports work
from allennlp.models import Model
import models

#################################
#################################
######## Data loading stuff
#################################
#################################

parser = argparse.ArgumentParser(description='train')
parser.add_argument(
    '--params',
    dest='params',
    help='Params location',
    type=str,
)
parser.add_argument(
    '--mode',
    type=str,
    choices=['answer', 'rationale', 'joint'],
    default='answer',
)
parser.add_argument(
    '--folder',
    dest='folder',
    help='folder location',
    type=str,
)
parser.add_argument(
    '--no_tqdm',
    dest='no_tqdm',
    action='store_true',
)
parser.add_argument(
    '--dist_url', default='env://', type=str,
    help='URL used to set up distributed training',
)
parser.add_argument(
    '--dist_backend', default='nccl', type=str,
    help='Distributed backend',
)
parser.add_argument(
    '--num_workers', default=None, type=int,
    help='Number of workers in data loader',
)
parser.add_argument(
    '--batch_size', default=96, type=int,
    help='Mini-batch size per node (8 GPUs). In a multi-node setup, '
         'the total batch size will be scaled up appropriately.',
)


def get_distributed_env():
    if 'WORLD_SIZE' in os.environ and len(os.environ['WORLD_SIZE']) > 0:
        world_size = int(os.environ['WORLD_SIZE'])
    else:
        world_size = 1

    if 'RANK' in os.environ and len(os.environ['RANK']) > 0:
        rank = int(os.environ['RANK'])
    else:
        rank = 0

    assert rank < world_size
    return rank, world_size

RANK, WORLD_SIZE = get_distributed_env()
MAIN_PID = os.getpid()


def sigterm_handler(signum, frame):
    print('Received SIGTERM, ignoring', flush=True)


def sigusr_handler(signum, frame):
    print('Received SIGUSR1', flush=True)

    # Only requeue from the main process
    if RANK != 0 or os.getpid() != MAIN_PID:
        return

    if 'SLURM_JOB_ID' not in os.environ:
        print('No SLURM_JOB_ID found, ignoring SIGUSR1', flush=True)
        return

    job_id = os.environ['SLURM_JOB_ID']
    command = 'scontrol requeue {}'.format(job_id)
    if os.system(command):
        raise RuntimeError('Failed to requeue job {}'.format(job_id))
    print('Requeued job {}'.format(job_id))


def main():
    args = parser.parse_args()

    signal.signal(signal.SIGUSR1, sigusr_handler)
    signal.signal(signal.SIGTERM, sigterm_handler)
    print('Signal handler installed', flush=True)

    distributed = WORLD_SIZE > 1

    if distributed:
        print('Distributed setup: world_size={} rank={} master={}:{}'.format(
            WORLD_SIZE, RANK, os.environ['MASTER_ADDR'],
            os.environ['MASTER_PORT']), flush=True)
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=WORLD_SIZE)

    params = Params.from_file(args.params)
    train, val, test = VCR.splits(
        mode=args.mode,
        embs_to_load=params['dataset_reader'].get('embs', 'bert_da'),
        only_use_relevant_dets=params['dataset_reader'].get('only_use_relevant_dets', True),
        use_omcs=params['dataset_reader'].get('use_omcs', True),
    )
    NUM_GPUS = torch.cuda.device_count()
    NUM_CPUS = mp.cpu_count()
    print("Found {} GPUs and {} CPUs".format(NUM_GPUS, NUM_CPUS), flush=True)
    if NUM_GPUS == 0:
        raise ValueError("you need gpus!")

    def _to_gpu(td):
        if NUM_GPUS > 1:
            return td
        for k in td:
            td[k] = {k2: v.cuda(async=True) for k2, v in td[k].items()} if isinstance(td[k], dict) else td[k].cuda(
                async=True)
        return td

    num_workers = 4 * NUM_GPUS if args.num_workers is None else args.num_workers
    print(f"Using {num_workers} workers out of {NUM_CPUS} possible", flush=True)

    train_sampler = DistributedSampler(train) if distributed else None
    loader_params = {
        'batch_size': args.batch_size // NUM_GPUS,
        'num_gpus': NUM_GPUS,
        'num_workers': num_workers,
    }
    train_loader = VCRLoader.from_dataset(train, sampler=train_sampler, **loader_params)
    val_loader = VCRLoader.from_dataset(val, **loader_params)
    test_loader = VCRLoader.from_dataset(test, **loader_params)

    ARGS_RESET_EVERY = 100
    print("Loading {} for {}".format(params['model'].get('type', 'WTF?'), args.mode), flush=True)

    model = Model.from_params(params=params['model'])
    for submodule in model.detector.backbone.modules():
        if isinstance(submodule, BatchNorm2d):
            submodule.track_running_stats = False
        for p in submodule.parameters():
            p.requires_grad = False

    if distributed:
        model.cuda()
        model = DistributedDataParallel(model)
    elif NUM_GPUS > 1:
        model = DataParallel(model).cuda()
    else:
        model.cuda()
    optimizer = Optimizer.from_params([x for x in model.named_parameters() if x[1].requires_grad],
                                      params['trainer']['optimizer'])

    lr_scheduler_params = params['trainer'].pop("learning_rate_scheduler", None)
    scheduler = LearningRateScheduler.from_params(optimizer, lr_scheduler_params) if lr_scheduler_params else None

    if os.path.exists(args.folder):
        print("Found folder! restoring", flush=True)
        start_epoch, val_metric_per_epoch = restore_checkpoint(model, optimizer, serialization_dir=args.folder,
                                                               learning_rate_scheduler=scheduler)
        print("Resuming from epoch {}".format(start_epoch), flush=True)
    else:
        print("Making directories")
        os.makedirs(args.folder, exist_ok=True)
        start_epoch, val_metric_per_epoch = 0, []
        shutil.copy2(args.params, args.folder)

    param_shapes = print_para(model)
    num_batches = 0
    for epoch_num in range(start_epoch, params['trainer']['num_epochs'] + start_epoch):
        if distributed:
            train_sampler.set_epoch(epoch_num)
        train_results = []
        norms = []
        model.train()
        start_time = time.time()
        for b, (time_per_batch, batch) in enumerate(time_batch(train_loader if args.no_tqdm else tqdm(train_loader), reset_every=ARGS_RESET_EVERY)):
            batch = _to_gpu(batch)
            optimizer.zero_grad()
            output_dict = model(**batch)
            loss = output_dict['loss'].mean() + output_dict['cnn_regularization_loss'].mean()
            loss.backward()

            num_batches += 1
            if scheduler:
                scheduler.step_batch(num_batches)

            norms.append(
                clip_grad_norm(model.named_parameters(), max_norm=params['trainer']['grad_norm'], clip=True, verbose=False)
            )
            optimizer.step()

            train_results.append(pd.Series({'loss': output_dict['loss'].mean().item(),
                                            'crl': output_dict['cnn_regularization_loss'].mean().item(),
                                            'accuracy': (model.module if NUM_GPUS > 1 else model).get_metrics(
                                                reset=(b % ARGS_RESET_EVERY) == 0)[
                                                'accuracy'],
                                            'sec_per_batch': time_per_batch,
                                            'hr_per_epoch': len(train_loader) * time_per_batch / 3600,
                                            }))
            if b % ARGS_RESET_EVERY == 0 and b > 0:
                norms_df = pd.DataFrame(pd.DataFrame(norms[-ARGS_RESET_EVERY:]).mean(), columns=['norm']).join(
                    param_shapes[['shape', 'size']]).sort_values('norm', ascending=False)

                print("e{:2d}b{:5d}/{:5d}. norms: \n{}\nsumm:\n{}\n~~~~~~~~~~~~~~~~~~\n".format(
                    epoch_num, b, len(train_loader),
                    norms_df.to_string(formatters={'norm': '{:.2f}'.format}),
                    pd.DataFrame(train_results[-ARGS_RESET_EVERY:]).mean(),
                ), flush=True)

        train_elapsed = time.time() - start_time
        print("Epoch {} done. Elapsed = {}s.".format(epoch_num, train_elapsed))
        print("---\nTRAIN EPOCH {:2d}:\n{}\n----".format(epoch_num, pd.DataFrame(train_results).mean()))

        val_probs = []
        val_labels = []
        val_loss_sum = 0.0
        model.eval()
        start_time = time.time()
        for b, (time_per_batch, batch) in enumerate(time_batch(val_loader)):
            with torch.no_grad():
                batch = _to_gpu(batch)
                output_dict = model(**batch)
                val_probs.append(output_dict['label_probs'].detach().cpu().numpy())
                val_labels.append(batch['label'].detach().cpu().numpy())
                val_loss_sum += output_dict['loss'].mean().item() * batch['label'].shape[0]
        val_labels = np.concatenate(val_labels, 0)
        val_probs = np.concatenate(val_probs, 0)
        val_loss_avg = val_loss_sum / val_labels.shape[0]

        val_metric_per_epoch.append(float(np.mean(val_labels == val_probs.argmax(1))))
        if scheduler:
            scheduler.step(val_metric_per_epoch[-1], epoch_num)

        val_elapsed = time.time() - start_time
        print("Val epoch {} done. Elapsed = {}s.".format(epoch_num, val_elapsed))
        print("Val epoch {} has acc {:.3f} and loss {:.3f}".format(epoch_num, val_metric_per_epoch[-1], val_loss_avg),
              flush=True)

        if int(np.argmax(val_metric_per_epoch)) < (len(val_metric_per_epoch) - 1 - params['trainer']['patience']):
            print("Stopping at epoch {:2d}".format(epoch_num), flush=True)
            break
        if RANK == 0:
            save_checkpoint(
                model, optimizer, args.folder, epoch_num, val_metric_per_epoch,
                is_best=int(np.argmax(val_metric_per_epoch)) == (len(val_metric_per_epoch) - 1))

    # todo - make blind test work.
    # print("STOPPING. now running on the test set", flush=True)
    # # Load best
    # restore_best_checkpoint(model, args.folder)
    # model.eval()
    # for eval_set, name in [(test_loader, 'test'), (val_loader, 'val')]:
    #     test_probs = []
    #     test_labels = []
    #     for b, (time_per_batch, batch) in enumerate(time_batch(eval_set)):
    #         with torch.no_grad():
    #             batch = _to_gpu(batch)
    #             output_dict = model(**batch)
    #             test_probs.append(output_dict['label_probs'].detach().cpu().numpy())
    #             test_labels.append(batch['label'].detach().cpu().numpy())
    #     test_labels = np.concatenate(test_labels, 0)
    #     test_probs = np.concatenate(test_probs, 0)
    #     acc = float(np.mean(test_labels == test_probs.argmax(1)))
    #     print("Final {} accuracy is {:.3f}".format(name, acc))
    #     np.save(os.path.join(args.folder, f'{name}preds.npy'), test_probs)


if __name__ == '__main__':
    main()
