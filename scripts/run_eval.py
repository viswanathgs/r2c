"""
Script to compute baseline for Q->AR task given trained models for
answer (Q->A) and rationate (QA->R) tasks. First runs the Q->A task with
the given model, and then conditioned on those predictions, runs the QA->R task.
"""

import argparse
import numpy as np
import torch
import pandas as pd

from allennlp.common.params import Params
from dataloaders.vcr import VCR, VCRLoader
from torch.nn import DataParallel
from utils.pytorch_misc import restore_model_state

from allennlp.models import Model
import models

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', level=logging.DEBUG)
LOG = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--params', dest='params', type=str, required=True,
        help='Params json file',
    )
    parser.add_argument(
        '--batch_size', default=96, type=int,
    )
    parser.add_argument(
        '--answer_model', type=str, default=None,
        help='Path to pytorch model file for Q->A task',
    )
    parser.add_argument(
        '--rationale_model', type=str, default=None,
        help='Path to pytorch model file for QA->R task',
    )
    parser.add_argument(
        '--ar_model', type=str, default=None,
        help='Path to pytorch model file for Q->AR task',
    )
    parser.add_argument(
        '--split', type=str, default='val', choices=['val', 'test'],
    )
    parser.add_argument(
        '--outfile', type=str, default='leaderboard.csv',
        help='Output file for leaderboard csv',
    )
    return parser.parse_args()


def load_val_set(split, params, mode, all_answers_for_rationale=False):
    return VCR(
        split=split,
        mode=mode,
        embs_to_load=params['dataset_reader'].get('embs', 'bert_da'),
        only_use_relevant_dets=params['dataset_reader'].get('only_use_relevant_dets', True),
        all_answers_for_rationale=all_answers_for_rationale,
        use_precomputed_omcs=params['dataset_reader'].get('use_precomputed_omcs', False),
    )


def run_eval(model, model_path, ds, batch_size, num_gpus):
    LOG.info('Loading model state from {}'.format(model_path))
    restore_model_state(model, model_path)

    ds_loader = VCRLoader.from_dataset(
        ds,
        batch_size=(batch_size // num_gpus),
        num_gpus=num_gpus,
        num_workers=(4 * num_gpus),
    )

    def _to_gpu(td):
        if num_gpus > 1:
            return td
        for k in td:
            if k != 'metadata':
                td[k] = {k2: v.cuda(async=True)
                        for k2, v in td[k].items()} if isinstance(td[k], dict) else td[k].cuda(async=True)
        return td

    model.eval()
    ids = []
    probs = []
    labels = []
    for b, batch in enumerate(ds_loader):
        with torch.no_grad():
            batch = _to_gpu(batch)
            output_dict = model(**batch)
            probs.append(output_dict['label_probs'].detach().cpu().numpy())
            ids += [m['annot_id'] for m in batch['metadata']]
            if 'label' in batch:
                labels.append(batch['label'].detach().cpu().numpy())
        LOG.info("{} / {}".format(b + 1, len(ds_loader)))
    labels = np.concatenate(labels, 0) if len(labels) > 0 else None
    probs = np.concatenate(probs, 0)
    return labels, probs, ids


def compute_baseline(model, params, args):
    if args.answer_model:
        LOG.info('Running baseline {} for Q->A task'.format(args.split))
        answer_val = load_val_set(args.split, params, 'answer')
        answer_gt, answer_probs, _ = run_eval(
            model,
            args.answer_model,
            answer_val,
            args.batch_size,
            num_gpus,
        )
        answer_pred = answer_probs.argmax(1)
        answer_accuracy = float(np.mean(answer_gt == answer_pred))
        LOG.info('Baseline Q->A accuracy: {}'.format(answer_accuracy))

    if args.rationale_model:
        LOG.info('Running baseline {} for QA->R task (with ground-truth answers)'.format(args.split))
        rationale_val = load_val_set(args.split, params, 'rationale')
        rationale_gt, rationale_probs, _ = run_eval(
            model,
            args.rationale_model,
            rationale_val,
            args.batch_size,
            num_gpus,
        )
        rationale_pred = rationale_probs.argmax(1)
        rationale_accuracy = float(np.mean(rationale_gt == rationale_pred))
        LOG.info('Baseline QA->R accuracy (ground-truth answers): {}'.format(rationale_accuracy))

    if args.answer_model and args.rationale_model:
        LOG.info('Running baseline {} for QA->R task (with predicted answers)'.format(args.split))
        rationale_val = load_val_set(args.split, params, 'rationale', all_answers_for_rationale=True)
        # Update gt answers with Q->A predictions
        rationale_val.set_answer_labels(answer_pred)
        rationale_gt, rationale_probs, _ = run_eval(
            model,
            args.rationale_model,
            rationale_val,
            args.batch_size,
            num_gpus,
        )
        rationale_pred = rationale_probs.argmax(1)
        rationale_accuracy_pred = float(np.mean(rationale_gt == rationale_pred))
        LOG.info('Baseline QA->R accuracy (predicted answers): {}'.format(rationale_accuracy_pred))

        # Compute accuracy for Q->AR
        ar_gt = list(zip(answer_gt, rationale_gt))
        ar_pred = list(zip(answer_pred, rationale_pred))
        ar_accuracy = float(np.mean([gt == pred for gt, pred in zip(ar_gt, ar_pred)]))
        LOG.info('Baseline Q->A: {}, QA->R: {}, Q->AR: {}'.format(
            answer_accuracy, rationale_accuracy, ar_accuracy))


def joint_eval(model, params, args):
    LOG.info('Running {} for Q->AR task with joint model'.format(args.split))

    ar_val = load_val_set(args.split, params, 'joint')
    ar_gt, ar_probs, _ = run_eval(
        model,
        args.ar_model,
        ar_val,
        args.batch_size,
        num_gpus,
    )
    ar_pred = ar_probs.argmax(1)
    ar_accuracy = float(np.mean(ar_gt == ar_pred))

    # Measure answer and rationale accuracy individually
    answer_gt, rationale_gt = ar_gt // 4, ar_gt % 4
    answer_pred, rationale_pred = ar_pred // 4, ar_pred % 4
    answer_accuracy = float(np.mean(answer_gt == answer_pred))
    rationale_accuracy = float(np.mean(rationale_gt == rationale_pred))

    # Measure QA->R accuracy with ground-truth answers by taking argmax of
    # the subset of the 16-way softmax that corresponds to ground-truth answer.
    rationale_pred = [probs[ans*4 : (ans+1)*4].argmax() \
            for probs, ans in zip(ar_probs, answer_gt)]
    rationale_accuracy_gt = float(np.mean(rationale_gt == rationale_pred))

    LOG.info('Joint Q->A: {}, QA->R: {}, QA->R (gt answers): {}, Q->AR: {}'.format(
        answer_accuracy, rationale_accuracy, rationale_accuracy_gt, ar_accuracy))


def to_leaderboard_csv(probs, ids, outfile):
    # Each row with 20 items such as:
    # [answer, rationale_conditioned_on_a0, rationale_conditioned_on_a1,
    #          rationale_conditioned_on_a2, rationale_conditioned_on_a3].
    assert probs.shape[0] == len(ids)
    assert probs.shape[1] == 20

    group_names = ['answer'] + [f'rationale_conditioned_on_a{i}' for i in range(4)]
    columns = [f'{group_name}_{i}' for group_name in group_names for i in range(4)]
    probs_df = pd.DataFrame(data=probs, columns=columns)
    probs_df['annot_id'] = ids
    probs_df = probs_df.set_index('annot_id', drop=True)
    probs_df.to_csv(outfile)


def joint_test(model, params, args):
    LOG.info('Running {} for Q->AR task with joint model'.format(args.split))

    ar_test = load_val_set(args.split, params, 'joint')
    _, ar_probs, ids = run_eval(
        model,
        args.ar_model,
        ar_test,
        args.batch_size,
        num_gpus,
    )

    # Convert probs to format expected by leaderboard.
    # Need to prepend with probabilities for answer choices.
    answer_probs = np.zeros((ar_probs.shape[0], 4), dtype=ar_probs.dtype)
    for i in range(4):
        answer_probs[:, i] = np.sum(ar_probs[:, i*4:(i+1)*4], axis=1)
    probs = np.concatenate([answer_probs, ar_probs], axis=1)
    to_leaderboard_csv(probs, ids, args.outfile)


if __name__ == '__main__':
    args = parse_args()

    params = Params.from_file(args.params)
    model = Model.from_params(params=params['model'])
    LOG.info('Loaded model {} from {}'.format(
        params['model'].get('type', ''), args.params))

    num_gpus = torch.cuda.device_count()
    assert num_gpus >= 1, "No CUDA devices found"
    LOG.info('Found {} GPUs'.format(num_gpus))
    model = DataParallel(model).cuda() if num_gpus > 1 else model.cuda()

    if args.answer_model or args.rationale_model:
        assert args.split == 'val', "Not yet supported"
        compute_baseline(model, params, args)

    if args.ar_model:
        if args.split == 'val':
            joint_eval(model, params, args)
        else:
            joint_test(model, params, args)
