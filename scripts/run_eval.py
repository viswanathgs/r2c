"""
Script to compute baseline for Q->AR task given trained models for
answer (Q->A) and rationate (QA->R) tasks. First runs the Q->A task with
the given model, and then conditioned on those predictions, runs the QA->R task.
"""

import argparse
import numpy as np
import torch

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
    return parser.parse_args()


def load_val_set(params, mode, all_answers_for_rationale=False):
    return VCR(
        split='val',
        mode=mode,
        embs_to_load=params['dataset_reader'].get('embs', 'bert_da'),
        only_use_relevant_dets=params['dataset_reader'].get('only_use_relevant_dets', True),
        all_answers_for_rationale=all_answers_for_rationale,
        use_omcs=params['dataset_reader'].get('use_omcs', False),
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
            td[k] = {k2: v.cuda(async=True) for k2, v in td[k].items()} if isinstance(td[k], dict) else td[k].cuda(
                async=True)
        return td

    model.eval()
    probs = []
    labels = []
    for b, batch in enumerate(ds_loader):
        with torch.no_grad():
            batch = _to_gpu(batch)
            output_dict = model(**batch)
            probs.append(output_dict['label_probs'].detach().cpu().numpy())
            labels.append(batch['label'].detach().cpu().numpy())
        LOG.info("{} / {}".format(b + 1, len(ds_loader)))
    labels = np.concatenate(labels, 0)
    probs = np.concatenate(probs, 0)
    return labels, probs


def compute_baseline(model, params, args):
    if args.answer_model:
        LOG.info('Running baseline val for Q->A task')
        answer_val = load_val_set(params, 'answer')
        answer_gt, answer_probs = run_eval(
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
        LOG.info('Running baseline val for QA->R task (with ground-truth answers)')
        rationale_val = load_val_set(params, 'rationale')
        rationale_gt, rationale_probs = run_eval(
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
        LOG.info('Running baseline val for QA->R task (with predicted answers)')
        rationale_val = load_val_set(params, 'rationale', all_answers_for_rationale=True)
        # Update gt answers with Q->A predictions
        rationale_val.set_answer_labels(answer_pred)
        rationale_gt, rationale_probs = run_eval(
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
    LOG.info('Running val for Q->AR task with joint model')

    ar_val = load_val_set(params, 'joint')
    ar_gt, ar_probs = run_eval(
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


if __name__ == '__main__':
    args = parse_args()

    params = Params.from_file(args.params)
    model = Model.from_params(params=params['model'])
    LOG.info('Loaded model {} from {}'.format(
        params['model'].get('type', ''), args.params))

    num_gpus = torch.cuda.device_count()
    assert num_gpus >= 1, "No CUDA devices found"
    LOG.info('Found {} GPUs'.format(num_gpus))
    model = DataParallel(model).cuda() if num_gpus > 1 else mode.cuda()

    if args.answer_model or args.rationale_model:
        compute_baseline(model, params, args)

    if args.ar_model:
        joint_eval(model, params, args)
