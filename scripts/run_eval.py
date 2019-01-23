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
        '--answer_model', type=str, required=True,
        help='Path to pytorch model file for Q->A task',
    )
    parser.add_argument(
        '--rationale_model', type=str, required=True,
        help='Path to pytorch model file for QA->R task',
    )
    return parser.parse_args()


def load_val_set(params, mode):
    assert mode in ['answer', 'rationale']
    return VCR(
        split='val',
        mode=mode,
        embs_to_load=params['dataset_reader'].get('embs', 'bert_da'),
        only_use_relevant_dets=params['dataset_reader'].get('only_use_relevant_dets', True),
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
    pred = probs.argmax(1)
    return labels, pred


def compute_baseline(args):
    params = Params.from_file(args.params)
    model = Model.from_params(params=params['model'])
    LOG.info('Loaded model {} from {}'.format(
        params['model'].get('type', ''), args.params))

    num_gpus = torch.cuda.device_count()
    assert num_gpus >= 1, "No CUDA devices found"
    LOG.info('Found {} GPUs'.format(num_gpus))
    model = DataParallel(model).cuda() if num_gpus > 1 else mode.cuda()

    LOG.info('Running val for Q->A task')
    answer_val = load_val_set(params, 'answer')
    answer_gt, answer_pred = run_eval(
        model,
        args.answer_model,
        answer_val,
        args.batch_size,
        num_gpus,
    )
    answer_accuracy = float(np.mean(answer_gt == answer_pred))
    LOG.info('Q->A accuracy: {}'.format(answer_accuracy))

    LOG.info('Running val for QA->R task (with ground-truth answers)')
    rationale_val = load_val_set(params, 'rationale')
    rationale_gt, rationale_pred = run_eval(
        model,
        args.rationale_model,
        rationale_val,
        args.batch_size,
        num_gpus,
    )
    rationale_accuracy = float(np.mean(rationale_gt == rationale_pred))
    LOG.info('QA->R accuracy (ground-truth answers): {}'.format(rationale_accuracy))

    LOG.info('Running val for QA->R task (with predicted answers)')
    rationale_val = load_val_set(params, 'rationale')
    # Update gt answers with Q->A predictions
    # TODO (viswanath): This doesn't work yet - BERT contextual embeddings need
    # to be computed for the (question, predicted_answer).
    rationale_val.set_answer_labels(answer_pred)
    rationale_gt, rationale_pred = run_eval(
        model,
        args.rationale_model,
        rationale_val,
        args.batch_size,
        num_gpus,
    )
    rationale_accuracy_pred = float(np.mean(rationale_gt == rationale_pred))
    LOG.info('QA->R accuracy (predicted answers): {}'.format(rationale_accuracy_pred))

    # Compute accuracy for Q->AR
    ar_gt = list(zip(answer_gt, rationale_gt))
    ar_pred = list(zip(answer_pred, rationale_pred))
    ar_accuracy = float(np.mean(ar_gt == ar_pred))
    LOG.info('Q->A: {}, QA->R: {}, Q->AR: {}'.format(
        answer_accuracy, rationale_accuracy, ar_accuracy))


if __name__ == '__main__':
    args = parse_args()
    compute_baseline(args)
