from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import faiss
import logging
import numpy as np
import torch
import os
import h5py
from torch import nn
from copy import deepcopy
from tqdm import tqdm
import time

from allennlp.nn.util import masked_softmax
from config import VCR_ANNOTS_DIR
from data.omcs.extract_omcs_features import load_omcs_embeddings

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--vcr_h5', type=str, required=True,
        help='h5 file with BERT embeddings for VCR dataset',
    )
    parser.add_argument(
        '--omcs_index', type=str, default=None,
        help='File to output faiss index to. If already present, index '
             'is reused and without regneration',
    )
    parser.add_argument(
        '--omcs_h5', type=str,
        default=os.path.join(VCR_ANNOTS_DIR, 'omcs/bert_da_omcs.h5'),
        help='h5 file with BERT embeddings generated for OMCS sentences',
    )
    parser.add_argument(
        '--outdir', type=str, default=os.path.join(VCR_ANNOTS_DIR, 'omcs'),
    )
    parser.add_argument(
        '-k', type=int, default=5,
        help='Maximum number of neighbors to use',
    )
    parser.add_argument(
        '--similarity_thresh', type=float, default=0.5,
        help='Only use embeddings with cosine similarity above this threshold',
    )
    return parser.parse_args()


def normalize(emb):
    norm = np.linalg.norm(emb, axis=-1)
    return emb / norm[:, np.newaxis]


def build_index(embeddings):
    # IndexFlat is sufficient for now
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)

    # Faiss inner product doesn't normalize. Let's normalize here
    # to match torch.nn.CosineSimilarity.
    # We could also use L2 here now since L2^2 = 2 - 2 * IP
    embs = normalize(embeddings)
    index.add(embs)
    return index


def load_omcs(args):
    # Embeddings are stored as float16, but faiss requires float32
    text, _, word_embs = load_omcs_embeddings(args.omcs_h5, dtype=np.float32)
    word_embs = np.vstack(word_embs)

    if args.omcs_index is not None and os.path.exists(args.omcs_index):
        index = faiss.read_index(args.omcs_index)
    else:
        index = build_index(word_embs)
        if args.omcs_index is not None:
            faiss.write_index(index, args.omcs_index)

    assert len(word_embs) == index.ntotal
    assert word_embs.shape[1] == index.d
    return word_embs, index


def get_omcs_embeddings_for_vcr(omcs_embs, omcs_index, vcr_embs, args):
    # vcr_embs: (n, d), omcs_embs: (N, d)
    # Index lookup. Normalize for cosine similarity.
    vcr_embs = normalize(vcr_embs)
    # D, I: (n, k)
    D, I = omcs_index.search(vcr_embs, k=args.k)

    # Compute softmax of similarity scores.
    # Only use those with cosine similarity scores above thresh.
    mask = (D >= args.similarity_thresh).astype(np.float32)
    # (n, k)
    attention_wts = masked_softmax(
        torch.from_numpy(D),
        torch.from_numpy(mask),
    )

    # Fetch the nearest found embeddings and then apply attention
    # using the computed weights.
    nearest_omcs_embs = torch.from_numpy(omcs_embs[I])  # (n, k, d)
    attended_omcs_embs = torch.einsum('nk,nkd->nd',
                                      (attention_wts, nearest_omcs_embs))
    return attended_omcs_embs


def enrich_vcr_with_omcs(args):
    omcs_embs, omcs_index = load_omcs(args)
    LOG.info('Loaded faiss index with OMCS emnbeddings, ntotal={}'.format(
        omcs_index.ntotal))

    co = faiss.GpuMultipleClonerOptions()
    co.shard = False  # Replica mode (dataparallel) instead of shard mode
    omcs_index = faiss.index_cpu_to_all_gpus(omcs_index, co)

    vcr_h5 = h5py.File(args.vcr_h5, 'r')
    LOG.info('Loaded VCR embeddings from {}, found {} entities'.format(
        args.vcr_h5, len(vcr_h5)))

    outfile = os.path.basename(args.vcr_h5).split('.')[0] + '_omcs.h5'
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    outfile = os.path.join(args.outdir, outfile)
    LOG.info('Writing output OMCS embeddings to {}'.format(outfile))
    output_h5 = h5py.File(outfile, 'w')
    for i in range(len(vcr_h5)):
        output_h5.create_group(f'{i}')

    for i in tqdm(range(len(vcr_h5))):
        grp = vcr_h5[str(i)]
        out_grp = output_h5[str(i)]

        # Each key has embeddings of dim (num_words, d). Batch over all keys.
        items = {k: np.array(v, dtype=np.float32) for k, v in grp.items()}
        vcr_embs = np.vstack(items.values())
        vcr_omcs_embs = get_omcs_embeddings_for_vcr(omcs_embs, omcs_index, vcr_embs, args)
        assert vcr_embs.shape == vcr_omcs_embs.shape

        # Convert back to float16 to match BERT VCR format
        vcr_omcs_embs = vcr_omcs_embs.numpy().astype(np.float16)

        # Unbatch based on word counts
        word_counts = [v.shape[0] for v in items.values()]
        vcr_omcs_embs = np.split(vcr_omcs_embs, np.cumsum(word_counts)[:-1])
        assert len(vcr_omcs_embs) == len(items)

        # Write in the same format as vcr_h5 file
        for key, data in zip(items.keys(), vcr_omcs_embs):
            out_grp.create_dataset(key, data=data)

    LOG.info('Success!')


if __name__ == '__main__':
    args = parse_args()
    enrich_vcr_with_omcs(args)
