# Script to extract BERT features from OMCS (Open Mind Common Sense) sentences.
# Based off of TF bert extract_features.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import requests
import zipfile
import math
import numpy as np
import pandas as pd
import json

from data.get_bert_embeddings import modeling
from data.get_bert_embeddings import tokenization
import tensorflow as tf
import h5py
from tqdm import tqdm
from data.get_bert_embeddings.vcr_loader import (InputExample, input_fn_builder,
        convert_examples_to_features, retokenize_with_alignment)
from data.get_bert_embeddings.extract_features import (load_bert,
        model_fn_builder, chunk, alignment_gather)
from config import VCR_ANNOTS_DIR

flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer(
    "omcs_score_thresh", 3,
    "Only use OMCS sentences with score >= this value.")
flags.DEFINE_string(
    "output_h5", "bert_da_omcs.h5",
    "Output h5 filename for OMCS BERT embeddings.")

tf.logging.set_verbosity(tf.logging.INFO)


def load_omcs():
    omcs_free = pd.read_csv(
        os.path.join(VCR_ANNOTS_DIR, 'omcs/omcs-sentences-free.txt'),
        sep='\t', header=0,
        error_bad_lines=False)
    omcs_free = omcs_free[omcs_free['language_id'] == 'en']

    omcs_more = pd.read_csv(
        os.path.join(VCR_ANNOTS_DIR, 'omcs/omcs-sentences-more.txt'),
        sep='\t', header=0,
        error_bad_lines=False)
    omcs_more = omcs_more[omcs_more['language_id'] == 'en']

    omcs = pd.concat([omcs_free, omcs_more]).reset_index()

    omcs = omcs[omcs["score"] >= FLAGS.omcs_score_thresh]
    omcs = omcs.drop_duplicates(subset='text', keep="last")

    tf.logging.info('Obtained {} OMCS sentences'.format(len(omcs)))
    return omcs


def omcs_iter(omcs, tokenizer, max_seq_length):
    # Refer to vcr_loader.data_iter and vcr_loader.process_ctx_ans_for_bert
    # for details.
    for i in range(len(omcs)):
        sentence = omcs.iloc[i]["text"]

        # BERT for VCR corpus has been computed such that there's a single
        # embedding per word. That is, if a single word is split into multiple
        # tokens by the word-piece tokenizer, then, mean of the individual BERT
        # embeddings are computed to form the embedding for the word as a whole.
        # We do the same here by first just applying the basic tokenizer to
        # split the sentence into words and punctuations, and then tokenize
        # further with the alignment info stored.
        tokens = tokenizer.basic_tokenizer.tokenize(sentence)
        tokens, alignment = retokenize_with_alignment(tokens, tokenizer)

        # We treat everything as single sentences, so just 2 extra tokens would
        # be added (CLS and SEP) and not 3.
        len_total = len(tokens) + 2
        if len_total > max_seq_length:
            take_away = len_total - max_seq_length
            tokens = tokens[take_away:]
        yield InputExample(unique_id=i, text_a=tokens, text_b=None), alignment


def load_omcs_embeddings(h5_filename, dtype=np.float16):
    '''
    Helper to load OMCS embeddings and the corresponding text.
    '''
    omcs_text = []
    omcs_embeddings = []
    with h5py.File(h5_filename, 'r') as h5:
        for i in range(len(h5)):
            metadata = json.loads(h5[f'{i}/metadata'][()])
            embedding = np.array(h5[f'{i}']['embedding'], dtype=dtype)
            omcs_embeddings.append(embedding)
            omcs_text.append(metadata['text'])
    return omcs_text, omcs_embeddings


if __name__ == '__main__':
    layer_indexes = [int(x) for x in FLAGS.layers.split(",")]

    bert_config, vocab_file = load_bert()
    tokenizer = tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=FLAGS.do_lower_case)

    omcs = load_omcs()
    examples = list(omcs_iter(omcs, tokenizer, FLAGS.max_seq_length))
    tf.logging.info('Obtained {} examples'.format(len(examples)))

    features = convert_examples_to_features(
        [x[0] for x in examples], FLAGS.max_seq_length, tokenizer)
    unique_id_to_ind = {}
    for i, feature in enumerate(features):
        unique_id_to_ind[feature.unique_id] = i
    tf.logging.info('Converted examples to features')

    # TF boilerplate
    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        master=FLAGS.master,
        tpu_config=tf.contrib.tpu.TPUConfig(
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))

    model_fn = model_fn_builder(
        bert_config=bert_config,
        init_checkpoint=FLAGS.init_checkpoint,
        layer_indexes=layer_indexes,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_one_hot_embeddings)

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        predict_batch_size=FLAGS.batch_size)

    chunk_size = max(min(FLAGS.chunk_size, len(features)), 1)
    num_chunks = math.ceil(len(features) / chunk_size)

    output_h5 = h5py.File(FLAGS.output_h5, 'w')
    for i in range(len(examples)):
        output_h5.create_group(f'{i}')

    for chunk_id, feats in enumerate(chunk(features, chunk_size)):
        tf.logging.info("Handling chunk {}/{}".format(chunk_id + 1, num_chunks))
        input_fn = input_fn_builder(features, FLAGS.max_seq_length)
        for result in tqdm(estimator.predict(input_fn, yield_single_examples=True)):
            # Just one layer for now
            layer = result['layer_output_0']

            ind = unique_id_to_ind[int(result["unique_id"])]
            _, alignment = examples[ind]
            embedding = alignment_gather([-1] + alignment, layer)
            output_h5[f'{ind}'].create_dataset('embedding', data=embedding)

            metadata = json.dumps({
                'text': omcs.iloc[ind]['text'],
            })
            output_h5[f'{ind}'].create_dataset('metadata', data=metadata)
