"""
Dataloaders for VCR
"""
import json
import os

import numpy as np
import torch
from allennlp.data.dataset import Batch
from allennlp.data.fields import TextField, ListField, LabelField, SequenceLabelField, ArrayField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import ELMoTokenCharactersIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.vocabulary import Vocabulary
from allennlp.nn.util import get_text_field_mask
from torch.utils.data import Dataset
from dataloaders.box_utils import load_image, resize_image, to_tensor_and_normalize
from dataloaders.mask_utils import make_mask
from dataloaders.bert_field import BertField
import h5py
from copy import deepcopy
from config import VCR_IMAGES_DIR, VCR_ANNOTS_DIR
from functools import partial

GENDER_NEUTRAL_NAMES = ['Casey', 'Riley', 'Jessie', 'Jackie', 'Avery', 'Jaime', 'Peyton', 'Kerry', 'Jody', 'Kendall',
                        'Peyton', 'Skyler', 'Frankie', 'Pat', 'Quinn']

# Here's an example jsonl
# {
# "movie": "3015_CHARLIE_ST_CLOUD",
# "objects": ["person", "person", "person", "car"],
# "interesting_scores": [0],
# "answer_likelihood": "possible",
# "img_fn": "lsmdc_3015_CHARLIE_ST_CLOUD/3015_CHARLIE_ST_CLOUD_00.23.57.935-00.24.00.783@0.jpg",
# "metadata_fn": "lsmdc_3015_CHARLIE_ST_CLOUD/3015_CHARLIE_ST_CLOUD_00.23.57.935-00.24.00.783@0.json",
# "answer_orig": "No she does not",
# "question_orig": "Does 3 feel comfortable?",
# "rationale_orig": "She is standing with her arms crossed and looks disturbed",
# "question": ["Does", [2], "feel", "comfortable", "?"],
# "answer_match_iter": [3, 0, 2, 1],
# "answer_sources": [3287, 0, 10184, 2260],
# "answer_choices": [
#     ["Yes", "because", "the", "person", "sitting", "next", "to", "her", "is", "smiling", "."],
#     ["No", "she", "does", "not", "."],
#     ["Yes", ",", "she", "is", "wearing", "something", "with", "thin", "straps", "."],
#     ["Yes", ",", "she", "is", "cold", "."]],
# "answer_label": 1,
# "rationale_choices": [
#     ["There", "is", "snow", "on", "the", "ground", ",", "and",
#         "she", "is", "wearing", "a", "coat", "and", "hate", "."],
#     ["She", "is", "standing", "with", "her", "arms", "crossed", "and", "looks", "disturbed", "."],
#     ["She", "is", "sitting", "very", "rigidly", "and", "tensely", "on", "the", "edge", "of", "the",
#         "bed", ".", "her", "posture", "is", "not", "relaxed", "and", "her", "face", "looks", "serious", "."],
#     [[2], "is", "laying", "in", "bed", "but", "not", "sleeping", ".",
#         "she", "looks", "sad", "and", "is", "curled", "into", "a", "ball", "."]],
# "rationale_sources": [1921, 0, 9750, 25743],
# "rationale_match_iter": [3, 0, 2, 1],
# "rationale_label": 1,
# "img_id": "train-0",
# "question_number": 0,
# "annot_id": "train-0",
# "match_fold": "train-0",
# "match_index": 0,
# }

def _fix_tokenization(tokenized_sent, bert_embs, old_det_to_new_ind, obj_to_type, token_indexers, pad_ind=-1):
    """
    Turn a detection list into what we want: some text, as well as some tags.
    :param tokenized_sent: Tokenized sentence with detections collapsed to a list.
    :param old_det_to_new_ind: Mapping of the old ID -> new ID (which will be used as the tag)
    :param obj_to_type: [person, person, pottedplant] indexed by the old labels
    :return: tokenized sentence
    """

    new_tokenization_with_tags = []
    for tok in tokenized_sent:
        if isinstance(tok, list):
            for int_name in tok:
                obj_type = obj_to_type[int_name]
                new_ind = old_det_to_new_ind[int_name]
                if new_ind < 0:
                    raise ValueError("Oh no, the new index is negative! that means it's invalid. {} {}".format(
                        tokenized_sent, old_det_to_new_ind
                    ))
                text_to_use = GENDER_NEUTRAL_NAMES[
                    new_ind % len(GENDER_NEUTRAL_NAMES)] if obj_type == 'person' else obj_type
                new_tokenization_with_tags.append((text_to_use, new_ind))
        else:
            new_tokenization_with_tags.append((tok, pad_ind))

    text_field = BertField([Token(x[0]) for x in new_tokenization_with_tags],
                           bert_embs,
                           padding_value=0)
    tags = SequenceLabelField([x[1] for x in new_tokenization_with_tags], text_field)
    return text_field, tags


class VCR(Dataset):
    def __init__(self, split, mode, only_use_relevant_dets=True,
            add_image_as_a_box=True, embs_to_load='bert_da',
            conditioned_answer_choice=None,
            all_answers_for_rationale=False,
            use_precomputed_omcs=False):
        """

        :param split: train, val, or test
        :param mode: answer or rationale
        :param only_use_relevant_dets: True, if we will only use the detections mentioned in the question and answer.
                                       False, if we should use all detections.
        :param add_image_as_a_box:     True to add the image in as an additional 'detection'. It'll go first in the list
                                       of objects.
        :param embs_to_load: Which precomputed embeddings to load.
        :param conditioned_answer_choice: If you're in test mode, the answer labels aren't provided, which could be
                                          a problem for the QA->R task. Pass in 'conditioned_answer_choice=i'
                                          to always condition on the i-th answer.
        :param all_answers_for_rationale: If set, then in rationale mode, the BERT
                                          embeddings are generated for all answers x rationales pairs and not
                                          just for the correct answer. This is irrelevant in answer mode, and
                                          is done by default for test split since we don't know the correct
                                          answer.
        """
        self.split = split
        self.mode = mode
        self.only_use_relevant_dets = only_use_relevant_dets
        print("Only relevant dets" if only_use_relevant_dets else "Using all detections", flush=True)

        self.add_image_as_a_box = add_image_as_a_box

        with open(os.path.join(VCR_ANNOTS_DIR, '{}.jsonl'.format(split)), 'r') as f:
            self.items = [json.loads(s) for s in f]

        if split not in ('test', 'train', 'val'):
            raise ValueError("Split must be in test, train, or val. Supplied {}".format(split))

        if mode not in ('answer', 'rationale', 'joint'):
            raise ValueError("Mode must be answer or rationale")

        if mode == 'rationale':
            self.all_answers_for_rationale = (split == 'test') or all_answers_for_rationale
            if conditioned_answer_choice is not None:
                answer_labels = [conditioned_answer_choice] * len(self.items)
                self.set_answer_labels(answer_labels)
        else:
            self.all_answers_for_rationale = False

        self.token_indexers = {'elmo': ELMoTokenCharactersIndexer()}
        # This makes VCR unpicklable with forkserver start method in
        # multiprocessing which is needed for NCCL based distributed training.
        # vocab here is anyway useless as BERT embeddings are precomputed per
        # dataset instance.
        # self.vocab = Vocabulary()

        with open(os.path.join(os.path.dirname(VCR_ANNOTS_DIR), 'dataloaders', 'cocoontology.json'), 'r') as f:
            coco = json.load(f)
        self.coco_objects = ['__background__'] + [x['name'] for k, x in sorted(coco.items(), key=lambda x: int(x[0]))]
        self.coco_obj_to_ind = {o: i for i, o in enumerate(self.coco_objects)}

        self.embs_to_load = embs_to_load
        if self.all_answers_for_rationale and split != 'test':
            # For test splits, '_all' is not appended to the h5 filename since
            # all embeddings are generated by default.
            self.h5fn = os.path.join(VCR_ANNOTS_DIR, f'{self.embs_to_load}_{self.mode}_{self.split}_all.h5')
        else:
            self.h5fn = os.path.join(VCR_ANNOTS_DIR, f'{self.embs_to_load}_{self.mode}_{self.split}.h5')
        print("Loading embeddings from {}".format(self.h5fn), flush=True)
        self.h5 = None

        self.h5fn_omcs = None
        if use_precomputed_omcs:
            self.h5fn_omcs = os.path.join(VCR_ANNOTS_DIR, 'omcs',
                os.path.basename(self.h5fn).split('.')[0] + '_omcs.h5')
            print("Loading OMCS embeddings from {}".format(self.h5fn_omcs))

    def set_answer_labels(self, answer_labels):
        """
        Updates the ground-truth 'answer_label' in json. This can be used to
        override the ground-truth with predicted answers to compute performance
        of QA->R conditioned on previously predicted answers.

        Note: This must be set right after __init__ before any iteration over
        or indexing into the dataset, and this is only possible if
        all_answers_for_rationale is set to obtain the relevant embeddings.
        """
        # Make sure all_answers_for_rationale is set. This is needed as
        # otherwise we may not have embeddings for (incorrect answer, rationale)
        # pairs.
        assert self.all_answers_for_rationale
        assert len(answer_labels) == len(self.items)
        for i, item in enumerate(self.items):
            label = answer_labels[i]
            assert label < len(item['answer_choices'])
            item['answer_label'] = label

    @property
    def is_train(self):
        return self.split == 'train'

    @classmethod
    def splits(cls, **kwargs):
        """ Helper method to generate splits of the dataset"""
        kwargs_copy = {x:y for x,y in kwargs.items()}
        if 'mode' not in kwargs:
            kwargs_copy['mode'] = 'answer'
        train = cls(split='train', **kwargs_copy)
        val = cls(split='val', **kwargs_copy)
        test = cls(split='test', **kwargs_copy)
        return train, val, test

    @classmethod
    def eval_splits(cls, **kwargs):
        """ Helper method to generate splits of the dataset. Use this for testing, because it will
            condition on everything."""
        for forbidden_key in ['mode', 'split', 'conditioned_answer_choice']:
            if forbidden_key in kwargs:
                raise ValueError(f"don't supply {forbidden_key} to eval_splits()")

        stuff_to_return = [cls(split='test', mode='answer', **kwargs)] + [
            cls(split='test', mode='rationale', conditioned_answer_choice=i, **kwargs) for i in range(4)]
        return tuple(stuff_to_return)

    def __len__(self):
        return len(self.items)

    def _get_dets_to_use(self, item):
        """
        We might want to use fewer detectiosn so lets do so.
        :param item:
        :param question:
        :param answer_choices:
        :return:
        """
        # Load questions and answers
        question = item['question']
        answer_choices = item['{}_choices'.format(self.mode)]

        if self.only_use_relevant_dets:
            dets2use = np.zeros(len(item['objects']), dtype=bool)
            people = np.array([x == 'person' for x in item['objects']], dtype=bool)
            for sent in answer_choices + [question]:
                for possibly_det_list in sent:
                    if isinstance(possibly_det_list, list):
                        for tag in possibly_det_list:
                            if tag >= 0 and tag < len(item['objects']):  # sanity check
                                dets2use[tag] = True
                    elif possibly_det_list.lower() in ('everyone', 'everyones'):
                        dets2use |= people
            if not dets2use.any():
                dets2use |= people
        else:
            dets2use = np.ones(len(item['objects']), dtype=bool)

        # we will use these detections
        dets2use = np.where(dets2use)[0]

        old_det_to_new_ind = np.zeros(len(item['objects']), dtype=np.int32) - 1
        old_det_to_new_ind[dets2use] = np.arange(dets2use.shape[0], dtype=np.int32)

        # If we add the image as an extra box then the 0th will be the image.
        if self.add_image_as_a_box:
            old_det_to_new_ind[dets2use] += 1
        old_det_to_new_ind = old_det_to_new_ind.tolist()
        return dets2use, old_det_to_new_ind

    def __getitem__(self, index):
        item = deepcopy(self.items[index])

        ###################################################################
        # Load questions and answers
        if self.mode == 'rationale':
            item['question'] += item['answer_choices'][item['answer_label']]
        elif self.mode == 'joint':
            item['joint_choices'] = [a + r for a in item['answer_choices'] \
                                            for r in item['rationale_choices']]
            if self.split != 'test':
                item['joint_label'] = item['answer_label'] * 4 + item['rationale_label']
        answer_choices = item['{}_choices'.format(self.mode)]
        dets2use, old_det_to_new_ind = self._get_dets_to_use(item)

        ###################################################################
        # Load in BERT. We'll get contextual representations of the context and the answer choices
        with h5py.File(self.h5fn, 'r') as h5:
            grp_items = {k: np.array(v, dtype=np.float16) for k,v in h5[str(index)].items()}

        omcs_items = None
        if self.h5fn_omcs is not None:
            with h5py.File(self.h5fn_omcs, 'r') as h5_omcs:
                omcs_items = {k: np.array(v, dtype=np.float16) for k,v in h5_omcs[str(index)].items()}

        if self.all_answers_for_rationale:
            # Keys in h5 file are in format [ctx|answer]_rationale[i][j].
            # Pick i based on the answer_label set.
            assert self.mode == 'rationale'
            answer_label = item['answer_label']
            key = f'{self.mode}{answer_label}'
        else:
            # Keys are in format [ctx|answer]_mode[j]
            key = f'{self.mode}'

        instance_dict = {}
        if 'endingonly' not in self.embs_to_load:
            if omcs_items is None:
                ctx_embs = [grp_items[f'ctx_{key}{j}'] for j in range(len(answer_choices))]
            else:
                ctx_embs = [
                    np.hstack([grp_items[f'ctx_{key}{j}'], omcs_items[f'ctx_{key}{j}']])
                    for j in range(len(answer_choices))
                ]
            questions_tokenized, question_tags = zip(*[_fix_tokenization(
                item['question'],
                ctx_embs[j],
                old_det_to_new_ind,
                item['objects'],
                token_indexers=self.token_indexers,
                pad_ind=0 if self.add_image_as_a_box else -1
            ) for j in range(len(answer_choices))])
            instance_dict['question'] = ListField(questions_tokenized)
            instance_dict['question_tags'] = ListField(question_tags)

        if omcs_items is None:
            answer_embs = [grp_items[f'answer_{key}{j}'] for j in range(len(answer_choices))]
        else:
            answer_embs = [
                np.hstack([grp_items[f'answer_{key}{j}'], omcs_items[f'answer_{key}{j}']])
                for j in range(len(answer_choices))
            ]
        answers_tokenized, answer_tags = zip(*[_fix_tokenization(
            answer,
            answer_embs[j],
            old_det_to_new_ind,
            item['objects'],
            token_indexers=self.token_indexers,
            pad_ind=0 if self.add_image_as_a_box else -1
        ) for j, answer in enumerate(answer_choices)])

        instance_dict['answers'] = ListField(answers_tokenized)
        instance_dict['answer_tags'] = ListField(answer_tags)
        if self.split != 'test':
            instance_dict['label'] = LabelField(item['{}_label'.format(self.mode)], skip_indexing=True)
        instance_dict['metadata'] = MetadataField({'annot_id': item['annot_id'], 'ind': index, 'movie': item['movie'],
                                                   'img_fn': item['img_fn'],
                                                   'question_number': item['question_number']})

        ###################################################################
        # Load image now and rescale it. Might have to subtract the mean and whatnot here too.
        image = load_image(os.path.join(VCR_IMAGES_DIR, item['img_fn']))
        image, window, img_scale, padding = resize_image(image, random_pad=self.is_train)
        image = to_tensor_and_normalize(image)
        c, h, w = image.shape

        ###################################################################
        # Load boxes.
        with open(os.path.join(VCR_IMAGES_DIR, item['metadata_fn']), 'r') as f:
            metadata = json.load(f)

        # [nobj, 14, 14]
        segms = np.stack([make_mask(mask_size=14, box=metadata['boxes'][i], polygons_list=metadata['segms'][i])
                          for i in dets2use])

        # Chop off the final dimension, that's the confidence
        boxes = np.array(metadata['boxes'])[dets2use, :-1]
        # Possibly rescale them if necessary
        boxes *= img_scale
        boxes[:, :2] += np.array(padding[:2])[None]
        boxes[:, 2:] += np.array(padding[:2])[None]
        obj_labels = [self.coco_obj_to_ind[item['objects'][i]] for i in dets2use.tolist()]
        if self.add_image_as_a_box:
            boxes = np.row_stack((window, boxes))
            segms = np.concatenate((np.ones((1, 14, 14), dtype=np.float32), segms), 0)
            obj_labels = [self.coco_obj_to_ind['__background__']] + obj_labels

        instance_dict['segms'] = ArrayField(segms, padding_value=0)
        instance_dict['objects'] = ListField([LabelField(x, skip_indexing=True) for x in obj_labels])

        if not np.all((boxes[:, 0] >= 0.) & (boxes[:, 0] < boxes[:, 2])):
            import ipdb
            ipdb.set_trace()
        assert np.all((boxes[:, 1] >= 0.) & (boxes[:, 1] < boxes[:, 3]))
        assert np.all((boxes[:, 2] <= w))
        assert np.all((boxes[:, 3] <= h))
        instance_dict['boxes'] = ArrayField(boxes, padding_value=-1)

        instance = Instance(instance_dict)
        # instance.index_fields(self.vocab)
        return image, instance


class MultiTaskVCR(Dataset):
    '''
    Wrapper around VCR to load data for both Q->A and QA->R tasks for
    multi-task learning.
    '''

    def __init__(self, split, mode, only_use_relevant_dets=True,
            add_image_as_a_box=True, embs_to_load='bert_da',
            conditioned_answer_choice=None,
            all_answers_for_rationale=False,
            use_precomputed_omcs=False):
        assert mode == 'multitask'

        self.answer_ds = VCR(
            split=split,
            mode='answer',
            only_use_relevant_dets=only_use_relevant_dets,
            add_image_as_a_box=add_image_as_a_box,
            embs_to_load=embs_to_load,
            conditioned_answer_choice=conditioned_answer_choice,
            all_answers_for_rationale=all_answers_for_rationale,
            use_precomputed_omcs=use_precomputed_omcs,
        )
        self.rationale_ds = VCR(
            split=split,
            mode='rationale',
            only_use_relevant_dets=only_use_relevant_dets,
            add_image_as_a_box=add_image_as_a_box,
            embs_to_load=embs_to_load,
            conditioned_answer_choice=conditioned_answer_choice,
            all_answers_for_rationale=all_answers_for_rationale,
            use_precomputed_omcs=use_precomputed_omcs,
        )

        assert len(self.answer_ds) == len(self.rationale_ds)

    def set_answer_labels(self, answer_labels):
        self.rationale_ds.set_answer_labels(answer_labels)

    @property
    def is_train(self):
        return self.answer_ds.split == 'train'

    @classmethod
    def splits(cls, **kwargs):
        """ Helper method to generate splits of the dataset"""
        kwargs_copy = {x:y for x,y in kwargs.items()}
        train = cls(split='train', **kwargs_copy)
        val = cls(split='val', **kwargs_copy)
        test = cls(split='test', **kwargs_copy)
        return train, val, test

    @classmethod
    def eval_splits(cls, **kwargs):
        """ Helper method to generate splits of the dataset. Use this for testing, because it will
            condition on everything."""
        for forbidden_key in ['split', 'conditioned_answer_choice']:
            if forbidden_key in kwargs:
                raise ValueError(f"don't supply {forbidden_key} to eval_splits()")
        if 'mode' in kwargs:
            assert kwargs['mode'] == 'multitask'
        else:
            kwargs['mode'] == 'multitask'

        # We also add a dummy conditioned_answer_choice for answer mode since
        # MultiTaskVCR loads both answers and rationales, and test partition
        # doesn't have `answer_label` set otherwise.
        stuff_to_return = [cls(split='test', conditioned_answer_choice=0, **kwargs)] + [
            cls(split='test', conditioned_answer_choice=i, **kwargs) for i in range(4)]
        return tuple(stuff_to_return)

    def __len__(self):
        return len(self.answer_ds)

    def __getitem__(self, index):
        '''
        Fetch corresponding instances for Q->A and Q->AR and merge them into a
        single training instance.
        '''
        image, instance = self.answer_ds[index]
        _, rationale_instance = self.rationale_ds[index]
        assert instance['metadata'].metadata == rationale_instance['metadata'].metadata

        instance.add_field('qa', rationale_instance['question'])
        instance.add_field('qa_tags', rationale_instance['question_tags'])
        instance.add_field('rationales', rationale_instance['answers'])
        instance.add_field('rationale_tags', rationale_instance['answer_tags'])
        instance.add_field('rationale_segms', rationale_instance['segms'])
        instance.add_field('rationale_objects', rationale_instance['objects'])
        instance.add_field('rationale_boxes', rationale_instance['boxes'])
        if 'label' in rationale_instance:
            instance.add_field('rationale_label', rationale_instance['label'])

        return image, instance


def collate_fn(data, to_gpu=False):
    """Creates mini-batch tensors
    """
    images, instances = zip(*data)
    images = torch.stack(images, 0)
    batch = Batch(instances)
    td = batch.as_tensor_dict()
    if 'question' in td:
        td['question_mask'] = get_text_field_mask(td['question'], num_wrapping_dims=1)
        td['question_tags'][td['question_mask'] == 0] = -2  # Padding

    td['answer_mask'] = get_text_field_mask(td['answers'], num_wrapping_dims=1)
    td['answer_tags'][td['answer_mask'] == 0] = -2

    td['box_mask'] = torch.all(td['boxes'] >= 0, -1).long()
    td['images'] = images

    # TODO (viswanath): Single batch for trunk?
    if 'qa' in td:
        td['qa_mask'] = get_text_field_mask(td['qa'], num_wrapping_dims=1)
        td['qa_tags'][td['qa_mask'] == 0] = -2  # Padding
    if 'rationales' in td:
        td['rationale_mask'] = get_text_field_mask(td['rationales'], num_wrapping_dims=1)
        td['rationale_tags'][td['rationale_mask'] == 0] = -2
    if 'rationale_boxes' in td:
        td['rationale_box_mask'] = torch.all(td['rationale_boxes'] >= 0, -1).long()

    if to_gpu:
        for k in td:
            if k != 'metadata':
                td[k] = {k2: v.cuda(async=True)
                        for k2, v in td[k].items()} if isinstance(td[k], dict) else td[k].cuda(async=True)
    # # No nested dicts
    # for k in sorted(td.keys()):
    #     if isinstance(td[k], dict):
    #         for k2 in sorted(td[k].keys()):
    #             td['{}_{}'.format(k, k2)] = td[k].pop(k2)
    #         td.pop(k)

    return td


def vcr_splits(**kwargs):
    if 'mode' in kwargs and kwargs['mode'] == 'multitask':
        return MultiTaskVCR.splits(**kwargs)
    else:
        return VCR.splits(**kwargs)


def vcr_eval_splits(**kwargs):
    if 'mode' in kwargs and kwargs['mode'] == 'multitask':
        return MultiTaskVCR.eval_splits(**kwargs)
    else:
        return VCR.eval_splits(**kwargs)


class VCRLoader(torch.utils.data.DataLoader):
    """
    Iterates through the data, filtering out None,
     but also loads everything as a (cuda) variable
    """

    @classmethod
    def from_dataset(cls, data, sampler=None, batch_size=3, num_workers=6,
                     num_gpus=3, **kwargs):
        loader = cls(
            dataset=data,
            batch_size=batch_size * num_gpus,
            sampler=sampler,
            shuffle=(data.is_train and sampler is None),
            num_workers=num_workers,
            collate_fn=partial(collate_fn, to_gpu=(num_workers == 0)),
            drop_last=data.is_train,
            pin_memory=False,
            **kwargs,
        )
        return loader
