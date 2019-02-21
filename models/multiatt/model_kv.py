from typing import Dict, List, Any

import faiss
import logging
import torch
import torch.nn.functional as F
import torch.nn.parallel
import numpy as np
import os
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder, FeedForward, InputVariationalDropout, TimeDistributed
from allennlp.training.metrics import CategoricalAccuracy, BooleanAccuracy
from allennlp.modules.matrix_attention import BilinearMatrixAttention
from utils.detector import SimpleDetector
from allennlp.nn.util import masked_softmax, weighted_sum, replace_masked_values
from allennlp.nn import InitializerApplicator

from config import VCR_ANNOTS_DIR
from data.omcs.extract_omcs_features import load_omcs_embeddings, normalize_embedding
from models.multiatt.kv_transformer import KeyValueTransformer

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)


@Model.register("KeyValueAttentionTrunk")
class KeyValueAttentionTrunk(Model):
    def __init__(
        self,
        span_encoder: Seq2SeqEncoder,
        input_dropout: float = 0.3,
        class_embs: bool=True,
        initializer: InitializerApplicator = InitializerApplicator(),
        learned_omcs: dict = {},
    ):
        # VCR dataset becomes unpicklable due to VCR.vocab, but we don't need
        # to pass in vocab from the dataset anyway as the BERT embeddings are
        # pretrained and stored in h5 files per dataset instance. Just pass
        # a dummy vocab instance for init.
        vocab = Vocabulary()
        super(KeyValueAttentionTrunk, self).__init__(vocab)

        self.detector = SimpleDetector(pretrained=True, average_pool=True, semantic=class_embs, final_dim=512)

        self.rnn_input_dropout = TimeDistributed(InputVariationalDropout(input_dropout)) if input_dropout > 0 else None

        self.span_encoder = TimeDistributed(span_encoder)
        span_dim = span_encoder.get_output_dim()

        self.question_mlp = torch.nn.Sequential(
            # 2 (bidirectional) * 4 (num_answers) * dim -> dim
            torch.nn.Linear(8 * span_dim, span_dim),
            torch.nn.Tanh(),
        )
        self.answer_mlp = torch.nn.Sequential(
            # 2 (bidirectional) * dim -> 2 (key-value) * dim
            torch.nn.Linear(2 * span_dim, 2 * span_dim),
            torch.nn.Tanh(),
        )
        self.obj_mlp = torch.nn.Sequential(
            # obj_dim -> 2 (key-value) * dim
            torch.nn.Linear(self.detector.final_dim, 2 * span_dim),
            torch.nn.Tanh(),
        )

        self.span_attention = BilinearMatrixAttention(
            matrix_1_dim=span_encoder.get_output_dim(),
            matrix_2_dim=span_encoder.get_output_dim(),
        )

        self.obj_attention = BilinearMatrixAttention(
            matrix_1_dim=span_encoder.get_output_dim(),
            matrix_2_dim=self.detector.final_dim,
        )

        self.kv_transformer = KeyValueTransformer(
            dim=span_dim,
            num_heads=8,
            num_steps=4,
        )

        self.omcs_index = None
        if learned_omcs.get('enabled', False):
            use_sentence_embs = learned_omcs.get('use_sentence_embeddings', True)
            omcs_embs, self.omcs_index = self.load_omcs(use_sentence_embs)
            # Let's replicate the OMCS embeddings to each device to attend over them
            # after FAISS lookup. We could also do faiss.search_and_reconstruct, but
            # that prevents us from using quantized indices for faster search which
            # we might need to.
            self.register_buffer('omcs_embs', omcs_embs)
            self.omcs_mlp = torch.nn.Sequential(
                torch.nn.Linear(768, self.omcs_index.d),
            )
            self.k = learned_omcs.get('max_neighbors', 5)
            self.similarity_thresh = learned_omcs.get('similarity_thresh', 0.0)

        initializer(self)

    def _collect_obj_reps(self, span_tags, object_reps):
        """
        Collect span-level object representations
        :param span_tags: [batch_size, ..leading_dims.., L]
        :param object_reps: [batch_size, max_num_objs_per_batch, obj_dim]
        :return:
        """
        span_tags_fixed = torch.clamp(span_tags, min=0)  # In case there were masked values here
        row_id = span_tags_fixed.new_zeros(span_tags_fixed.shape)
        row_id_broadcaster = torch.arange(0, row_id.shape[0], step=1, device=row_id.device)[:, None]

        # Add extra diminsions to the row broadcaster so it matches row_id
        leading_dims = len(span_tags.shape) - 2
        for i in range(leading_dims):
            row_id_broadcaster = row_id_broadcaster[..., None]
        row_id += row_id_broadcaster
        return object_reps[row_id.view(-1), span_tags_fixed.view(-1)].view(*span_tags_fixed.shape, -1)

    def embed_span(self, span, span_tags, span_mask, object_reps):
        """
        :param span: Thing that will get embed and turned into [batch_size, ..leading_dims.., L, word_dim]
        :param span_tags: [batch_size, ..leading_dims.., L]
        :param object_reps: [batch_size, max_num_objs_per_batch, obj_dim]
        :param span_mask: [batch_size, ..leading_dims.., span_mask
        :return:
        """
        if self.omcs_index is not None:
            # TODO (viswanath): Batch faiss lookup for question and answer embeddings?
            vcr_omcs_embs = self.attended_omcs_embeddings(span['bert'])
            bert = torch.cat((span['bert'], vcr_omcs_embs), -1)
        else:
            bert = span['bert']

        retrieved_feats = self._collect_obj_reps(span_tags, object_reps)
        span_rep = torch.cat((bert, retrieved_feats), -1)
        # Add recurrent dropout here
        if self.rnn_input_dropout:
            span_rep = self.rnn_input_dropout(span_rep)

        # [batch, num_answers, seq_length, dim]
        span_rep = self.span_encoder(span_rep, span_mask)

        # Just take the first and last time steps and obtain a single embedding
        # per question or answer.
        batch_size, num_answers, _, _ = span_rep.shape
        span_rep = span_rep[:, :, [0, -1], :]
        span_rep = span_rep.contiguous().view(batch_size, num_answers, -1)
        return span_rep

    def load_omcs(self, use_sentence_embs=True):
        omcs_h5_file = os.path.join(VCR_ANNOTS_DIR, 'omcs', 'bert_da_omcs.h5')
        # Embeddings are stored as float16, but faiss requires float32
        _, sentence_embs, word_embs = load_omcs_embeddings(omcs_h5_file, dtype=np.float32)

        if use_sentence_embs:
            embs = np.vstack(sentence_embs)
            index_file = 'bert_da_omcs_sentences.faissindex'
        else:
            embs = np.vstack(word_embs)
            index_file = 'bert_da_omcs_words.faissindex'
        index_file = os.path.join(VCR_ANNOTS_DIR, 'omcs', index_file)

        index = faiss.read_index(index_file)
        assert len(embs) == index.ntotal
        assert embs.shape[1] == index.d
        LOG.info('Loaded faiss index with OMCS embeddings from {}, ntotal={}'.format(
            index_file, index.ntotal))

        self.co = faiss.GpuMultipleClonerOptions()
        self.co.shard = False  # Replica mode (dataparallel) instead of shard mode
        index = faiss.index_cpu_to_all_gpus(index, self.co)
        return torch.from_numpy(embs), index

    def normalize_embedding(self, embs):
        return embs / torch.norm(embs, dim=1).view(-1, 1)

    def attended_omcs_embeddings(self, vcr_embs):
        projected_embs = self.normalize_embedding(
            self.omcs_mlp(vcr_embs).view(-1, vcr_embs.shape[-1])
        )
        n, d = projected_embs.size()
        device = projected_embs.get_device()

        def swig_ptr_from_FloatTensor(x):
            assert x.is_contiguous()
            assert x.dtype == torch.float32
            return faiss.cast_integer_to_float_ptr(x.storage().data_ptr())

        def swig_ptr_from_LongTensor(x):
            assert x.is_contiguous()
            assert x.dtype == torch.int64, 'dtype=%s' % x.dtype
            return faiss.cast_integer_to_long_ptr(x.storage().data_ptr())

        D = torch.empty((n, self.k), dtype=torch.float32, device=device)
        I = torch.empty((n, self.k), dtype=torch.int64, device=device)
        torch.cuda.synchronize()
        self.omcs_index.at(device).search_c(
            n,
            swig_ptr_from_FloatTensor(projected_embs),
            self.k,
            swig_ptr_from_FloatTensor(D),
            swig_ptr_from_LongTensor(I),
        )
        torch.cuda.synchronize()

        # Compute softmax of similarity scores.
        # Only use those with cosine similarity scores above thresh.
        # TODO (viswanath): Use scaled-dot-product attn w/o normalization?
        mask = (D >= self.similarity_thresh)
        attention_wts = masked_softmax(D, mask)

        # Fetch the nearest found embeddings and then apply attention
        # using the computed weights.
        nearest_omcs_embs = self.omcs_embs[I]  # (n, k, d)
        attended_omcs_embs = torch.einsum('nk,nkd->nd',
                                          (attention_wts, nearest_omcs_embs))

        # Reshape to match original vcr_embs
        return attended_omcs_embs.view(*vcr_embs.shape[:-1], -1)

    def forward(self,
                images: torch.Tensor,
                objects: torch.LongTensor,
                segms: torch.Tensor,
                boxes: torch.Tensor,
                box_mask: torch.LongTensor,
                question: Dict[str, torch.Tensor],
                question_tags: torch.LongTensor,
                question_mask: torch.LongTensor,
                answers: Dict[str, torch.Tensor],
                answer_tags: torch.LongTensor,
                answer_mask: torch.LongTensor,
                metadata: List[Dict[str, Any]] = None,
                label: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        """
        :param images: [batch_size, 3, im_height, im_width]
        :param objects: [batch_size, max_num_objects] Padded objects
        :param boxes:  [batch_size, max_num_objects, 4] Padded boxes
        :param box_mask: [batch_size, max_num_objects] Mask for whether or not each box is OK
        :param question: AllenNLP representation of the question. [batch_size, num_answers, seq_length]
        :param question_tags: A detection label for each item in the Q [batch_size, num_answers, seq_length]
        :param question_mask: Mask for the Q [batch_size, num_answers, seq_length]
        :param answers: AllenNLP representation of the answer. [batch_size, num_answers, seq_length]
        :param answer_tags: A detection label for each item in the A [batch_size, num_answers, seq_length]
        :param answer_mask: Mask for the As [batch_size, num_answers, seq_length]
        :param metadata: Ignore, this is about which dataset item we're on
        :param label: Optional, which item is valid
        """
        # Trim off boxes that are too long. this is an issue b/c dataparallel, it'll pad more zeros that are
        # not needed
        max_len = int(box_mask.sum(1).max().item())
        objects = objects[:, :max_len]
        box_mask = box_mask[:, :max_len]
        boxes = boxes[:, :max_len]
        segms = segms[:, :max_len]

        for tag_type, the_tags in (('question', question_tags), ('answer', answer_tags)):
            if int(the_tags.max()) > max_len:
                raise ValueError("Oh no! {}_tags has maximum of {} but objects is of dim {}. Values are\n{}".format(
                    tag_type, int(the_tags.max()), objects.shape, the_tags
                ))

        obj_reps = self.detector(images=images, boxes=boxes, box_mask=box_mask, classes=objects, segms=segms)

        # Now get the question representations
        q_rep = self.embed_span(question, question_tags, question_mask, obj_reps['obj_reps'])
        # [batch, num_answers, dim] -> [batch, num_answers * dim]
        q_rep = q_rep.contiguous().view(q_rep.shape[0], -1)
        q_rep = self.question_mlp(q_rep)

        a_rep = self.embed_span(answers, answer_tags, answer_mask, obj_reps['obj_reps'])
        a_rep = self.answer_mlp(a_rep)

        o_rep = self.obj_mlp(obj_reps['obj_reps'])

        qt, a_alpha, o_alpha = self.kv_transformer(q_rep, a_rep, o_rep)

        output_dict = {
            'pooled_rep': qt,
            'probs': a_alpha,
            'cnn_regularization_loss': obj_reps['cnn_regularization_loss'],
        }
        return output_dict


@Model.register("KeyValueAttention")
class KeyValueAttention(Model):
    def __init__(
        self,
        span_encoder: Seq2SeqEncoder,
        input_dropout: float = 0.3,
        class_embs: bool=True,
        initializer: InitializerApplicator = InitializerApplicator(),
        learned_omcs: dict = {},
    ):
        # VCR dataset becomes unpicklable due to VCR.vocab, but we don't need
        # to pass in vocab from the dataset anyway as the BERT embeddings are
        # pretrained and stored in h5 files per dataset instance. Just pass
        # a dummy vocab instance for init.
        vocab = Vocabulary()
        super(KeyValueAttention, self).__init__(vocab)

        self.trunk = KeyValueAttentionTrunk(
            span_encoder,
            input_dropout,
            class_embs,
            initializer,
            learned_omcs,
        )

        self._accuracy = BooleanAccuracy()
        self._loss = torch.nn.NLLLoss()
        initializer(self)

    def forward(self,
                images: torch.Tensor,
                objects: torch.LongTensor,
                segms: torch.Tensor,
                boxes: torch.Tensor,
                box_mask: torch.LongTensor,
                question: Dict[str, torch.Tensor],
                question_tags: torch.LongTensor,
                question_mask: torch.LongTensor,
                answers: Dict[str, torch.Tensor],
                answer_tags: torch.LongTensor,
                answer_mask: torch.LongTensor,
                metadata: List[Dict[str, Any]] = None,
                label: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        features = self.trunk.forward(
            images,
            objects,
            segms,
            boxes,
            box_mask,
            question,
            question_tags,
            question_mask,
            answers,
            answer_tags,
            answer_mask,
        )

        probs = features['probs']
        output_dict = {
            'label_probs': features['probs'],
            'cnn_regularization_loss': features['cnn_regularization_loss'],
            # Uncomment to visualize attention, if you want
            # 'qa_attention_weights': features['qa_attention_weights'],
            # 'atoo_attention_weights': features['atoo_attention_weights'],
        }

        if label is not None:
            # We use NLLLoss as don't have the logits.
            # Need to take log(softmax_probs) first.
            loss = self._loss(torch.log(probs), label.long().view(-1))
            self._accuracy(probs.argmax(dim=1), label)
            output_dict["loss"] = loss[None]

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {'accuracy': self._accuracy.get_metric(reset)}
