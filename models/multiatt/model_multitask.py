from typing import Dict, List, Any

import torch
import torch.nn.functional as F
import torch.nn.parallel
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder, FeedForward, InputVariationalDropout, TimeDistributed
from allennlp.training.metrics import CategoricalAccuracy, BooleanAccuracy
from allennlp.modules.matrix_attention import BilinearMatrixAttention
from utils.detector import SimpleDetector
from allennlp.nn.util import masked_softmax, weighted_sum, replace_masked_values
from allennlp.nn import InitializerApplicator

import models
from models.multiatt.model import AttentionQATrunk


@Model.register("MultiTaskMultiHopAttentionQA")
class MultiTaskAttentionQA(Model):
    def __init__(self,
                 span_encoder: Seq2SeqEncoder,
                 reasoning_encoder: Seq2SeqEncoder,
                 input_dropout: float = 0.3,
                 hidden_dim_maxpool: int = 1024,
                 class_embs: bool=True,
                 reasoning_use_obj: bool=True,
                 reasoning_use_answer: bool=True,
                 reasoning_use_question: bool=True,
                 pool_reasoning: bool = True,
                 pool_answer: bool = True,
                 pool_question: bool = False,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 ):
        # VCR dataset becomes unpicklable due to VCR.vocab, but we don't need
        # to pass in vocab from the dataset anyway as the BERT embeddings are
        # pretrained and stored in h5 files per dataset instance. Just pass
        # a dummy vocab instance for init.
        vocab = Vocabulary()
        super(MultiTaskAttentionQA, self).__init__(vocab)

        self.trunk = AttentionQATrunk(
            span_encoder,
            reasoning_encoder,
            input_dropout,
            hidden_dim_maxpool,
            class_embs,
            reasoning_use_obj,
            reasoning_use_answer,
            reasoning_use_question,
            pool_reasoning,
            pool_answer,
            pool_question,
            initializer,
        )

        self.answer_mlp = torch.nn.Sequential(
            torch.nn.Dropout(input_dropout, inplace=False),
            torch.nn.Linear(self.trunk.output_dim, hidden_dim_maxpool),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(input_dropout, inplace=False),
            torch.nn.Linear(hidden_dim_maxpool, 1),
        )
        self.rationale_mlp = torch.nn.Sequential(
            torch.nn.Dropout(input_dropout, inplace=False),
            torch.nn.Linear(self.trunk.output_dim, hidden_dim_maxpool),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(input_dropout, inplace=False),
            torch.nn.Linear(hidden_dim_maxpool, 1),
        )
        self._answer_accuracy = CategoricalAccuracy()
        self._rationale_accuracy = CategoricalAccuracy()
        self._multitask_accuracy = BooleanAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()
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
                rationale_objects: torch.LongTensor,
                rationale_segms: torch.Tensor,
                rationale_boxes: torch.Tensor,
                rationale_box_mask: torch.LongTensor,
                qa: Dict[str, torch.Tensor],
                qa_tags: torch.LongTensor,
                qa_mask: torch.LongTensor,
                rationales: Dict[str, torch.Tensor],
                rationale_tags: torch.LongTensor,
                rationale_mask: torch.LongTensor,
                metadata: List[Dict[str, Any]] = None,
                label: torch.LongTensor = None,
                rationale_label: torch.LongTensor = None,
        ) -> Dict[str, torch.Tensor]:
        answer_features = self.trunk.forward(
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
        rationale_features = self.trunk.forward(
            images,
            rationale_objects,
            rationale_segms,
            rationale_boxes,
            rationale_box_mask,
            qa,
            qa_tags,
            qa_mask,
            rationales,
            rationale_tags,
            rationale_mask,
        )

        answer_logits = self.answer_mlp(answer_features['pooled_rep']).squeeze(2)
        answer_probs = F.softmax(answer_logits, dim=-1)

        rationale_logits = self.rationale_mlp(rationale_features['pooled_rep']).squeeze(2)
        rationale_probs = F.softmax(rationale_logits, dim=-1)

        cnn_reg_loss = answer_features['cnn_regularization_loss'] + \
                rationale_features['cnn_regularization_loss']
        output_dict = {
            'label_logits': answer_logits,
            'label_probs': answer_probs,
            'rationale_logits': rationale_logits,
            'rationale_probs': rationale_probs,
            'cnn_regularization_loss': cnn_reg_loss,
        }

        answer_loss = 0.0
        if label is not None:
            self._answer_accuracy(answer_logits, label)
            loss = self._loss(answer_logits, label.long().view(-1))
            answer_loss = loss[None]

        rationale_loss = 0.0
        if rationale_label is not None:
            self._rationale_accuracy(rationale_logits, rationale_label)
            loss = self._loss(rationale_logits, rationale_label.long().view(-1))
            rationale_loss = loss[None]

        # Track multi-task/joint accuracy for Q->A and QA->R
        if label is not None and rationale_label is not None:
            answer_pred = answer_probs.argmax(dim=1)
            rationale_pred = rationale_probs.argmax(dim=1)
            self._multitask_accuracy(
                torch.stack((answer_pred, rationale_pred), dim=1),
                torch.stack((label, rationale_label), dim=1),
            )

        output_dict['loss'] = answer_loss + rationale_loss
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
            'accuracy': self._multitask_accuracy.get_metric(reset),
            'answer_accuracy': self._answer_accuracy.get_metric(reset),
            'rationale_accuracy': self._rationale_accuracy.get_metric(reset),
        }
