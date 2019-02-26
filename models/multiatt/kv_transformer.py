from typing import Dict, List, Any

import faiss
import math
import logging
import torch
import torch.nn.functional as F
import torch.nn.parallel
import numpy as np
import os
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder, FeedForward, InputVariationalDropout, TimeDistributed
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.modules.matrix_attention import BilinearMatrixAttention
from utils.detector import SimpleDetector
from allennlp.nn.util import masked_softmax, weighted_sum, replace_masked_values

from config import VCR_ANNOTS_DIR

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)


@Model.register("KeyValueTransformer")
class KeyValueTransformer(Model):
    def __init__(self, dim, num_heads=8, num_steps=4):
        vocab = Vocabulary()
        super(KeyValueTransformer, self).__init__(vocab)

        assert dim % num_heads == 0
        self.dim = dim
        self.dim_h = dim // num_heads
        self.num_heads = num_heads
        self.num_steps = num_steps

        self.Wa = torch.nn.Linear(self.dim, self.dim)
        self.Wo = torch.nn.Linear(self.dim, self.dim)

    def multi_head_attention(self, Q, K, V, W):
        """
        Q: [batch_size x k]
        K: [batch_size x n x k]
        V: [batch_size x n x k]
        """
        Q = Q[:, None, :]
        # -- [batch_size, 1+n+n, dim]
        M = torch.cat([Q, K, V], dim=1)

        # -- [batch_size, 1+n+n, dim]
        Mt = W(M)
        # -- [batch_size, 1+n+n, num_heads, dim_h]
        Mt = Mt.contiguous().view(
                Mt.shape[0], Mt.shape[1], self.num_heads, self.dim_h)
        # -- [batch_size, num_heads, dim_h]
        Qt = Mt[:, 0, :, :].squeeze()
        # -- [batch_size, n, num_heads, dim_h]
        Kt, Vt = Mt[:, 1:, :, :].chunk(2, dim=1)

        # -- [batch_size, num_heads, n]
        alpha = F.softmax(torch.einsum('bhk,bnhk->bhn', (Qt, Kt)) / math.sqrt(self.dim_h), dim=2)
        # -- [batch_size, num_heads, dim_h]
        qt1 = torch.einsum("bhn,bnhw->bhw", [alpha, Vt])
        # -- [batch_size, dim]
        qt1 = qt1.view(qt1.shape[0], self.dim)
        # -- [batch_size, n]
        alpha = alpha.mean(1)
        return qt1, alpha

    def forward(self, q_rep, a_rep, o_rep):
        qt = q_rep
        o_key, o_val = o_rep.chunk(2, dim=2)
        a_key, a_val = a_rep.chunk(2, dim=2)
        # TODO: add and norm
        for i in range(self.num_steps):
            qt, o_alpha = self.multi_head_attention(qt, o_key, o_val, self.Wo)
            qt, a_alpha = self.multi_head_attention(qt, a_key, a_val, self.Wa)
        return qt, a_alpha, o_alpha
