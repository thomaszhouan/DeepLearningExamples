import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from seq2seq.models.dfxp import Linear_q, ActivationQuantizer, WeightQuantizer, GradientQuantizer


class BahdanauAttention(nn.Module):
    """
    Bahdanau Attention (https://arxiv.org/abs/1409.0473)
    Implementation is very similar to tf.contrib.seq2seq.BahdanauAttention
    """
    def __init__(self, bits, query_size, key_size, num_units, normalize=False,
                 batch_first=False, init_weight=0.1):
        """
        Constructor for the BahdanauAttention.

        :param bits: number of DFXP bits
        :param query_size: feature dimension for query
        :param key_size: feature dimension for keys
        :param num_units: internal feature dimension
        :param normalize: whether to normalize energy term
        :param batch_first: if True batch size is the 1st dimension, if False
            the sequence is first and batch size is second
        :param init_weight: range for uniform initializer used to initialize
            Linear key and query transform layers and linear_att vector
        """
        super(BahdanauAttention, self).__init__()

        self.normalize = normalize
        self.batch_first = batch_first
        self.num_units = num_units

        self.linear_q = Linear_q(bits, query_size, num_units, bias=False)
        self.linear_k = Linear_q(bits, key_size, num_units, bias=False)
        nn.init.uniform_(self.linear_q.weight.data, -init_weight, init_weight)
        nn.init.uniform_(self.linear_k.weight.data, -init_weight, init_weight)

        self.linear_att = Parameter(torch.Tensor(num_units))

        self.mask = None

        if self.normalize:
            self.normalize_scalar = Parameter(torch.Tensor(1))
            self.normalize_bias = Parameter(torch.Tensor(num_units))
        else:
            self.register_parameter('normalize_scalar', None)
            self.register_parameter('normalize_bias', None)

        self.reset_parameters(init_weight)

        # forward quantizers
        self.sum_tanh_q = ActivationQuantizer(bits)
        self.linear_att_q = WeightQuantizer(bits, self.linear_att)
        self.score_q = ActivationQuantizer(bits)
        self.key_q = ActivationQuantizer(bits)

        # backward quantizers
        self.sum_tanh_grad_q = GradientQuantizer(bits)
        self.score_grad_q = GradientQuantizer(bits)
        self.context_grad_q = GradientQuantizer(bits)

    def reset_parameters(self, init_weight):
        """
        Sets initial random values for trainable parameters.
        """
        stdv = 1. / math.sqrt(self.num_units)
        self.linear_att.data.uniform_(-init_weight, init_weight)

        if self.normalize:
            self.normalize_scalar.data.fill_(stdv)
            self.normalize_bias.data.zero_()

    def set_mask(self, context_len, context):
        """
        sets self.mask which is applied before softmax
        ones for inactive context fields, zeros for active context fields

        :param context_len: b
        :param context: if batch_first: (b x t_k x n) else: (t_k x b x n)

        self.mask: (b x t_k)
        """

        if self.batch_first:
            max_len = context.size(1)
        else:
            max_len = context.size(0)

        indices = torch.arange(0, max_len, dtype=torch.int64,
                               device=context.device)
        self.mask = indices >= (context_len.unsqueeze(1))

    def calc_score(self, att_query, att_keys):
        """
        Calculate Bahdanau score

        :param att_query: b x t_q x n
        :param att_keys: b x t_k x n

        returns: b x t_q x t_k scores
        """

        b, t_k, n = att_keys.size()
        t_q = att_query.size(1)

        att_query = att_query.unsqueeze(2).expand(b, t_q, t_k, n)
        att_keys = att_keys.unsqueeze(1).expand(b, t_q, t_k, n)
        sum_qk = att_query + att_keys

        if self.normalize:
            sum_qk = sum_qk + self.normalize_bias
            linear_att = self.linear_att / self.linear_att.norm()
            linear_att = linear_att * self.normalize_scalar
        else:
            linear_att = self.linear_att

        linear_att = self.linear_att_q(linear_att)
        sum_tanh = torch.tanh(sum_qk)
        sum_tanh = self.sum_tanh_q(self.sum_tanh_grad_q(sum_tanh))
        out = self.score_grad_q(sum_tanh.matmul(linear_att))

        # out = torch.tanh(sum_qk).matmul(linear_att)
        return out

    def forward(self, query, keys):
        """

        :param query: if batch_first: (b x t_q x n) else: (t_q x b x n)
        :param keys: if batch_first: (b x t_k x n) else (t_k x b x n)

        :returns: (context, scores_normalized)
        context: if batch_first: (b x t_q x n) else (t_q x b x n)
        scores_normalized: if batch_first (b x t_q x t_k) else (t_q x b x t_k)
        """

        # first dim of keys and query has to be 'batch', it's needed for bmm
        if not self.batch_first:
            keys = keys.transpose(0, 1)
            if query.dim() == 3:
                query = query.transpose(0, 1)

        if query.dim() == 2:
            single_query = True
            query = query.unsqueeze(1)
        else:
            single_query = False

        b = query.size(0)
        t_k = keys.size(1)
        t_q = query.size(1)

        # FC layers to transform query and key
        processed_query = self.linear_q(query)
        processed_key = self.linear_k(keys)

        # scores: (b x t_q x t_k)
        scores = self.calc_score(processed_query, processed_key)

        if self.mask is not None:
            mask = self.mask.unsqueeze(1).expand(b, t_q, t_k)
            # I can't use -INF because of overflow check in pytorch
            scores.data.masked_fill_(mask, -65504.0)

        # Normalize the scores, softmax over t_k
        scores_normalized = F.softmax(scores, dim=-1)

        # forward quantize
        scores_normalized = self.score_q(scores_normalized)
        keys = self.key_q(keys)

        # Calculate the weighted average of the attention inputs according to
        # the scores
        # context: (b x t_q x n)
        context = torch.bmm(scores_normalized, keys)

        # backward quantize
        context = self.context_grad_q(context)

        if single_query:
            context = context.squeeze(1)
            scores_normalized = scores_normalized.squeeze(1)
        elif not self.batch_first:
            context = context.transpose(0, 1)
            scores_normalized = scores_normalized.transpose(0, 1)

        return context, scores_normalized
