# coding=utf-8


import numpy as np

from models.model_base import Embedding_layer
from models.model_base import gen_mask
from models.model_base import normalize
from models.model_base import kernel_mu
from models.model_base import kernel_sigma


class KNRM:

    def __init__(self, emb, config):
        self.emb = emb
        self.config = config
        self.init_param()

    def init_param(self):
        self.mus = np.array(kernel_mu(self.config['n_kernels']))[np.newaxis, np.newaxis, np.newaxis, :]
        self.sigmas = np.array(kernel_sigma(self.config['n_kernels']))[np.newaxis, np.newaxis, np.newaxis, :]

    def build(self, q, t):
        embedding_layer = Embedding_layer(self.emb)
        q_emb = embedding_layer(q)
        t_emb = embedding_layer(t)
        qt_mask = gen_mask(q, t)[:, :, :, np.newaxis]
        q_emb_norm = normalize(q_emb)
        t_emb_norm = normalize(t_emb)
        log_pooling_sum = self.interaction_matrix(q_emb_norm, t_emb_norm, qt_mask)
        return log_pooling_sum

    def interaction_matrix(self, q_emb_norm, t_emb_norm, qt_mask, q_weight=None):
        match_matrix = np.matmul(q_emb_norm, np.transpose(t_emb_norm, (0, 2, 1)))[:, :, :, np.newaxis]
        kernel_pooling = np.exp(-((match_matrix - self.mus) ** 2) / (2 * (self.sigmas ** 2)))
        # kernel_pooling_row --> batch_size * query_length * title_length * n_kernel
        kernel_pooling_row = kernel_pooling * qt_mask
        # pooling_row_sum --> batch_size * query_length * n_kernel
        pooling_row_sum = np.sum(kernel_pooling_row, 2)
        # kernel_pooling --> batch_size * query_length * n_kernel
        log_pooling = np.log(np.clip(pooling_row_sum, a_min=1e-10, a_max=np.inf)) * 0.01  # scale down the data
        log_pooling_weight = log_pooling * q_weight if q_weight is not None else log_pooling
        log_pooling_sum = np.sum(log_pooling_weight, 1)
        return log_pooling_sum

    def __call__(self, q, t):
        return self.build(q, t)
