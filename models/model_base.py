# -*- coding:utf-8 -*-


import gensim
import numpy as np

from utils.normalizer import row_norm
from utils.normalizer import handle_zeros_in_scale


def kernel_mu(n_kernels, manual=False):
    """
    get mu for each guassian kernel, Mu is the middele of each bin
    :param n_kernels: number of kernels( including exact match). first one is the exact match
    :return: mus, a list of mu """
    mus = [1]  # exact match
    if n_kernels == 1:
        return mus
    bin_step = (1 - (-1)) / (n_kernels - 1)  # score from [-1, 1]
    mus.append(1 - bin_step / 2)  # the margain mu value
    for k in range(1, n_kernels - 1):
        mus.append(mus[k] - bin_step)
    if manual:
        return [1, 0.95, 0.90, 0.85, 0.8, 0.6, 0.4, 0.2, 0, -0.2, -0.4, -0.6, -0.80, -0.85, -0.90, -0.95]
    else:
        return mus


def kernel_sigma(n_kernels):
    """
    get sigmas for each guassian kernel.
    :param n_kernels: number of kernels(including the exact match)
    :return: sigmas, a list of sigma
    """
    sigmas = [0.001]  # exact match small variance means exact match ?
    if n_kernels == 1:
        return sigmas
    return sigmas + [0.1] * (n_kernels - 1)


def gen_mask(q, t):
    mask = np.zeros((len(q), len(q[0]), len(t[0])))
    for b in range(len(q)):
        for r in range(np.count_nonzero(q[b])):
            mask[b, r, :np.count_nonzero(t[b])] = 1
    return mask


def normalize(seq_emb):
    if isinstance(seq_emb, np.ndarray):
        seq_vector = np.empty_like(seq_emb)
        if seq_emb.ndim == 3:
            for r in range(seq_emb.shape[0]):
                seq_vector[r] = norm(seq_emb[r])
        elif seq_emb.ndim == 2:
            seq_vector = norm(seq_emb)
        else:
            NotImplementedError("The input data dim is not support")
        return seq_vector
    else:
        NotImplementedError("The input data Type is not support")


def norm(X, norm='l2', axis=1, return_norm=False):
    if norm == 'l2':
        # import ipdb; ipdb.set_trace()
        norms = row_norm(X)
    else:
        raise NotImplementedError(f"norm is not implemented")
    norms = handle_zeros_in_scale(norms, copy=False)
    X /= norms[:, np.newaxis]
    if return_norm:
        return X, norms
    else:
        return X


def pooling_emb(vector, n_gram=1):
    if n_gram == 1:
        return vector
    vec = np.zeros((vector.shape[0], vector.shape[1] - (n_gram - 1), vector.shape[2]))
    for idx in range(vector.shape[0]):
        for index in range(vector.shape[1] - n_gram + 1):
            vec[idx, index] = np.mean(vector[idx, index: index + n_gram], 0)
    return vec


def remask(mask, n_gram=1):
    return mask[:, n_gram - 1:]


class Embedding_layer:

    def __init__(self, emb):
        self.emb = emb
        if isinstance(emb.w2v, gensim.models.keyedvectors.Word2VecKeyedVectors):
            self.embedding = emb.w2v.vectors
        else:
            raise NotImplementedError

    def __call__(self, batch):
        if isinstance(batch, np.ndarray):
            #vectors = np.zeros((len(batch), len(batch[0]), self.emb.w2v.vector_size)
            vec = [self.get_vector(seq_id) for seq_id in batch]
            vectors = np.stack(vec, axis=0)
            return vectors
        else:
            raise NotImplementedError("Embedding_layer input should be typle type! Please checkout the input")

    def get_vector(self, seq_ids):
        if len(seq_ids) < 1:
            pass  # TODO 此处直接忽略可能不合理
        # print(seq_ids)
        return self.embedding[seq_ids]


if __name__ == '__main__':
    mus = kernel_mu(16)
    sigmas = kernel_sigma(16)
    print(f"mus:{mus}")
    print(f"f0:{mus[0]} | f1:{mus[1]} | f10:{mus[10]} | f2:{mus[2]}")
    print(f"sigmas:{sigmas}")
    embs = pooling_emb(np.ones((2, 10, 10)), n_gram=1)
    print(embs.shape)
    mask = remask(np.ones((2, 10, 10)), n_gram=1)
    print(mask.shape)
