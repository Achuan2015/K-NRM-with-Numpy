# -*- coding:utf-8 -*-


import numpy as np
import os

from gensim.models import KeyedVectors


class Embedding:

    def __init__(self, vec_path='', cache_vec_path='', logger=None):
        self.vec_path = vec_path
        self.cache_vec_path = cache_vec_path
        self.logger = logger
        self._load_vec()
        self._load_cache_vec()

    def _load_vec(self):
        if os.path.exists(self.vec_path):
            if self.vec_path.endswith('model'):
                self.w2v = KeyedVectors.load(self.vec_path)
            elif self.vec_path.endswith('txt'):
                self.w2v = KeyedVectors.load_word2vec_format(self.vec_path, binary=False)
            else:
                raise FileNotFoundError(f"The {self.vec_path} is not exists ! ")    
        else:
            # self.logger.warning("Step X: the w2v file not exists ! ")
            raise FileNotFoundError("The w2v file is not exists ! ")

    def save_cache_model(self):
        self.w2v.save(self.cache_vec_path)

    def _load_cache_vec(self):
        if os.path.exists(self.cache_vec_path):
            self.cache_vec = {}
            # self.cache_vec = zload(self.cache_vec_path)
            pass
        else:
            self.cache_vec = {}

    def new_vector(self, v):
        return v if v is not None else np.random.randn(self.w2v.vector_size).astype("float32")

    def max_match_segment(self, text, window_max=5):
        match = False
        idx = 0
        word = []
        while idx < len(text):
            for i in range(window_max, 0, -1):
                cand = text[idx:idx + i]
                if cand in self.w2v:
                    word.append(cand)
                    match = True
                    break
            if not match:
                i = 1
                word.append(text[idx])
            idx += 1
        return word

    def __getitem__(self, word):
        v = None
        if word is None:
            return None
        if word in self.cache_vec:
            return self.cache_vec[word]
        if word in self.w2v:
            v = self.w2v[word]
        else:
            words = self.max_match_segment(word)
            vector = np.zeros(self.vector_size, dtype="float32")
            for token in words:
                if token in self.w2v:
                    vector += self.w2v[token]
                else:
                    # ignore word not in vocab after the max_match_segment function
                    vector += np.zeros(self.vector_size).astype("float32")
            v = vector / len(words)
        v = self.new_vector(v)
        self.cache_vec[word] = v
        return v

    def __contains__(self, word):
        return word in self.w2v

    @property
    def vector_size(self):
        return self.w2v.vector_size


if __name__ == '__main__':
    # emb = Embedding('../data/w2v.model')
    emb = Embedding('../data/embedding/embed.txt')
