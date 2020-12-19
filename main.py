#!/usr/bin/env python
# encoding=utf-8


from w2v import Embedding
from models.knrm import KNRM
from data_reader import loadyaml


# load config file
config = loadyaml('config/knrm.yaml')
# load embedding file
emb = Embedding(vec_path='data/embed.txt')
# build knrm model to gen features
model = KNRM(emb, config)
