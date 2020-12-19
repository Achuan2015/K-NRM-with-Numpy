# coding=utf-8


import os

import numpy as np
from tqdm import tqdm

from utils.utils_common import request


def gen_weights(dataloader, model, config, logger, path=''):
    # config
    if not os.path.exists(f'{path}'):
        with open(f'{path}', 'wb') as tr:
            for batch in tqdm(dataloader):
                idx, q_ids, t_ids, q_len, t_len, labels, sims, bm25 = batch
                q_ids, t_ids = np.array(q_ids), np.array(t_ids)
                features = model(q_ids, t_ids)
                assert len(labels) == features.shape[0]
                #import ipdb; ipdb.set_trace()
                qt_len = np.append(np.array(list(zip(q_len))), np.array(list(zip(t_len))), axis=1)
                features = np.append(features, qt_len, axis=1)
                features = np.append(features, np.array(sims)[:, np.newaxis], axis=1)
                features = np.append(features, np.array(bm25)[:, np.newaxis], axis=1)
                np.savetxt(tr, np.append(features, np.array(labels)[:, np.newaxis], axis=1), delimiter=',')


def gen_weights_online(dataloader, model):
    for batch in dataloader:
        idx, q_ids, t_ids, q_len, t_len= batch
        q_ids, t_ids = np.array(q_ids), np.array(t_ids)
        features = model(q_ids, t_ids)
        assert len(idx) == features.shape[0]
        qt_len = np.append(np.array(list(zip(q_len))), np.array(list(zip(t_len))), axis=1)
        features = np.append(features, qt_len, axis=1)
        #eilmdata = {'query': list(query), 'target': target}
        #eilm = request(url='http://192.168.1.36:6000/api/esim', data=eilmdata)
        #features = np.append(features, np.array(eilm)[:, np.newaxis], axis=1)
        return (idx, features)


def gen_weights_local(dataloader, model, config, logger, path=''):
    # config
    if not os.path.exists(f'{path}'):
        with open(f'{path}', 'wb') as tr:
            for batch in tqdm(dataloader):
                idx, q_ids, t_ids, q_len, t_len, labels = batch
                q_ids, t_ids = np.array(q_ids), np.array(t_ids)
                features = model(q_ids, t_ids)
                assert len(labels) == features.shape[0]
                #import ipdb; ipdb.set_trace()
                qt_len = np.append(np.array(list(zip(q_len))), np.array(list(zip(t_len))), axis=1)
                features = np.append(features, qt_len, axis=1)
                np.savetxt(tr, np.append(features, np.array(labels)[:, np.newaxis], axis=1), delimiter=',')


def accuracy_score(predict, label):
    if isinstance(predict, np.ndarray):
        correct = sum(predict == label)
        return round(correct / len(predict), 5)
    else:
        raise NotImplementedError("The input data type only support np.ndarray")
