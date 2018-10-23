#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
from configparser import ConfigParser

import tflearn
from numpy import argmax
from sklearn import metrics
from sklearn.utils import shuffle

import training


config = ConfigParser()
config.read('config.ini')
black_files = config['training']['black_files']
white_files = config['training']['white_files']
model_path = config['training']['model_path']
model_record = config['training']['model_record']


def test_model(x1_code, y1_label, x2_code, y2_label):
    global model_path, model_record

    x1_code.extend(x2_code)
    y1_label.extend(y2_label)

    x_test, y_test = shuffle(x1_code, y1_label)
    print('size of testing set: {0}'.format(len(x_test)))

    print('serializing opcode from testing set')
    training.serialize_codes(x_test)

    record = json.load(open(model_record, 'r'))
    seq_length = record['seq_length']
    optimizer = record['optimizer']
    learning_rate = record['learning_rate']
    loss = record['loss']

    x_test = tflearn.data_utils.pad_sequences(x_test, maxlen=seq_length, value=0.)
    y_test = tflearn.data_utils.to_categorical(y_test, nb_classes=2)

    network = training.create_network(seq_length, optimizer=optimizer, learning_rate=learning_rate, loss=loss)
    model = tflearn.DNN(network, tensorboard_verbose=0)
    model.load(model_path)

    y_pred = model.predict(x_test)
    y_pred = argmax(y_pred, axis=1)

    print('metrics.accuracy_score:')
    print(metrics.accuracy_score(y_test, y_pred))
    print('metrics.confusion_matrix:')
    print(metrics.confusion_matrix(y_test, y_pred))
    print('metrics.precision_score:')
    print(metrics.precision_score(y_test, y_pred))
    print('metrics.recall_score:')
    print(metrics.recall_score(y_test, y_pred))
    print('metrics.f1_score:')
    print(metrics.f1_score(y_test, y_pred))


if __name__ == '__main__':
    print('loading black files...')
    black_code_list = training.get_all_opcode(black_files)
    black_label = [1] * len(black_code_list)
    print('{0} black files loaded'.format(len(black_code_list)))

    print('loading white files...')
    white_code_list = training.get_all_opcode(white_files)
    white_label = [0] * len(white_code_list)
    print('{0} white files loaded'.format(len(white_code_list)))

    test_model(black_code_list, black_label, white_code_list, white_label)
