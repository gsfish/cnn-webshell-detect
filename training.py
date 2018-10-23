#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
import re
import subprocess
from configparser import ConfigParser
from functools import reduce

import tensorflow as tf
import tflearn
from sklearn.utils import shuffle


config = ConfigParser()
config.read('config.ini')
black_files = config['training']['black_files']
white_files = config['training']['white_files']
opcode_file = config['training']['opcode_file']
model_path = config['training']['model_path']
model_record = config['training']['model_record']
train_epoch = int(config['training']['epoch'])
train_batch = int(config['training']['batch'])


def get_php_file(base_dir):
    file_list = []
    for path, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.php'):
                filename = os.path.realpath(os.path.join(path, file))
                file_list.append(filename)
    return file_list


def get_all_opcode(base_dir):
    file_list = get_php_file(base_dir)
    opcode_list = []
    for file in file_list:
        opcode = get_file_opcode(file)
        if opcode:
            opcode_list.append(opcode)
    return opcode_list


def get_file_opcode(filename):
    php_path = '/usr/bin/php'
    cmd = ('{0} -dvld.active=1 -dvld.execute=0'.format(php_path)).split()
    cmd.append(filename)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    opcodes = []
    pattern = r'([A-Z_]{2,})\s+'
    for line in p.stdout.readlines()[8:-3]:
        try:
            match = re.search(pattern, str(line, encoding='utf-8'))
        except UnicodeDecodeError:
            match = re.search(pattern, str(line))
        if match:
            opcodes.append(match.group(1))
    p.terminate()
    return opcodes


def serialize_decorator(func):
    global opcode_file

    with open(opcode_file, 'r') as f:
        code_record = list(map(lambda x: x.strip(), f.readlines()))

    def wrapped(*args, **kwargs):
        return func(*args, _code_record=code_record, **kwargs)
    return wrapped


@serialize_decorator
def serialize_codes(code_list, _code_record):
    for file_code in code_list:
        for index, code in enumerate(file_code):
            if _code_record.count(code):
                file_code[index] = _code_record.index(code) + 1
            else:
                file_code[index] = 0


def create_network(seq_length, optimizer, learning_rate, loss):
    # 输入参数的最大长度为序列的最大长度
    network = tflearn.input_data(shape=[None, seq_length], name='input')

    # CNN 模型, 使用 3 个数量为 128, 长度分别为 3, 4, 5 的一维卷积函数处理数据
    network = tflearn.embedding(network, input_dim=100000, output_dim=128)
    branch1 = tflearn.layers.conv.conv_1d(network, 128, 3, padding='valid', activation='relu', regularizer='L2')
    branch2 = tflearn.layers.conv.conv_1d(network, 128, 4, padding='valid', activation='relu', regularizer="L2")
    branch3 = tflearn.layers.conv.conv_1d(network, 128, 5, padding='valid', activation='relu', regularizer="L2")
    tflearn.layers.merge_ops.merge([branch1, branch2, branch3], mode='concat', axis=1)

    network = tf.expand_dims(network, 2)
    network = tflearn.layers.conv.global_max_pool(network)
    network = tflearn.layers.core.dropout(network, 0.8)
    network = tflearn.fully_connected(network, 2, activation='softmax')
    network = tflearn.layers.estimator.regression(
        network,
        optimizer=optimizer,
        learning_rate=learning_rate,
        loss=loss,
        name='target')
    return network


def train_model(x1_code, y1_label, x2_code, y2_label):
    global model_path, model_record

    train_optimizer = 'adam'
    train_learning_rate = 0.001
    train_loss = 'categorical_crossentropy'

    x1_code.extend(x2_code)
    y1_label.extend(y2_label)

    # Shuffle the ordinary of dataset
    x_train, y_train = shuffle(x1_code, y1_label)

    # Serialize the opcodes into numbers
    print('[*] serializing opcode from persistence set')
    serialize_codes(x_train)

    # Find the max length of sequence
    train_seq_length = len(reduce(lambda x, y: x if len(x) > len(y) else y, x1_code))
    print('[+] max length of persistence set: {0}'.format(train_seq_length))

    # Padding all sequence to max length
    x_train = tflearn.data_utils.pad_sequences(x_train,
                                               maxlen=train_seq_length,
                                               value=0.)

    # Categorical label
    y_train = tflearn.data_utils.to_categorical(y_train, nb_classes=2)

    # tflearn.config.init_graph(gpu_memory_fraction=0.9, soft_placement=True)

    network = create_network(
        train_seq_length,
        optimizer=train_optimizer,
        learning_rate=train_learning_rate,
        loss=train_loss)

    # 实例化 CNN 对象并训练数据
    print('[*] traning started')
    print('[+] epoch: {0}, batch_size: {1}'.format(train_epoch, train_batch))
    model = tflearn.DNN(network, tensorboard_verbose=0)
    model.fit(
        x_train, y_train,
        n_epoch=train_epoch,
        shuffle=True,
        validation_set=0.1,
        show_metric=True,
        batch_size=train_batch,
        run_id='webshell')

    record = {
        'seq_length': train_seq_length,
        'optimizer': train_optimizer,
        'learning_rate': train_learning_rate,
        'loss': train_loss,
        'n_epoch': train_epoch,
        'batch_size': train_batch
    }
    json.dump(record, open(model_record, 'w'))
    model.save(model_path)
    print('[+] model saved in {0}'.format(model_path))

    return model


def get_model():
    global model_path, model_record, black_files, white_files

    if os.path.isfile(os.path.join(os.path.dirname(model_path), 'checkpoint')):
        print('[*] loading model...')

        record = json.load(open(model_record, 'r'))
        seq_length = record['seq_length']
        optimizer = record['optimizer']
        learning_rate = record['learning_rate']
        loss = record['loss']

        network = create_network(
            seq_length,
            optimizer=optimizer,
            learning_rate=learning_rate,
            loss=loss)
        model = tflearn.DNN(network, tensorboard_verbose=1)
        model.load(model_path)
        print('[+] {0} loaded'.format(model_path))
        return model

    print('[*] loading black files...')
    black_code_list = get_all_opcode(black_files)
    black_label = [1] * len(black_code_list)
    print('[+] {0} black files loaded'.format(len(black_code_list)))

    print('[*] loading white files...')
    white_code_list = get_all_opcode(white_files)
    white_label = [0] * len(white_code_list)
    print('[+] {0} white files loaded'.format(len(white_code_list)))

    print('[*] training model...')
    model = train_model(black_code_list, black_label,
                        white_code_list, white_label)
    print('[+] training complete')
    return model


if __name__ == '__main__':
    get_model()
