#!/usr/bin/env python
# -*- coding: utf-8 -*-

import atexit
import hashlib
import logging
import os
import time
from configparser import ConfigParser

import tflearn
from flask import *
from numpy import argmax

import training
from lib import Database


config = ConfigParser()
config.read('config.ini')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = config['api']['upload_path']
app.config['MAX_CONTENT_LENGTH'] = int(config['api']['upload_max_length'])

logging.basicConfig(level=logging.DEBUG, filename='server.log', filemode='w',
                    format='%(asctime)s %(filename)s [line:%(lineno)d] %(levelname)s %(message)s')


class TempFile():

    def __init__(self, path, name):
        self.filepath = os.path.abspath(path)
        self.filename = name

    def get_filename(self):
        return self.filename

    def get_filepath(self):
        return os.path.realpath(os.path.join(self.filepath, self.filename))

    def __del__(self):
        # file = os.join(self.filepath, self.filename)
        # if os.path.isfile(file):
        #     os.remove(self.file)
        pass


def check_with_model(file_id):
    global model

    file = TempFile(os.path.join(app.config['UPLOAD_FOLDER']), file_id)
    file_opcodes = [training.get_file_opcode(file.get_filepath())]
    training.serialize_codes(file_opcodes)
    file_opcodes = tflearn.data_utils.pad_sequences(file_opcodes, maxlen=seq_length, value=0.)

    res_raw = model.predict(file_opcodes)
    res = {
        # revert from categorical
        'judge': True if argmax(res_raw, axis=1)[0] else False,
        'chance': float(res_raw[0][argmax(res_raw, axis=1)[0]])
    }
    return res


def vaild_file(filename):
    ALLOWED_EXTENSIONS = ['php']
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/check/result/<file_id>')
def check_webshell(file_id):
    if os.path.isfile(os.path.join(app.config['UPLOAD_FOLDER'], file_id)):
        db = Database()
        fetch = db.check_result(file_id)
        if fetch:
            logging.info('got previous record: {0}'.format(file_id))
            malicious_judge = (fetch[0] == 1)
            malicious_chance = fetch[1]
            res = {
                'file_id': file_id,
                'malicious_judge': malicious_judge,
                'malicious_chance': malicious_chance
            }
        else:
            logging.info('checking file: {0}'.format(file_id))
            res_check = check_with_model(file_id)
            res = {
                'file_id': file_id,
                'malicious_judge': res_check['judge'],
                'malicious_chance': res_check['chance']
            }
            db.create_result(file_id, res_check['judge'], res_check['chance'])
            logging.info('record created: {0}'.format(file_id))
    else:
        res = {
            'file_id': file_id,
            'malicious_judge': None,
            'malicious_chance': None
        }
    return jsonify(res)


@app.route('/check/upload', methods=['GET', 'POST'])
def receive_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and vaild_file(file.filename):
            file_id = hashlib.md5((file.filename+str(time.time())).encode('utf-8')).hexdigest()
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], file_id))
            return redirect(url_for('check_webshell', file_id=file_id))
        else:
            return abort(400)

    elif request.method == 'GET':
        return render_template('upload.html')


@app.route('/')
def index():
    return redirect(url_for('receive_file'))


@atexit.register
def atexit():
    logging.info('server stop')


if __name__ == '__main__':
    global model, seq_length

    host = config['server']['host']
    port = int(config['server']['port'])

    model_record = config['training']['model_record']
    seq_length = json.load(open(model_record, 'r'))['seq_length']

    model = training.get_model()
    logging.info('model loaded')

    logging.info('server started')
    app.run(host='0.0.0.0', debug=True)
