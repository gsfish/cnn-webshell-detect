#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import time
from configparser import ConfigParser

import pymysql
from flask import abort


config = ConfigParser()
config.read('config.ini')
db_config = {
    'host': config['database']['host'],
    'user': config['database']['username'],
    'password': config['database']['password'],
    'database': config['database']['database']
}


class Database():

    def __init__(self):
        global db_config

        self.conn = pymysql.connect(**db_config)
        self.curs = self.conn.cursor()


    def __del__(self):
        self.curs.close()
        self.conn.close()


    def create_result(self, file_id, malicious_judge, malicious_chance):
        sql = 'INSERT INTO result(file_id, malicious_judge, malicious_chance, created_at) VALUES (%s, %s, %s, %s);'
        parm = (file_id, malicious_judge, malicious_chance, time.strftime("%Y-%m-%d %H:%M:%S"))
        try:
            self.curs.execute(sql, parm)
        except Exception:
            self.conn.rollback()
            logging.exception('SQL: {0}'.format(sql))
            abort(500)
        else:
            self.conn.commit()


    def check_result(self, file_id):
        sql = 'SELECT malicious_judge, malicious_chance, created_at FROM result WHERE file_id = %s;'
        param = (file_id,)
        try:
            self.curs.execute(sql, param)
        except Exception:
            logging.exception('SQL: {0}'.format(sql))
            abort(500)
        else:
            if self.curs.rowcount == 0:
                return None
            result = self.curs.fetchone()
            return result
