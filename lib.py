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


    def create_result(self, fid, judge, chance):
        sql = 'INSERT INTO result(fid, judge, chance, ctime) VALUES (%s, %s, %s, %s);'
        parm = (fid, judge, chance, time.strftime("%Y-%m-%d %H:%M:%S"))
        try:
            self.curs.execute(sql, parm)
        except Exception:
            self.conn.rollback()
            logging.exception('SQL: {0}'.format(sql))
            abort(500)
        else:
            self.conn.commit()


    def check_result(self, fid):
        sql = 'SELECT judge, chance, ctime FROM result WHERE fid = %s;'
        parm = (fid,)
        try:
            self.curs.execute(sql, parm)
        except Exception:
            logging.exception('SQL: {0}'.format(sql))
            abort(500)
        else:
            if self.curs.rowcount == 0:
                return None
            result = self.curs.fetchone()
            return result
