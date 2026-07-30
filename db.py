import sqlite3

import pandas as pd


class Database:
    def __init__(self, db_filename):
        self.db_filename = db_filename

    def connect(self):
        return sqlite3.connect(self.db_filename)

    def read(self, sql, params=None):
        conn = self.connect()
        try:
            cur = conn.cursor()
            cur.execute(sql, params or [])
            return cur.fetchall()
        finally:
            conn.close()

    def read_df(self, sql, params=None):
        conn = self.connect()
        try:
            return pd.read_sql_query(sql, conn, params=params)
        finally:
            conn.close()

    def write(self, sql, params=None):
        conn = self.connect()
        try:
            cur = conn.cursor()
            cur.execute(sql, params or [])
            conn.commit()
        finally:
            conn.close()

    def write_many(self, sql, params_list):
        conn = self.connect()
        try:
            cur = conn.cursor()
            cur.executemany(sql, params_list)
            conn.commit()
        finally:
            conn.close()

    def write_df(self, df, table_name, if_exists="append"):
        conn = self.connect()
        try:
            df.to_sql(table_name, conn, if_exists=if_exists, index=False)
            conn.commit()
        finally:
            conn.close()
