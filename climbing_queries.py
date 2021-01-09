import sqlite3 as sql 
import numpy as np
import pandas as pd 
import requests
from os import getcwd

MP_API_KEY = 'SECRET_KEY'
GET_USER_INFO = 'https://www.mountainproject.com/data/get-user'

DB_PATH = getcwd() + '/datasets/climbing_full.sqlite'

def get_uid(email):
    query = {
        'key':MP_API_KEY,
        'email':email
    }
    rs = requests.get(GET_USER_INFO,params=query)
    js = rs.json()
    return js['id']


def get_most_recent(uid,k=10):
    with sql.connect(DB_PATH) as conn:
        most_recent = pd.read_sql_query('''
        SELECT user_id, route_id, date
        FROM ticks 
        WHERE user_id = ? 
        ORDER BY date DESC
        LIMIT ?
        ''',conn,params=[uid,k])
    return most_recent 

def get_most_recent_route_info(uid,k=10):
    with sql.connect(DB_PATH) as conn:
        most_recent = pd.read_sql_query(
            '''
            SELECT * FROM routes 
            INNER JOIN (
                SELECT route_id 
                FROM ticks 
                WHERE user_id = ? 
                ORDER BY DATE DESC 
                LIMIT ?
            ) ON id = route_id; 
            '''
            , conn,params=[uid,k]
        )
    return most_recent

def get_route_info(iids):
    '''
    Returns Full Route Information for
    list of route ids

    Param
    ------
    iids: list
        - list of item ids
    
    Returns 
    --------
    pd.Dataframe
        - Dataframe of route info
    '''
    query = f'''
    SELECT * 
    FROM routes
    WHERE id IN ({','.join(['?' for _ in iids])})
    '''

    with sql.connect(DB_PATH) as conn:
        route_info = pd.read_sql_query(query,conn,params=iids)
    route_info.set_index('id',inplace=True)
    route_info = route_info.reindex(np.array(list(map(int,iids)))).dropna(how='all')

    return route_info