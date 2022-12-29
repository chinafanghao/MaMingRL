import regex as re
from datetime import datetime
import subprocess
from importlib import reload
import pydash as ps
import pandas as pd
import torch
import pickle
import json
import numpy as np
import sys,os
import ujson
import yaml
from mamingRL import ROOT_DIR

#正则匹配形如'2018_12_02_082510‘的字符串，并返回
#Regular matching is like '2018_ 12_ 02_ 082510 and return
RE_FILE_TS = re.compile(r'(\d{4}_\d{2}_\d{2}_\d{6})')
#用于文件名的时间格式
FILE_TS_FORMAT = '%Y_%m_%d_%H%M%S'

class LabJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, (np.ndarray, pd.Series)):
            return obj.tolist()
        else:
            return str(obj)

def smart_path(path):
    '''
    :param path: the input path to be resolved,transform abspath combined with root dir
    :return: normal path after resolved which is combined with root dir 返回标准的绝对路径
    '''
    if os.path.isabs(path):
        path=os.path.join(ROOT_DIR,path)

    return os.path.normpath(path)

def get_file_ext(path):
    '''
    返回文件的后缀名
    :param path:
    :return:the extend form of file
    @example:
    'a.csv'===>'.csv'
    '''
    return os.path.splitext(path)[-1]
def read_as_df(path,**kwargs):
    return pd.read_csv(path,**kwargs)

def read_as_pickle(path,**kwargs):
    with open(path,'rb') as f:
        data=pickle.load(f,*kwargs)
    return data

def read_as_pth(path,**kwargs):
    return torch.load(path,**kwargs)

def read_as_plain(path,**kwargs):
    open_file=open(path,'r')
    ext=get_file_ext(path)
    if ext=='.json':
        data=ujson.load(open_file,**kwargs)
    elif ext=='.yml':
        data=yaml.load(open_file,**kwargs)
    else:
        data=open_file.read()
    open_file.close()
    return data

def read(path,**kwargs):
    '''
    :param path:input path of the file,and
        firstly transform it to a normal form,将输入路径转换为标准的路径形式
        secondly read context according to its file type,such as .csv .pkl .pth 根据文件类型读取文件内容
    :return: the context of the file
    '''

    path=smart_path(path)
    try:
        assert os.path.isfile(path),f'not a file,please input file path'
    except AssertionError:
        raise FileNotFoundError(path)
    file_ext=get_file_ext(path)
    if file_ext=='csv':
        data=read_as_df(path,**kwargs)
    elif file_ext=='.pkl':
        data=read_as_pickle(path,**kwargs)
    elif file_ext=='.pth':
        data=read_as_pth(path,**kwargs)
    else:
        data=read_as_plain(path,**kwargs)

    return data