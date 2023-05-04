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
from MaMingRL import ROOT_DIR
import collections
import random
from tqdm import tqdm

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

def moving_average(data,wind_size):
    data=np.cumsum(np.insert(data,0,0))
    middle=(data[wind_size:]-data[:-wind_size])/wind_size
    r=np.arange(1,wind_size)
    begin=data[:wind_size-1]/r
    ends=(data[-1]-data[-wind_size:-1])/(r[::-1])
    return np.concatenate((begin,middle,ends))


class BufferReplay:
    def __init__(self,BufferSize):
        self.buffer=collections.deque(maxlen=BufferSize)
        self.buffersize=BufferSize

    def __len__(self):
        return len(self.buffer)

    def add(self,data):
        if len(self.buffer)<self.buffersize:
            self.buffer.append(data)
        else:
            self.buffer.popleft()
            self.buffer.append(data)

    def sample(self,sample_size):
        transition=random.sample(self.buffer,sample_size)
        state,action,reward,next_state,done,truncted=zip(*transition)
        return np.array(state),action,reward,np.array(next_state),done,truncted

def compuate_GAE(lmbda,gamma,td_delta):
    td_delta=td_delta.detach().numpy()
    advantage=0
    all_advantage=[]
    for delta in td_delta[::-1]:
        advantage=lmbda*gamma*advantage+delta
        all_advantage.append(advantage)
    all_advantage.reverse()
    return torch.tensor(all_advantage,dtype=torch.float32)

def train_on_policy_agent(env,agent,num_episodes):
    return_list=[]
    for i in range(10):
        with tqdm(total=int(num_episodes/10),desc='Iteration %d' % i) as pbar:
            for i_iteration in range(int(num_episodes/10)):
                episode_return=0
                done=False
                state,_=env.reset()
                transition_dict={'states':[],'actions':[],'rewards':[],'next_states':[],'dones':[],'truncateds':[]}
                while not done:
                    action=agent.take_action(state)
                    next_state,reward,done,truncated,_=env.step(action)
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['rewards'].append(reward)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['dones'].append(done)
                    transition_dict['truncateds'].append(truncated)
                    episode_return+=reward
                    if truncated:
                        break
                    state=next_state
                agent.update(transition_dict)
                return_list.append(episode_return)
                if (i_iteration+1)%10==0:
                    pbar.set_postfix({'episode':num_episodes/10*i+i_iteration+1,'return':'%.3f'% np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list

def train_off_policy_agent(env,agent,num_episodes,buffer,minimal_size,batch_size):
    return_list=[]
    for i in range(10):
        with tqdm(total=int(num_episodes/10),desc='Iteration %d'%(i+1)) as pbar:
            for i_iteration in range(int(num_episodes/10)):
                state,_=env.reset()
                done=False
                episode_return=0
                while not done:
                    action=agent.take_action(state)
                    next_state,reward,done,truncated,_=env.step(action)
                    transition_dict={'state':state,'action':action,'reward':reward,'next_state':next_state,'done':done,'truncated':truncated}
                    buffer.add(transition_dict)
                    episode_return+=reward
                    if truncated:
                        break
                return_list.append(episode_return)
                if len(buffer)>=minimal_size:
                    transition=buffer.sample(batch_size)
                    agent.update(transition)

                if (i_iteration+1)%10==0:
                    pbar.set_postfix({'episodes:':num_episodes/10*i+i_iteration+1,'return':'%.3f'%np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list