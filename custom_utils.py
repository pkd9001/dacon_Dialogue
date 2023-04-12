# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 14:35:53 2022

@author: user
"""

import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from customdataset import *
import numpy as np
import random
import os
from datasets import load_dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def normal_data(data):
    return data

def make_new_data(data):
    empty = [""]
    empty = pd.DataFrame(empty)
    add_new_data = []
    null_num = []
    for i in range(len(data['Dialogue_ID'].value_counts())+1): # 0~1038 
        if i not in data['Dialogue_ID'].values:
            n = i
            null_num.append(n)
            print(null_num)
    for i in range(len(data['Dialogue_ID'].value_counts())+1):# 0~1038
        for t in null_num:
            if i == t:
                continue
            elif len(data[data['Dialogue_ID'] == i]['Utterance'][:-1]) == 0:
                add_data = empty    
            else: 
                add_data = pd.concat([empty, data[data['Dialogue_ID']==i]['Utterance'][:-1]])# 다이올로그 뽑기(60번 없음 처리 필요)
            add_data_ = add_data
            add_new_data.append(add_data_)
        
    add_new_data = np.concatenate(add_new_data).tolist()
    add_new_data = np.concatenate(add_new_data).tolist()
    add_new_data = pd.DataFrame(add_new_data)
    add_new_data.columns = ['Utterance_add']
    add_new_data = add_new_data.reset_index(drop=True)
    new_train_data = pd.concat([add_new_data, data], axis=1)
    
    return new_train_data

def make_new_data_stack(data): # data 2 sentense stack
    empty = [""]
    empty = pd.DataFrame(empty)
    add_new_data = []
    null_num = []
    for i in range(len(data['Dialogue_ID'].value_counts())+1): # 빈 수 검사
        if i not in data['Dialogue_ID'].values:
            n = i
            null_num.append(n)
            print(null_num)
    for i in range(len(data['Dialogue_ID'].value_counts())+1):# 0~1038, 다이올로그 뽑기(60번 없음 처리 필요)
        for t in null_num:
            if i == t:
                continue
            elif len(data[data['Dialogue_ID'] == i]['Utterance'][:-1]) == 0:
                add_data = empty
            else: 
                add_text = data[data['Dialogue_ID'] == i]
                add_text = add_text['Utterance'][:-1]
                add_text = pd.DataFrame(add_text)
                add_text = add_text.reset_index(drop=True)

                add_value = ''
                add_values = []
                add_values_ = []
                for r in add_text['Utterance']:
                    add_value = str(add_value) + " " + r
                    add_values.append(add_value)

                add_values = pd.DataFrame(add_values)
                add_values = pd.concat([empty, add_values])
                add_values = add_values.reset_index(drop=True)
                add_values.columns = ["Utterance_add"]
                add_data = add_values
            add_data_ = add_data
            add_new_data.append(add_data_)
        
    add_new_data = np.concatenate(add_new_data).tolist()
    add_new_data = np.concatenate(add_new_data).tolist()
    add_new_data = pd.DataFrame(add_new_data)
    add_new_data.columns = ['Utterance_add']
    add_new_data = add_new_data.reset_index(drop=True)
    new_train_data = pd.concat([add_new_data, data], axis=1)
    
    return new_train_data


def make_new_data_stack2(data): # 단일 data stack

    empty = [""]
    empty = pd.DataFrame(empty)
    add_new_data = []
    null_num = []
    for i in range(len(data['Dialogue_ID'].value_counts())+1): # 빈 수 검사
        if i not in data['Dialogue_ID'].values:
            n = i
            null_num.append(n)
            print(null_num)
    for i in range(len(data['Dialogue_ID'].value_counts())+1):# 0~1038, 다이올로그 뽑기(60번 없음 처리 필요)
        for t in null_num:
            if i == t:
                continue
            elif len(data[data['Dialogue_ID'] == i]['Utterance']) == 0:
                add_data = empty
            else: 
                add_text = data[data['Dialogue_ID'] == i]
                add_text = add_text['Utterance']
                add_text = pd.DataFrame(add_text)
                add_text = add_text.reset_index(drop=True)
    
                add_value = ''
                add_values = []
                add_values_ = []
                for r in add_text['Utterance']:
                    add_value = str(add_value) + " " + r
                    add_values.append(add_value)
    
                add_values = pd.DataFrame(add_values)
                add_values = pd.concat([add_values])
                add_values = add_values.reset_index(drop=True)
                add_values.columns = ["Utterance_add"]
                add_data = add_values
            add_data_ = add_data
            add_new_data.append(add_data_)
        
    add_new_data = np.concatenate(add_new_data).tolist()
    add_new_data = np.concatenate(add_new_data).tolist()
    add_new_data = pd.DataFrame(add_new_data)
    add_new_data.columns = ['Utterance_add']
    add_new_data = add_new_data.reset_index(drop=True)
    new_train_data = pd.concat([add_new_data, data], axis=1)

    return new_train_data

def dataloader(batch_size, num_workers, make_dataset):
    train = pd.read_csv("./data/train.csv")
    train = pd.DataFrame(train)
    
    test = pd.read_csv("./data/test.csv")
    test = pd.DataFrame(test)
    
    train_data = make_dataset(train)
    test_data = make_dataset(test)

    le = LabelEncoder()
    le=le.fit(train_data['Target'])
    train_data['Target']=le.transform(train_data['Target'])
    
    try:
        X_train, X_test, y_train, y_test = train_test_split(train_data[['Utterance_add','Utterance']],
                                                            train_data['Target'],
                                                            test_size=0.1)
    except:
        try:
            X_train, X_test, y_train, y_test = train_test_split(train_data['Utterance_add'],
                                                                train_data['Target'],
                                                                test_size=0.1)
        except:
            X_train, X_test, y_train, y_test = train_test_split(train_data['Utterance'],
                                                                train_data['Target'],
                                                                test_size=0.1)
    
    X = pd.concat([X_train, y_train], axis=1)
    Y = pd.concat([X_test,y_test], axis=1)
    
    train_dataset = CustomDataset(X, mode="train") # bert
    val_dataset = CustomDataset(Y, mode="train")
    test_dataset = CustomDataset(test_data, mode="test")
    
    # train_dataset = CustomDataset_stack2(X, mode="train") # bert_stack
    # val_dataset = CustomDataset_stack2(Y, mode="train")
    # test_dataset = CustomDataset_stack2(test_data, mode="test")    
    
    # train_dataset = CustomDataset_ro(X, mode="train") # roberta
    # val_dataset = CustomDataset_ro(Y, mode="train")
    # test_dataset = CustomDataset_ro(test_data, mode="test")
    
    # train_dataset = CustomDataset_em(X, mode="train") # bert_em
    # val_dataset = CustomDataset_em(Y, mode="train")
    # test_dataset = CustomDataset_em(test_data, mode="test")
    
    # train_dataset = CustomDataset_em_1_sentense(X, mode="train") # bert_em
    # val_dataset = CustomDataset_em_1_sentense(Y, mode="train")
    # test_dataset = CustomDataset_em_1_sentense(test_data, mode="test")
    
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=num_workers)#, collate_fn=None)
    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=False, pin_memory=True,
                            num_workers=num_workers)#, collate_fn=None)
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             pin_memory=True,
                             num_workers=num_workers)#, collate_fn=None)
    
    return train_loader, val_loader, test_loader

def seed_everything(seed:int = 1004):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = False  # type: ignore

class LabelSmoothingLoss(nn.Module):

    def __init__(self, smoothing=0.0):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

def competition_metric(true, pred):
    return f1_score(true, pred, average="macro")




















