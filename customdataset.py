# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 22:18:36 2022

@author: user
"""

import torch
import torch.nn as nn
from transformers import (AutoTokenizer,
                          RobertaTokenizer,
                          XLMRobertaTokenizer
                          )

class CustomDataset: # bert
    def __init__(self, dataset=None, mode = "train"):
        self.dataset = dataset
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        self.mode = mode
  
    def __len__(self):
        return len(self.dataset)
  
    def __getitem__(self, idx):
        dataset = self.dataset.iloc[idx]
        try:
            datas = self.tokenizer(dataset['Utterance_add'], dataset['Utterance'],
                                   #padding='max_length', truncation=True)
                                   padding='max_length',
                                   truncation=True#,
                                   # max_length=max_length,
                                   #return_special_tokens_mask=True
                                   )
        except:
            datas = self.tokenizer(dataset['Utterance'],
                                   #padding='max_length', truncation=True)
                                   padding='max_length',
                                   truncation=True#,
                                   # max_length=max_length,
                                   #return_special_tokens_mask=True
                                   )
        input_ids = torch.tensor(datas['input_ids'])
        token_type_ids = torch.tensor(datas['token_type_ids'])
        attention_mask = torch.tensor(datas['attention_mask'])
        if self.mode == "train":
            labels = torch.tensor(self.dataset.iloc[idx]['Target'])
            return input_ids, token_type_ids, attention_mask, labels
        else:
            return input_ids, token_type_ids, attention_mask

class CustomDataset_stack2: # bert 1-sentence
    def __init__(self, dataset=None, mode = "train"):
        self.dataset = dataset
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        self.mode = mode
  
    def __len__(self):
        return len(self.dataset)
  
    def __getitem__(self, idx):
        dataset = self.dataset.iloc[idx]
        datas = self.tokenizer(dataset['Utterance_add'],
                               #padding='max_length', truncation=True)
                               padding='max_length',
                               truncation=True#,
                               # max_length=max_length,
                               #return_special_tokens_mask=True
                               )
        input_ids = torch.tensor(datas['input_ids'])
        token_type_ids = torch.tensor(datas['token_type_ids'])
        attention_mask = torch.tensor(datas['attention_mask'])
        if self.mode == "train":
            labels = torch.tensor(self.dataset.iloc[idx]['Target'])
            return input_ids, token_type_ids, attention_mask, labels
        else:
            return input_ids, token_type_ids, attention_mask

class CustomDataset_ro: # roberta
    def __init__(self, dataset=None, mode = "train"):
        self.dataset = dataset
        self.tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
        self.mode = mode
  
    def __len__(self):
        return len(self.dataset)
  
    def __getitem__(self, idx):
        dataset = self.dataset.iloc[idx]
        datas = self.tokenizer(dataset['Utterance_add'], dataset['Utterance'],
                               #padding='max_length', truncation=True)
                               padding='max_length',
                               truncation=True#,
                               # max_length=max_length,
                               #return_special_tokens_mask=True
                               )
        input_ids = torch.tensor(datas['input_ids'])
        attention_mask = torch.tensor(datas['attention_mask'])
        if self.mode == "train":
            labels = torch.tensor(self.dataset.iloc[idx]['Target'])
            return input_ids, attention_mask, labels
        else:
            return input_ids, attention_mask

class CustomDataset_em: # bert
    def __init__(self, dataset=None, mode = "train"):
        self.dataset = dataset
        self.tokenizer = AutoTokenizer.from_pretrained("tae898/emoberta-base")
        self.mode = mode
  
    def __len__(self):
        return len(self.dataset)
  
    def __getitem__(self, idx):
        dataset = self.dataset.iloc[idx]
        datas = self.tokenizer(dataset['Utterance_add'], dataset['Utterance'],
                               #padding='max_length', truncation=True)
                               padding='max_length',
                               truncation=True#,
                               # max_length=max_length,
                               #return_special_tokens_mask=True
                               )
        input_ids = torch.tensor(datas['input_ids'])
        # token_type_ids = torch.tensor(datas['token_type_ids'])
        attention_mask = torch.tensor(datas['attention_mask'])
        if self.mode == "train":
            labels = torch.tensor(self.dataset.iloc[idx]['Target'])
            return input_ids, attention_mask, labels
        else:
            return input_ids, attention_mask

class CustomDataset_em_1_sentense: # bert
    def __init__(self, dataset=None, mode = "train"):
        self.dataset = dataset
        self.tokenizer = AutoTokenizer.from_pretrained("tae898/emoberta-base")
        self.mode = mode
  
    def __len__(self):
        return len(self.dataset)
  
    def __getitem__(self, idx):
        dataset = self.dataset.iloc[idx]
        datas = self.tokenizer(dataset['Utterance'],
                               #padding='max_length', truncation=True)
                               padding='max_length',
                               truncation=True#,
                               # max_length=max_length,
                               #return_special_tokens_mask=True
                               )
        input_ids = torch.tensor(datas['input_ids'])
        # token_type_ids = torch.tensor(datas['token_type_ids'])
        attention_mask = torch.tensor(datas['attention_mask'])
        if self.mode == "train":
            labels = torch.tensor(self.dataset.iloc[idx]['Target'])
            return input_ids, attention_mask, labels
        else:
            return input_ids, attention_mask








