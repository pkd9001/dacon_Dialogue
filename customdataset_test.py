# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 22:18:36 2022

@author: user
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer

class CustomDataset_test:
    def __init__(self, dataset=None):
        self.dataset = dataset
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
  
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
        token_type_ids = torch.tensor(datas['token_type_ids'])
        attention_mask = torch.tensor(datas['attention_mask'])
        
        return input_ids, token_type_ids, attention_mask