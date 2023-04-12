# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 21:17:45 2022

@author: user
"""

import torch
import torch.nn as nn

from transformers import (AutoTokenizer,
                          AdamW,
                          BertForSequenceClassification,
                          RobertaForSequenceClassification,
                          RobertaTokenizer,
                          XLMRobertaTokenizer,
                          XLMRobertaForSequenceClassification,
                          AutoModelForMaskedLM
                          )
from transformers.optimization import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

import numpy as np
import pandas as pd
from gluonnlp.data import SentencepieceTokenizer
from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split

from tqdm import tqdm
import re

import os
import random
import math
from custom_utils import (LabelSmoothingLoss,
                          seed_everything,
                          dataloader,
                          make_new_data,
                          make_new_data_stack,
                          competition_metric)

from sklearn.preprocessing import LabelEncoder

pretrain = "bert-base-cased"

num_labels = 7
batch_size = 1
num_workers = 0

le = LabelEncoder()

train = pd.read_csv("train.csv")
train = pd.DataFrame(train)

le = LabelEncoder()
le=le.fit(train['Target'])

_, _, test_loader = dataloader(batch_size,
                               num_workers,
                               # make_new_data
                               make_new_data_stack
                               )

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = BertForSequenceClassification.from_pretrained(pretrain, num_labels=num_labels,
                                                      # output_attentions=True,
                                                      # output_hidden_states = True,
                                                      # attention_probs_dropout_prob=0.5,
                                                      # hidden_dropout_prob=0.5,
                                                      ).to(device)

# model = XLMRobertaForSequenceClassification.from_pretrained("xlm-roberta-base",
#                                              num_labels=num_labels
#                                              ).to(device)

model_state_dict = torch.load("model/Epoch_5_loss_0.0095.pt") # roberta
model.load_state_dict(model_state_dict)

predicted = []
for batch in tqdm(test_loader): # bert
    batch = tuple(v.to(device) for v in batch)
    input_ids, token_type_ids, attention_masks = batch
    
    with torch.no_grad():
        out = model(input_ids=input_ids,
                    token_type_ids=token_type_ids,
                    attention_mask=attention_masks)
        
# for batch in tqdm(test_loader): # roberta
#     batch = tuple(v.to(device) for v in batch)
#     input_ids, attention_masks = batch
    
#     with torch.no_grad():
#         out = model(input_ids=input_ids,
#                     attention_mask=attention_masks)
        
        logits = out[0]
        
        _, pred = torch.max(logits, 1)
        predicted += pred.detach().cpu().numpy().tolist()

preds = le.inverse_transform(predicted)

submit = pd.read_csv('./data/sample_submission.csv')

submit['Target'] = preds

submit.to_csv('./data/submit.csv', index=False)