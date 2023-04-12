# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 22:18:54 2022

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
                          AutoModelForMaskedLM,
                          AutoModelForSequenceClassification
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
from custom_utils import *

seed_everything(42)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
pretrain = "bert-base-cased"
# pretrain = "tae898/emoberta-base"

EPOCHS = 10
batch_size = 32 # bert 32
warmup_ratio = 0.1
max_grad_norm = 1
max_length = 512
num_workers = 0
smoothing = 0.0
num_labels = 7


train_loader, val_loader, test_loader = dataloader(batch_size,
                                                   num_workers,
                                                   normal_data,
                                                   # make_new_data
                                                   # make_new_data_stack,
                                                   # make_new_data_stack2
                                                   )

total_steps = len(train_loader) * EPOCHS
warmup_step = int(total_steps * warmup_ratio)

model = BertForSequenceClassification.from_pretrained(pretrain, num_labels=num_labels,
                                                      # output_attentions=True,
                                                      # output_hidden_states = True,
                                                      # attention_probs_dropout_prob=0.5,
                                                      # hidden_dropout_prob=0.5,
                                                      ).to(device)

# model = XLMRobertaForSequenceClassification.from_pretrained("xlm-roberta-base",
#                                                             num_labels=num_labels
#                                                             ).to(device)

# model = BertForSequenceClassification.from_pretrained("nateraw/bert-base-uncased-emotion",
#                                                       num_labels=num_labels
#                                                       ).to(device)

optimizer = AdamW(model.parameters(), lr=3e-5) # bert
# optimizer = AdamW(model.parameters(), lr=1e-5) # roberta 1e-5, bert_em
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=warmup_step,
                                            num_training_steps=total_steps)
loss_ls = LabelSmoothingLoss(smoothing)

PATH = './model/'

for i in range(EPOCHS):
    total_loss = 0.0
    correct_train = 0
    correct_eval = 0
    total_train = 0
    total_eval = 0
    batches = 0
    
    f1_score = 0

    model.train()
    for batch in tqdm(train_loader): # bert
        batch = tuple(v.to(device) for v in batch)
        input_ids, token_type_ids, attention_masks, labels = batch
        labels = labels.type(torch.LongTensor)
        labels = labels.to(device)
        
        out = model(input_ids=input_ids,
                    token_type_ids=token_type_ids,
                    attention_mask=attention_masks,
                    labels = labels)
    # for batch in tqdm(train_loader): # bert_em
    #     batch = tuple(v.to(device) for v in batch)
    #     input_ids, attention_masks, labels = batch
    #     labels = labels.type(torch.LongTensor)
    #     labels = labels.to(device)
        
    #     out = model(input_ids=input_ids,
    #                 attention_mask=attention_masks,
    #                 labels = labels)
    #     ##########################################
    # for batch in tqdm(train_loader): # roberta
    #     batch = tuple(v.to(device) for v in batch)
    #     input_ids, attention_masks, labels = batch
    #     labels = labels.type(torch.LongTensor)
    #     labels = labels.to(device)

    #     out = model(input_ids=input_ids,
    #                 attention_mask=attention_masks,
    #                 labels = labels)
    #     ###########################################

        logits = out[1]
        
        loss = loss_ls(logits, labels)
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        model.zero_grad()
          
        total_loss += loss.item()
        
        _, predicted = torch.max(logits, 1)
        correct_train += (predicted == labels).sum()
        total_train += len(labels)

    print("")
    print("epoch {} Train Loss {:.6f} train acc {:.6f}".format(i+1,
                                                        torch.true_divide(total_loss, total_train),
                                                        torch.true_divide(correct_train, total_train)))
    print("")

    torch.save(model.state_dict(), PATH + 'Epoch_{}_loss_{:.4f}.pt'.format(i+1,
                                                                            torch.true_divide(total_loss,
                                                                                              total_train)))
    
    model.eval()
    for batch in tqdm(val_loader): # bert
        batch = tuple(v.to(device) for v in batch)
        input_ids, token_type_ids, attention_masks, labels = batch
        labels = labels.type(torch.LongTensor)
        labels = labels.to(device)        
        
        with torch.no_grad():
            out = model(input_ids=input_ids,
                        token_type_ids=token_type_ids,
                        attention_mask=attention_masks,
                        labels = labels)
            #############################################
    
    # for batch in tqdm(val_loader): # bert_em
    #     batch = tuple(v.to(device) for v in batch)
    #     input_ids, attention_masks, labels = batch
    #     labels = labels.type(torch.LongTensor)
    #     labels = labels.to(device)        
        
    #     with torch.no_grad():
    #         out = model(input_ids=input_ids,
    #                     attention_mask=attention_masks,
    #                     labels = labels)
    #         #############################################

    # for batch in tqdm(val_loader): # roberta
    #     batch = tuple(v.to(device) for v in batch)
    #     input_ids, attention_masks, labels = batch
    #     labels = labels.type(torch.LongTensor)
    #     labels = labels.to(device)        
        
    #     with torch.no_grad():
    #         out = model(input_ids=input_ids,
    #                     attention_mask=attention_masks,
    #                     labels = labels)
    #         #################################################
            
            logits = out[1]
            
            _, predicted = torch.max(logits, 1)
            correct_eval += (predicted == labels).sum()
            total_eval += len(labels)
    
            f1_score += competition_metric(labels.cpu().numpy(), predicted.cpu().numpy())
    f1_score /= len(val_loader)
    
    print("")
    print("epoch {} Test acc {:.6f} F1 score {:.6f}".format(i+1,
                                                            torch.true_divide(correct_eval, total_eval),
                                                            f1_score))  
                                                            