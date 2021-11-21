#!/usr/bin/python
# -*- coding: utf-8 -*-

# from gensim.models.keyedvectors import Vocab
# from transformers.file_utils import CONFIG_NAME
from utils.dataloader import w2v_data
import torch
import tqdm
import pickle
import logging
import os
import time
import json
from copy import deepcopy

from utils.utils import Averager
from utils.dataloader import bert_data
from models.mdfend import Trainer as MDFENDTrainer

class Run():
    def __init__(self,
                 config
                 ):
        self.configinfo = config

        self.use_cuda = config['use_cuda']
        self.model_name = config['model_name']
        self.lr = config['lr']
        self.batchsize = config['batchsize']
        self.emb_type = config['emb_type']
        self.emb_dim = config['emb_dim']
        self.max_len = config['max_len']
        self.num_workers = config['num_workers']
        self.vocab_file = config['vocab_file']
        self.early_stop = config['early_stop']
        self.bert = config['bert']
        self.root_path = config['root_path']
        self.mlp_dims = config['model']['mlp']['dims']
        self.dropout = config['model']['mlp']['dropout']
        self.seed = config['seed']
        self.weight_decay = config['weight_decay']
        self.epoch = config['epoch']
        self.save_param_dir = config['save_param_dir']

        self.train_path = self.root_path + 'train.pkl'
        self.val_path = self.root_path + 'val.pkl'
        self.test_path = self.root_path + 'test.pkl'

        self.category_dict = {
            "科技": 0,  
            "军事": 1,  
            "教育考试": 2,  
            "灾难事故": 3,  
            "政治": 4,  
            "医药健康": 5,  
            "财经商业": 6,  
            "文体娱乐": 7,  
            "社会生活": 8
        }

    def get_dataloader(self):
        if self.emb_type == 'bert':
            loader = bert_data(max_len = self.max_len, batch_size = self.batchsize, vocab_file = self.vocab_file,
                        category_dict = self.category_dict, num_workers=self.num_workers)
        elif self.emb_type == 'w2v':
            loader = w2v_data(max_len=self.max_len, vocab_file=self.vocab_file, emb_dim = self.emb_dim,
                    batch_size=self.batchsize, category_dict=self.category_dict, num_workers= self.num_workers)
        train_loader = loader.load_data(self.train_path, True)
        val_loader = loader.load_data(self.val_path, False)
        test_loader = loader.load_data(self.test_path, False)
        return train_loader, val_loader, test_loader
    
    def config2dict(self):
        config_dict = {}
        for k, v in self.configinfo.items():
            config_dict[k] = v
        return config_dict

    def main(self):
        train_loader, val_loader, test_loader = self.get_dataloader()
        if self.model_name == 'mdfend':
            trainer = MDFENDTrainer(emb_dim = self.emb_dim, mlp_dims = self.mlp_dims, bert = self.bert, emb_type = self.emb_type,
                use_cuda = self.use_cuda, lr = self.lr, train_loader = train_loader, dropout = self.dropout, weight_decay = self.weight_decay, val_loader = val_loader, test_loader = test_loader, category_dict = self.category_dict, early_stop = self.early_stop, epoches = self.epoch,
                save_param_dir = os.path.join(self.save_param_dir, self.model_name))    
        trainer.train()
