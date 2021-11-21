import torch
import random
import pandas as pd
import tqdm
import numpy as np
import pickle
import re
import jieba
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader
from gensim.models.keyedvectors import KeyedVectors, Vocab

def _init_fn(worker_id):
    np.random.seed(2021)

def read_pkl(path):
    with open(path, "rb") as f:
        t = pickle.load(f)
    return t

def df_filter(df_data):
    df_data = df_data[df_data['category'] != '无法确定']
    return df_data

def word2input(texts, vocab_file, max_len):
    tokenizer = BertTokenizer(vocab_file=vocab_file)
    token_ids = []
    for i, text in enumerate(texts):
        token_ids.append(
            tokenizer.encode(text, max_length=max_len, add_special_tokens=True, padding='max_length',
                             truncation=True))
    token_ids = torch.tensor(token_ids)
    masks = torch.zeros(token_ids.shape)
    mask_token_id = tokenizer.pad_token_id
    for i, tokens in enumerate(token_ids):
        masks[i] = (tokens != mask_token_id)
    return token_ids, masks

class bert_data():
    def __init__(self, max_len, batch_size, vocab_file, category_dict, num_workers=2):
        self.max_len = max_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.vocab_file = vocab_file
        self.category_dict = category_dict
    
    def load_data(self, path, shuffle):
        self.data = df_filter(read_pkl(path))
        content = self.data['content'].to_numpy()
        label = torch.tensor(self.data['label'].astype(int).to_numpy())
        category = torch.tensor(self.data['category'].apply(lambda c: self.category_dict[c]).to_numpy())
        content_token_ids, content_masks = word2input(content, self.vocab_file, self.max_len)
        dataset = TensorDataset(content_token_ids,
                                content_masks,
                                label,
                                category
                                )
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=shuffle,
            worker_init_fn=_init_fn
        )
        return dataloader

class w2v_data():
    def __init__(self, max_len, batch_size, emb_dim, vocab_file, category_dict, num_workers = 2):
        self.max_len = max_len
        self.batch_size = batch_size
        self.emb_dim = emb_dim
        self.vocab_file = vocab_file
        self.category_dict = category_dict
        self.num_workers = num_workers
    
    def tokenization(self, content):
        pattern = "&nbsp;|展开全文|秒拍视频|O网页链接|网页链接"
        repl = ""
        tokens = []
        for c in content:
            c = re.sub(pattern, repl, c, count = 0)
            cut_c = jieba.cut(c, cut_all = False)
            words = [word for word in cut_c]
            tokens.append(words)
        return tokens
    
    def get_mask(self, tokens):
        masks = []
        for token in tokens:
            if(len(token)< self.max_len):
                masks.append([1] * len(token) + [0] * (self.max_len - len(token)))
            else:
                masks.append([1] * self.max_len)
        
        return torch.tensor(masks)
    
    def encode(self, token_ids):
        w2v_model = KeyedVectors.load(self.vocab_file)
        embedding = []
        for token_id in token_ids:
            words = [w for w in token_id[: self.max_len]]
            words_vec = []
            for word in words:
                words_vec.append(w2v_model[word] if word in w2v_model else np.zeros([self.emb_dim]))
            for i in range(len(words_vec), self.max_len):
                words_vec.append(np.zeros([self.emb_dim]))
            embedding.append(words_vec)
        return torch.tensor(np.array(embedding, dtype = np.float32))

    def load_data(self, path, shuffle = False):
        self.data = df_filter(read_pkl(path))
        content = self.data['content'].to_numpy()
        label = torch.tensor(self.data['label'].astype(int).to_numpy())
        category = torch.tensor(self.data['category'].apply(lambda c: self.category_dict[c]).to_numpy())

        content_token_ids = self.tokenization(content)
        content_masks = self.get_mask(content_token_ids)
        emb_content = self.encode(content_token_ids)

        dataset = TensorDataset(emb_content,
                                content_masks,
                                label,
                                category
                                )
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=shuffle
        ) 
        return dataloader
