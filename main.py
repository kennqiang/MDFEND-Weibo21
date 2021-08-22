import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default='mdfend')#textcnn bigru bert eann eddfn mmoe mose mdfend
parser.add_argument('--split_type', default='random')
parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--max_len', type=int, default=170)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--early_stop', type=int, default=3)
parser.add_argument('--bert_vocab_file', default='/data/wyy/pretrain_model/chinese_roberta_wwm_base_ext_pytorch/vocab.txt') 
parser.add_argument('--root_path', default='/data/zhuyongchun/data/mdfnd/') #data path
parser.add_argument('--bert_emb', default='/data/wyy/pretrain_model/chinese_roberta_wwm_base_ext_pytorch')  
parser.add_argument('--batchsize', type=int, default=64)
parser.add_argument('--seed', type=int, default=2021)
parser.add_argument('--gpu', default='0')
parser.add_argument('--bert_emb_dim', type=int, default=768)
parser.add_argument('--w2v_emb_dim', type=int, default=200)
parser.add_argument('--lr', type=float, default=5e-5)
parser.add_argument('--emb_type', default='bert') #bert or w2v
parser.add_argument('--w2v_vocab_file', default='/data/nanqiong/weibo20/pretrain_model/w2v/Tencent_AILab_Chinese_w2v_model.kv')
parser.add_argument('--log_dir', default= './logs')
parser.add_argument('--model_param_dir', default= './param_model') #saved model path
parser.add_argument('--param_results_dir', default = './logs/param')  #saved parameter and corresponding results path

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

from grid_search import Run
import torch
import numpy as np
import random

seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

if args.emb_type == 'bert':
    emb_dim = args.bert_emb_dim
    vocab_file = args.bert_vocab_file
elif args.emb_type == 'w2v':
    emb_dim = args.w2v_emb_dim
    vocab_file = args.w2v_vocab_file

print('lr: {}; model name: {}; batchsize: {}; epoch: {}; gpu: {}; emb_dim: {}'.format(args.lr, args.model_name, args.batchsize, args.epoch, args.gpu, emb_dim))

config = {
        'use_cuda': True,
        'batchsize': args.batchsize,
        'max_len': args.max_len,
        'early_stop': args.early_stop,
        'num_workers': args.num_workers,
        'vocab_file': vocab_file,
        'emb_type': args.emb_type,
        'split_type': args.split_type,
        'bert_emb': args.bert_emb,
        'root_path': args.root_path,
        'weight_decay': 5e-5,
        'model':
            {
            'mlp': {'dims': [384], 'dropout': 0.2}
            },
        'emb_dim': emb_dim,
        'lr': args.lr,
        'epoch': args.epoch,
        'model_name': args.model_name,
        'seed': args.seed,
        'log_dir': args.log_dir,
        'model_param_dir': args.model_param_dir,
        'param_results_dir': args.param_results_dir
        }

if __name__ == '__main__':
    Run(config = config
        ).main()

