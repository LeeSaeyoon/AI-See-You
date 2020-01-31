import json
import os
import time
import numpy as np
import torchvision.models as models
import torch.nn.utils as torch_utils

from PIL import ImageFile
from collections import defaultdict
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import trange
from model.model_rnn_simple import *
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--phase', required=True, default='train', help='[train, test, val]')
parser.add_argument('--vocab_path', required=True, default='../data/vocab.korean_morp.list')
parser.add_argument('--image_path', required=True, default="../data/kts/total/")
parser.add_argument('--ckpt_path', required=True, default="./model_results/")
parser.add_argument('--epochs', required=True, default=10)
parser.add_argument('--batch_size', required=True, default=64)
parser.add_argument('--max_poem_lens', required=True, default=500
parser.add_argument('--max_keyword_counts', required=True, default=120)
parser.add_argument('--max_topic_len', required=True, default=120)
parser.add_argument('--rnn_hidden_dim', required=True, default=300)
parser.add_argument('--rnn_layer', required=True, default=2)
parser.add_argument('--rnn_bidre', required=True, default=False)# setting it True seems to break the data structure
parser.add_argument('--rnn_dropout', required=True, default=0)
parser.add_argument('--dense_dim', required=True, default=300)
parser.add_argument('--dense_h_dropout', required=True, default=0.5)

args = parser.parse_args()

ImageFile.LOAD_TRUNCATED_IMAGES = True

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
Tensor = FloatTensor

# Load KorBert Vocab
vocab2id, id2vocab = construct_vocab(file_=args.vocab_path)
vocab = {"w2i" : vocab2id, "i2w" : id2vocab}

transform = transforms.Compose([
    transforms.Resize([224,224]),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))])

# Load Data
data_total = PoetryRNNData(args.image_path, transform, vocab, vocab2id, args.max_poem_lens, args.args.max_keyword_counts)
all_data = data_total.reconstruct()
train, test, val = data_total.split(all_data)
train = [train[i]  for i in range(len(train)) if int(train[i]['keywords'][2]) != 0  ]
test = [test[i]  for i in range(len(test)) if int(test[i]['keywords'][2]) != 0  ]
val = [val[i]  for i in range(len(val)) if int(val[i]['keywords'][2]) != 0  ]

train = PoetryRNNDataSplit(transform, vocab, train, args.max_poem_lens, args.max_keyword_counts)
test = PoetryRNNDataSplit(transform, vocab, test, args.max_poem_lens, args.max_keyword_counts)
val = PoetryRNNDataSplit(transform, vocab, val, args.max_poem_lens, args.max_keyword_counts)

train_loader = DataLoader(train, batch_size=args.batch_size)

# Load Embedding
prt_emb = torch.load("../data/pytorch_model.bin") 
weights = prt_emb['../data/bert.embeddings.word_embeddings.weight']
embed_weights = torch.cat([weights[:5], weights[7:]], dim=0)

model = PoetryRNN(encoder_embed=embed_weights, num_pixels=196,
                  encoder_k_len=max_keyword_counts,
                  decoder_embed=embed_weights, rnn_hidden_dim=args.rnn_hidden_dim,
                  rnn_layers=rnn_layer,
                  rnn_bidre=rnn_bidre, rnn_dropout=rnn_dropout,
                  dense_dim=args.dense_dim, dense_h_dropout=dense_h_dropout,
                  freeze_embed=True)

# create model_results folder
try:
    if not os.path.exists("./model_results"):
        os.makedirs("./model_results")
except OSError:
    print("Error:Creating directory." + "./model_resutls")

# if you have saved model, load model
ckpt = args.ckpt_path
ckpt_model = os.path.join(ckpt, "rnn_model.pk")
ckpt_state_dict = os.path.join(ckpt, "model_check_point.pk")
ckpt_opt = os.path.join(ckpt, 'optimizer_check_point.pk')
ckpt_all_loss = os.path.join(ckpt,'all_loss_check_point.pk')


if os.path.isdir(ckpt):
    model = torch.load(ckpt_model)
    model.load_state_dict(torch.load(ckpt_state_dict, map_location='cpu'))
    optimizer.load_state_dict(torch.load(ckpt_opt))
    all_loss = torch.load(ckpt_all_loss)

    model.load_state_dict(torch.load(model_check_point, map_location='cpu'))
    model.train(True)
    print("Load previous training parameters.")

if use_cuda:
    model = model.cuda()
    print("Dump to cuda")


if args.phase == 'train':
    model.train(True)
    learning_rate = 1e-3
    loss_fn = nn.NLLLoss()
    model_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = optim.Adam(model_params, lr=learning_rate, amsgrad=True,weight_decay=0)

    if os.path.isfile("check_point_optim.pkl"):
        optimizer.load_state_dict(torch.load("check_point_optim.pkl"))
        print("Load previous optimizer.")

    lr_scheme = ReduceLROnPlateau(optimizer, mode='min', factor=0.4, patience=10, min_lr=1e-7,
                                verbose=True)

    counter = 0
    t_range = trange(epochs)
    for iteration in t_range:
        print("starting...")
        losses = []
        for batch in train_data:
            out = sort_batches(batch)
            model.zero_grad()
            target_score, hidden = model(out['img'],
                                        out['keyword'][0], out['keyword'][1], out['keyword'][2],
                                        #  out['x_pre'][0], out['x_pre'][1], out['x_pre'][2],
                                        out['x'][0], out['x'][1], out['x'][2])        # batch*seq_len x vocab dim
            loss = loss_fn(target_score, out['y'][0])

            loss.backward()
            torch_utils.clip_grad_norm_(model.parameters(), 0.25)
            optimizer.step()
            # lr_scheme.step(loss.data[0])
            print("Current batch {}, loss {}".format(counter, loss.item()))
            losses.append(loss.item())
            counter += 1
            all_loss.append(loss.item())

        one_ite_loss = np.mean(losses)
        lr_scheme.step(one_ite_loss)
        print("One iteration loss {:.3f}".format(one_ite_loss))
        # Need to add Validation code

    torch.save(model, ckpt_model)
    torch.save(model.state_dict(), ckpt_state_dict)
    torch.save(optimizer.state_dict(), ckpt_opt)
    torch.save(all_loss, ckpt_all_loss)


if args.phase == 'test':
    # for test