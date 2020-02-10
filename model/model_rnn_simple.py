import logging
from itertools import chain
from collections import defaultdict
import numpy as np
import torch
import pickle
from PIL import Image
import os
import re
import tqdm
from glob import glob
import json
from sklearn.model_selection import train_test_split
from torch import nn
from torchvision import transforms
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F
import torchvision.models as models

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
Tensor = FloatTensor


# +++++++++++++++++++++++++++++++++++++
#           Prepare Dataloader      
# -------------------------------------
image_path = "data/kts/total/"

class PoetryRNNData(Dataset):
    def __init__(self, image_path, transform, vocab, vocab2id, max_poem_lens, max_keyword_counts):
        # when chunk size = 120, evenly div
        self.vocab = vocab
        self.transform = transform
        self.vocab2id = vocab2id
        self.train_test_flag = "all"
        
        
        self.image_path_list = []
        self.json_path_list = []
        self.json_path_list_lens = []
        
        self.keywords = []
        self.dec_data = []
        
#         self.category = category
        self.category = {"nature-scene": ["beach", "cave", "island", "lake", "mountain"],
             "person-made": ["amusement park", "palace", "park", "restaurant", "tower"]}

        for key, val in self.category.items():
            for i ,v in enumerate(val):
                paths_img = glob(os.path.join((image_path + key + '/' + v + '/images/'+'*jpg')))
                paths_json= glob(os.path.join((image_path + key + '/' + v +'/*json')))[0]
        

                with open(paths_json, encoding='UTF-16') as f:
                    js = json.load(f)
                   
                f.close()
                
                self.json_path_list_lens = []
                self.json_path_list.append(paths_json) #json list
                self.image_path_list.extend(paths_img) #image list

                # print("IMG NAME ", paths_img[0].split('\\')[-1].split('.')[0]) #image_name

                for j in range(len(paths_img)):
                    # print(paths_img[j].split('\\')[-1].split('.')[0])
                    # print(js[j]['img_name'])
                    for k in range(len(js)):
                        # print(paths_img[j].split('\\')[-1].split('.')[0])
                        if int(paths_img[j].split('\\')[-1].split('.')[0].split('/')[-1]) == int(js[k]['image']):
                            # JSON 내에 있는 정보를 모두 list에 저장
                            keyword_arr = []
                            keyword_lens = []
                            dabs = js[k]['keyword']
                            dabs = list(filter(None, dabs))
                            keyword_lens.append(len(dabs))

                            keywords = [
                                self.vocab2id[wd] if wd in self.vocab2id
                                else self.vocab2id['[UNK]']
                                for wd in dabs
                            ]


                            trg_arr = []
                            trg_lens = []
                            arr = js[k]['text'][:-1]
                            dabs = re.split('\s', arr)
                            dabs = list(filter(None, dabs)) + ['[STOP]']
                            trg_lens.append(len(dabs))
                            
                            text = [
                                self.vocab2id[wd] if wd in self.vocab2id
                                else self.vocab2id['[UNK]']
                                for wd in dabs
                            ]
                            self.keywords.append(keywords)
                            self.dec_data.append(text)
                
       
        self.lens = len(self.dec_data)
        
        # 형태소 기반인 경우 max_keyword_counts는 키워드 개수를 말함.
        # self.keywords = ["단어1", "단어2", ..., ]
        self.keywords_len = torch.LongTensor([min(len(x), max_keyword_counts) for x in self.keywords])
        self.keyword_pad = torch.zeros((self.lens, max_keyword_counts)).long()

        self.dec_len = torch.LongTensor([min(len(x), max_poem_lens) -1 for x in self.dec_data])
        self.lens = len(self.dec_len)

        # pad data
        # max_len 채워질 때 까지 패딩해 주는거
        max_len = min(max(self.dec_len)+1, max_poem_lens) 
        
        self.dec_x_pad = [
            itm[:-1] + [self.vocab2id['[PAD]']]*(1+int(max_len)-len(itm))
            for itm in self.dec_data
        ]
        self.dec_y_pad = [
            itm[1:] + [self.vocab2id['[PAD]']]*(1+int(max_len)-len(itm))
            for itm in self.dec_data
        ]
        
        self.dec_x_pad = Variable(torch.LongTensor(self.dec_x_pad))
        self.dec_y_pad = Variable(torch.LongTensor(self.dec_y_pad))
        # 여기까지 하면 dec_x_pad, dec_y_pad에 각각 패딩 된 임베딩값들 할당 됩니다
        
        # 이 밑에 코드는 모르겠어요 ㅠㅠㅠ
        for i in range(self.lens):
            n_keywords = self.keywords_len[i]

            self.keyword_pad[i, :n_keywords] = torch.LongTensor(self.keywords[i][:n_keywords])
     # Next character

        if use_cuda:
            self.keywords_len = self.keywords_len
            self.dec_len = self.dec_len.cuda()

            self.keyword_pad = self.keyword_pad.cuda()
            self.dec_x_pad = self.dec_x_pad.cuda()
            self.dec_y_pad = self.dec_y_pad.cuda()


    
    def __len__(self):
        return self.lens
    
    def __getitem__(self, index):

        # imgaes
        image = self.image_path_list[index]
        image = Image.open(image)
        image = image.convert('RGB')
        image = self.transform(image)

        # Poet Data & Target
        self.dec_x_pad[index, :], self.dec_y_pad[index, :], self.dec_len[index]
        
        out = {'image' : image,
                  'keyword' : [self.keywords[index, :], self.keywords_len[index]],
                  'dec_x' : [self.dec_x_pad[index, :], self.dec_len[index]],
                  'dec_y' : [self.dec_y_pad[index, :]]}
        
        return out
    
    def reconstruct(self):
        # 모든 데이터를 all_data 리스트에 저장해
        # 각 클래스별로 train, test split할 수 있게함
        all_data = defaultdict(list)
        for i in tqdm.tqdm(range(self.lens)):
            ctg = self.image_path_list[i].split('/')[-3]
            all_data[ctg].append({
            'img' : self.image_path_list[i],
            'keywords' : [self.keywords[i],
            self.keyword_pad[i],
            self.keywords_len[i]],
            'dec' : [self.dec_data[i],
            self.dec_len[i],
            self.dec_x_pad[i],
            self.dec_y_pad[i]]
            })
        # all_data = {ctg : [{}, {}, ... ]}
        return all_data
    
    def split(self, all_data):
        # all_data를 train, test, val로 분할
        train = []
        val = []
        test = []
        for k, v in self.category.items():
        #     print("key", k)
            for ctg in v:
        #         print("c", c)
                train_, test_ = train_test_split(all_data[ctg], test_size=0.2, train_size=0.8, random_state=4)
                test_, val_ = train_test_split(test_, test_size=0.5, train_size=0.5, random_state=4)
                train.extend(train_)
                val.extend(val_)
                test.extend(test_)
        return train, val, test

#     def preprocess_data(self, poem_i, vocab2id):
#         # poem_i : json_data[i]
#         img = poem_i['image']
        
#         keyword_arr = []
#         keyword_lens = []
#         dabs = poem_i['keyword']
#         dabs = list(filter(None, dabs))
#         keyword_lens.append(len(dabs))

#         keywords = [
#             vocab2id[wd] if wd in vocab2id
#             else vocab2id['[UNK]']
#             for wd in dabs
#         ]
        
        
#         trg_arr = []
#         trg_lens = []
#         arr = poem_i['text'][:-1]
#         dabs = re.split('\s', arr)
#         dabs = list(filter(None, dabs)) + ['[STOP]']
#         trg_lens.append(len(dabs))

#         dec_data = [
#             vocab2id[wd] if wd in vocab2id
#             else vocab2id['[UNK]']
#             for wd in dabs
#         ]
#         # encoder, decoder 데이터로 분리
        
#         return img, keywords, dec_data

class PoetryRNNDataSplit(Dataset):
    def __init__(self, transform, vocab, list_data, max_poem_lens, max_keyword_counts):
        # 위에서 리스트 형태로 나온 데이터를 입력받아
        # 학습 위한 데이터셋으로
        self.vocab = vocab
        self.transform = transform
        
        self.image_path_list = None
        self.json_path_list = None

        self.keywords = None
        self.keyword_pad = None
        self.keywords_len = None

        self.dec_data = None
        self.dec_len = None
        self.dec_x_pad = None
        self.dec_y_pad = None
        

        self.load_list_data(list_data)
        
        self.lens = len(self.dec_data)
        
    def __len__(self):
        return self.lens

    
    def __getitem__(self, index):

        # imgaes
        image = self.image_path_list[index]
        image = Image.open(image)
        image = self.transform(image)

        # Poet Data & Target
        sample = {'img' : image,
                  'keyword' : [self.keyword_pad[index, :], self.keywords_len[index]],
                  'dec_x' : [self.dec_x_pad[index, :], self.dec_len[index]],
                  'dec_y' : [self.dec_y_pad[index, :]]}
        
        return sample

    def load_list_data(self, list_data):
    # 각 리스트별로 저장된 것을 다시 Dastset의 각 인스턴스변수 형태로 변경
        # for ctg in self.category['person-made']:

        self.image_path_list = []
        self.keywords = []
        self.keyword_pad = []
        self.keywords_len = []
        self.dec_data = []
        self.dec_len = []
        self.dec_x_pad = []
        self.dec_y_pad = []

        for d in list_data:
            self.image_path_list.append(d['img'])
            self.keywords.append(d['keywords'][0])
            self.keyword_pad.append(d['keywords'][1])
            self.keywords_len.append(d['keywords'][2])
            self.dec_data.append(d['dec'][0])
            self.dec_len.append(d['dec'][1])
            self.dec_x_pad.append(d['dec'][2])
            self.dec_y_pad.append(d['dec'][3])
        
        keyword_pad_len = len(self.keyword_pad)
        dec_x_pad_len = len(self.dec_x_pad)
        dec_y_pad_len = len(self.dec_y_pad)
        self.keyword_pad = torch.cat(self.keyword_pad).reshape(keyword_pad_len, -1)
        self.dec_x_pad = torch.cat(self.dec_x_pad).reshape(dec_x_pad_len,-1)
        self.dec_y_pad = torch.cat(self.dec_y_pad).reshape(dec_y_pad_len,-1)
        self.lens = len(self.dec_data)


def construct_vocab(file_, max_size=50000, mincount=5):
    vocab2id = {'[CLS]': 2, '[SEP]': 3, '[PAD]': 0, '[UNK]': 1, '[STOP]': 4}
    id2vocab = {2: '[CLS]', 3: '[SEP]', 0: '[PAD]', 1: '[UNK]', 4: '[STOP]'}
    word_pad = {'[CLS]': 2, '[SEP]': 3, '[PAD]': 0, '[UNK]': 1, '[STOP]': 4}

    cnt = len(vocab2id)
    with open(file_, 'r',encoding='utf-8') as fp:
        for line in fp:
            arr = re.split('\t', line[:-1])
            if (arr[0] in [' ', 'n_iters=10000', 'max_length=16', '[MASK]','<S>','<T>']) :
                continue
            if arr[0] in word_pad:
                continue
            if int(arr[1]) >= mincount:
                vocab2id[arr[0]] = cnt
                id2vocab[cnt] = arr[0]
                cnt += 1
            if len(vocab2id) == max_size:
                  break

    return vocab2id, id2vocab


def prepare_data_loader(image_path, transform, vocab, poem, max_keyword_counts, max_poem_lens, batch_size, shuffle):
    dataset = PoetryRNNData(image_path, transform, vocab, poem, max_poem_lens, max_keyword_counts)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

def sort_batches(batches):
    img = batches['img'].cuda()
    ks, ks_len = batches['keyword']
    x, x_lengths = batches['dec_x']
    y = batches['dec_y'][0]
    
    # keyword sort & invert
    sorted_ks_len, sorted_ks_idx = ks_len.sort(dim=0, descending=True)
    sorted_ks_idx = sorted_ks_idx.cuda()
    sorted_ks_len = sorted_ks_len.cuda()
    print(ks.index_select(dim=0, index=sorted_ks_idx.squeeze()))
    sorted_ks = ks.index_select(dim=0, index=sorted_ks_idx.squeeze())
#     sorted_ks = sorted_ks.cuda()
    pad_ks_lens = sorted_ks_len.squeeze()
    _, inverted_ks_idx = sorted_ks_idx.sort(dim=0, descending=False)
    inverted_ks_idx = inverted_ks_idx.cuda()

# 아래는 시의 이전 행과 현재 행 정보
# encoder RNN 내부에서 계산될 때는 Batch상의 순서가 무관하지만
# attention 계산 전에 원래의 Batch 순서대로 돌려놓아야 함.

    # x : 현재 행
    sorted_lens, sorted_idx = x_lengths.sort(dim=0, descending=True)
    sorted_idx = sorted_idx.squeeze()
    sorted_x = x.index_select(dim=0, index=sorted_idx)  # x[sorted_idx, :]
    sorted_y = y.index_select(dim=0, index=sorted_idx)  # y[sorted_idx, :]
    _, inverted_idx = sorted_idx.sort(dim=0, descending=False)

    pad_len = sorted_lens.squeeze()
    
    # Target값은 [batch*seq_len, hidden] 으로 reshape 됨. 빠른 loss 계산 위함
    unpad_y = [sorted_y[i, :pad_len[i]] for i in range(len(pad_len))]
    unpad_y = torch.cat(unpad_y)

    # 최종적으로 loss를 계산할 때는 Decoder Input이 sort된대로 계산하므로
    # 나머지 image와 keyword에 대해서 원래대로 배치 순서대로 되돌린 것을
    # Decoder Input Length 내림차 순서대로 맞춰 정렬해야 Attention이 제대로 적용됨
    # 아래는 이를 위한 인덱스를 구한 것으로 각각의 encoder 연산 후 되돌리는 작업이 진행되어야함
    inverted_ks_idx = inverted_ks_idx[sorted_idx]
    
    # 이미지는 decoder의 순서에 맞게 sorting
    img = img.index_select(dim=0, index=sorted_idx)
    img = img.contiguous()
    sorted_x = sorted_x.contiguous()
    unpad_y = unpad_y.contiguous()

    # if use_cuda:
    #    sorted_x = sorted_x.cuda()
    #    unpad_y = unpad_y.cuda()

    out = {"img" : img, "keyword" : [sorted_ks, pad_ks_lens, inverted_ks_idx],
           "x" : [sorted_x, pad_len, inverted_idx], # x는 sort할 때 썼던 idx가 나온다.
           "y" : unpad_y}
    return out

def unsort_batches(self, input, invert_order):
    '''
    Recover the origin order

    Input:
            input:        batch_size-num_zero, seq_len, hidden_dim
            invert_order: batch_size
    Output:
            out:   batch_size, seq_len, *
    '''
    input = input.index_select(dim=0, index=invert_order.squeeze()).contiguous()
    return input

# +++++++++++++++++++++++++++++++++++++
#           Prepare RNN model      
# -------------------------------------

# ##1. VisualEncoder() 작성하여 아래 코드에 반영하기
# ##2. Batch에서 sort가 Poem 부분과 Topic 부분이 맞지 않으므로 맞추어줄 것.
# ##3. 키워드는?
# ##4. Dataloader 코드는?

class VisualEncoder(nn.Module):
    def __init__(self, embed_size):
        """Load the pretrained vgg19 and replace over the conv_5_4 layer."""
        super(VisualEncoder, self).__init__()
        vgg19 = models.vgg19(pretrained=True)
        modules = list(vgg19.children())[0][:-2]
        self.vgg19 = nn.Sequential(*modules)
        self.linear = nn.Linear(196, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        
    def forward(self, images):
        """Extract feature vectors from input images."""
        with torch.no_grad():
            features = self.vgg19(images)
        print("before", features.shape)
        features = features.reshape(features.size(0), 512, -1).permute(0,2,1).contiguous()
        print(features.shape)
        return features

class PoetryEncoder(nn.Module):
    def __init__(self, encoder_embed,
                 rnn_hidden_dim, rnn_layers,
                 freeze_embed=False):
        super(PoetryEncoder, self).__init__()
        self.embed = nn.Embedding.from_pretrained(encoder_embed, freeze=freeze_embed)
        self.vocab_dim, self.embed_hidden = encoder_embed.size()

        # GRU: the output looks similar to hidden state of standard RNN
        self.rnn_hidden_dim = rnn_hidden_dim // 2
        self.rnn_layers = rnn_layers
        self.rnn = nn.GRU(self.embed_hidden, self.rnn_hidden_dim, batch_first=True,
                          num_layers=self.rnn_layers, bidirectional=True)

        # attention

    def forward(self, batch_input, sorted_lens, jamo=True):
        '''
        batch_input:   batch, seq_len -
        sorted_lens:   batch,
        embed output:   batch, seq_len,embed_dim
        pack_pad_seq input: batch, seq_len, *
        rnn input :  batch, seq_len, input_size;
        output: seq_len, batch, rnn_hidden_dim
        pad_pack_seq output: seq_len, batch, *
        '''
        word_vec = self.embed(batch_input)
        print(word_vec)
        word_vec = pack_padded_sequence(word_vec, sorted_lens.tolist(), batch_first=True)
        rnn_out, hidden = self.rnn(word_vec)  # hidden : layer*direction, batch, hidden dim
        rnn_out, _ = pad_packed_sequence(rnn_out, batch_first=True, total_length=batch_input.size(1))
        hidden = self.reshape_bidirec_hidden(hidden)
        return rnn_out, hidden

    def merge_bidirec_hidden(self, hidden_state):
        # in bi-directions layers, each layer contains 2 directions of hidden states, so
        # take their average for each layer
        h = hidden_state
        h = torch.cat(list(((h[i] + h[i + 1]) / 2).unsqueeze(0) for i in range(0, self.rnn_layers * 2, 2)))
        # c = torch.cat(list(((c[i] + c[i + 1])/2).unsqueeze(0) for i in range(0, self.rnn_layers * 2, 2)))
        return (h, h)

    def reshape_bidirec_hidden(self, hidden_state):
        h = hidden_state
        num_layers, batch, hidden_size = h.size()
        h = h.reshape(num_layers // 2, batch, -1)

        # c = torch.zeros_like(h)
        return (h, h)


class KeywordAttention(nn.Module):
    def __init__(self, encoder_len, encoder_hidden_dim, decoder_embed_hidden, attention_dropout=0.1):
        super(KeywordAttention, self).__init__()
        print(encoder_hidden_dim)
        print(decoder_embed_hidden)
        self.atten_weights = nn.Linear(encoder_hidden_dim, decoder_embed_hidden)
        self.softmax = nn.Softmax(dim=2)
        self.context_out = nn.Linear(encoder_len + decoder_embed_hidden, decoder_embed_hidden)
        self.dropout = nn.Dropout(attention_dropout)
        self.activation_out = nn.SELU()

    def forward(self, decoder_input, encoder_output):
        # decoder_input: batch, seq_len, embedding_hidden
        # rnn_output: batch, seq_len, rnn_hidden
        # encoder_output: batch, topic_len, rnn_hidden
        # context_state = encoder_hidden[0].t()  # --> batch, num_layer, hidden_dim
        context_state = self.dropout(encoder_output)
        attention = self.atten_weights(context_state).transpose(1, 2)  # --> batch, decoder_embed_hidden, topic_len

        attention_w = decoder_input.bmm(attention)  # batch, seq_len, topic_len
        attention = self.softmax(attention_w)

        context_concat = torch.cat([decoder_input, attention], dim=2)  # batch, seq_len, topic_len+decoder_embed_hidden
        out = self.context_out(context_concat)  # batch, seq_len, decoder_embed_hidden
        out = self.activation_out(out)
        return out

class VisualAttention(nn.Module):
    def __init__(self, num_pixels, encoder_hidden_dim, decoder_embed_hidden, attention_dropout=0.1):
        super(VisualAttention, self).__init__()
        '''
        Using Visual Features
        '''
        self.atten_weights = nn.Linear(encoder_hidden_dim, decoder_embed_hidden)
        self.softmax = nn.Softmax(dim=2)
        self.context_out = nn.Linear(num_pixels + decoder_embed_hidden, decoder_embed_hidden)
        self.dropout = nn.Dropout(attention_dropout)
        self.activation_out = nn.SELU()

    def forward(self, decoder_input, v_encoder_output):
        # decoder_input: batch, seq_len, embedding_hidden
        # rnn_output: batch, seq_len, rnn_hidden
        # visual_encoder_output: batch, num_pixels, cnn_dim
        # context_state = encoder_hidden[0].t()  # --> batch, num_layer, hidden_dim
        context_state = self.dropout(v_encoder_output)
        attention = self.atten_weights(context_state).transpose(1, 2)  # --> batch, decoder_embed_hidden, num_pixels
        
        attention_w = decoder_input.bmm(attention)  # batch, seq_len, num_pixels
        attention = self.softmax(attention_w)

        context_concat = torch.cat([decoder_input, attention], dim=2)  # batch, seq_len, embed_hidden + num_pixels
        out = self.context_out(context_concat)  # batch, seq_len, decoder_embed_hidden
        out = self.activation_out(out)
        return out

class PoetryDecoder(nn.Module):
    def __init__(self, decoder_embed,
                 rnn_hidden_dim, rnn_layers, rnn_bidre, rnn_dropout,
                 dense_dim, dense_h_dropout,
                 freeze_embed=False):
        super(PoetryDecoder, self).__init__()

        # pre-trained word embedding
        self.embed = nn.Embedding.from_pretrained(decoder_embed, freeze=freeze_embed)
        self.vocab_dim, self.embed_hidden = decoder_embed.size()
        
        # Propotion of Image, Preceding Lines, Keywords of Image
        self.alpha = nn.Parameter(torch.tensor(1, dtype=torch.float))
        self.beta = nn.Parameter(torch.tensor(1, dtype=torch.float))
        self.gamma = nn.Parameter(torch.tensor(1, dtype=torch.float))
        self.tanh = nn.Tanh()
        

        # LSTM
        self.rnn_hidden_dim = rnn_hidden_dim // 2 if rnn_bidre else rnn_hidden_dim
        self.rnn_layers = rnn_layers
        self.bi_direc = 2 if rnn_bidre else 1
        self.rnn = nn.LSTM(self.embed_hidden, self.rnn_hidden_dim, batch_first=True,
                           num_layers=rnn_layers, bidirectional=rnn_bidre, dropout=rnn_dropout)

        # self.init_rnn_xavier_normal()

        # dense hidden layers
        self.dense_h_dropout = dense_h_dropout
        self.dense_h0 = self.dense_layer(rnn_hidden_dim, dense_dim, nn.SELU(), dropout=True)

        self.dense_h1 = self.dense_layer(dense_dim, dense_dim, nn.SELU())
        self.dense_h2 = self.dense_layer(dense_dim, dense_dim, nn.SELU())
        self.dense_h3 = self.dense_layer(dense_dim, dense_dim, nn.SELU())

        # output layer
        self.output_linear = nn.Linear(dense_dim, self.vocab_dim)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward_(self, batch_input, sorted_lens,
                    encoded_output_v,
                    # encoded_output, 
                 encoded_hidden,
                    encoded_output_k,
                    attention_v,
                    # attention,
                    attention_k):
        # batch_input:   batch, seq_len -
        # sorted_lens:   batch,
        # encoder_output_v: batch, num_pixels, img_feature_dim
        # encoder_output: batch, pre_len, hidden_dim
        # encoded_hidden: (h,c), h: num_layers, batch, hidden_dim (concat)
        # encoder_output_k: batch, keyword_len, hidden_dim
        # attention_v : batch, seq_len(decoder's), hidden_dim
        # attention : batch, seq_len(decoder's), hidden_dim
        # attention_k : batch, seq_len(decoder's), hidden_dim

        # embed output:   batch, seq_len,embed_dim
        # pack_pad_seq input: batch, Seq_len, *
        # rnn input :  batch, seq_len,input_size; output: seq_len, batch, rnn_hidden_dim
        # pad_pack_seq output: seq_len, batch, *

        word_vec = self.embed(batch_input)
        # mask out zeros for padded 0s
        word_vec = self.mask_zeros(word_vec, sorted_lens)

        # attention
        word_vec_v = attention_v.forward(word_vec, encoded_output_v)
        # word_vec = attention.forward(word_vec, encoded_output)
        word_vec_key = attention_k.forward(word_vec, encoded_output_k)

        # attention propotions
        # word_vec = self.alpha*word_vec_v + self.beta*word_vec + self.gamma*word_vec_key
        word_vec = self.tanh(self.alpha*word_vec_v + self.beta*word_vec_key + self.gamma)
        
        # RNN
        word_vec = pack_padded_sequence(word_vec, sorted_lens.tolist(), batch_first=True)
        rnn_out, hidden = self.rnn(word_vec, encoded_hidden)  # hidden : layer*direction, batch, hidden dim
        rnn_out, _ = pad_packed_sequence(rnn_out, batch_first=True, total_length=batch_input.size(1))

        # attention

        # dense layers
        unpad = [rnn_out[i, :sorted_lens[i], :] for i in range(len(sorted_lens))]
        # Linear > LayerNorm > Activation > Linear > 
        decoded = [self.forward_dense(x) for x in unpad]

        # final output
        return decoded, hidden

    def forward(self,batch_input, sorted_lens,
                    encoded_output_v,
                    encoded_hidden,
                    encoded_output_k,
                    attention_v,
                    
                    attention_k):
        decoded, hidden = self.forward_(batch_input, sorted_lens,
                                        encoded_output_v,
                                         encoded_hidden,
                                        encoded_output_k, 
                                        attention_v,
                                        
                                        attention_k)
 
        
        target_score = [self.log_softmax(x) for x in decoded]
        target_score = torch.cat(target_score)  # batch*seq_len, vocab_size
        return target_score, hidden

    def forward_dense(self, rnn_out):
        # hidden layers
        dense_h0 = self.dense_h0(rnn_out)
        # dense_h1 = self.dense_h1(dense_h0)
        # dense_h2 = self.dense_h2(dense_h1)
        # dense_h3 = self.dense_h3(dense_h2)
        #
        # denseout = dense_h1 + dense_h3
        # output layer
        decoded = self.output_linear(dense_h0)
        return decoded

    def dense_layer(self, input_dim, output_dim, activation, dropout=True):
        dense = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            activation
        )
        if dropout:
            dense.add_module("Dropout", nn.Dropout(self.dense_h_dropout))
        return dense

    @staticmethod
    def mask_zeros(word_vec, sorted_lens):
        # word_vec: batch, seq_len, embed_dim
        # Each example has different lengths, but the padded 0s have value after the embedding layer, so mask them 0

        for i in range(len(sorted_lens)):
            if sorted_lens[i] < word_vec[i].size(0):
                word_vec[i, sorted_lens[i]:, :] = 0
        return word_vec

    def init_rnn_xavier_normal(self):
        for name, weights in self.rnn.named_parameters():
            weights = weights.view(-1, 1)
            torch.nn.init.xavier_normal_(weights)
            weights.squeeze()

    def predict_softmax_score(self,batch_input, sorted_lens, encoder_output, rnn_hidden, attention):
        assert not self.training
        decoded, hidden = self.forward(batch_input, sorted_lens, encoder_output, rnn_hidden, attention)
        target_score = [F.softmax(x, dim=1) for x in decoded]
        target_score = torch.cat(target_score)  # batch*seq_len, vocab_size
        return target_score, hidden

class PoetryRNN(nn.Module):
    def __init__(self, encoder_embed, encoder_v_len,  encoder_k_len,
                 decoder_embed, rnn_hidden_dim, rnn_layers, rnn_bidre, rnn_dropout,
                 dense_dim, dense_h_dropout,
                 freeze_embed=True):
        super(PoetryRNN, self).__init__()
    # Encoders
        self.encoder_v = VisualEncoder(encoder_embed.size(1))

        # self.encoder = PoetryEncoder(encoder_embed,
        #                              rnn_hidden_dim, rnn_layers,
        #                              freeze_embed)

        self.encoder_k = PoetryEncoder(encoder_embed,
                                     rnn_hidden_dim, rnn_layers,
                                     freeze_embed)
    # Attentions
        # 이미지에 대한 attention
        self.attention_v = VisualAttention(encoder_v_len, rnn_hidden_dim, encoder_embed.size(1))
        # # 직전 행에 대한 attention
        # self.attention = KeywordAttention(encoder_len, rnn_hidden_dim, encoder_embed.size(1))
        # 이미지 키워드에 대한 attention
        self.attention_k = KeywordAttention(encoder_k_len, rnn_hidden_dim, encoder_embed.size(1))

    # Decoder
        self.decoder = PoetryDecoder(decoder_embed,
                                     rnn_hidden_dim, rnn_layers, rnn_bidre, rnn_dropout,
                                     dense_dim, dense_h_dropout,
                                     freeze_embed)

    def forward(self, batch_img,
                batch_k, k_lens, k_lens_sort,
                # batch_pre, pre_lens, pre_lens_sort,
                batch_input, sorted_lens, inverted_ks_idx):
        # batch_img : 이미지
        # batch_k : 이미지에서 추출한 keyword 정보
        # k_lens : 이미지에서 추출한 keyword의 길이(토큰개수)
        # k_lens_sort :  
        # batch_input : 현재 행의 정보(decoder input)
        # sorted_lens : 내림차순으로 정렬된 seq length
        # inverted_ks_idx : decoder 내림차순으로 encoder의 순서를 정렬하는 idx
        
    # Encoder
        # 바뀐 순서를 여기서 처리???
        encoded_output_v = self.encoder_v.forward(batch_img)
        # encoded_output, encoded_hidden = self.encoder.forward(batch_pre, pre_lens)
        # encoded_output = encoded_output.index_select(dim=0, index=sorted_idx)

        # keyword encoder는 hidden state를 다음 행의 초기 hidden state로 사용하여 keyword 맥락이 계속 유지되게 함
        encoded_output_k, encoded_hidden  = self.encoder_k.forward(batch_k, k_lens)
        encoded_output_k = encoded_output_k.index_select(dim=0, index=inverted_ks_idx)
        encoded_hidden = tuple(i.index_select(dim=1, index=inverted_ks_idx) for i in encoded_hidden)

        # Decoder
        target_score, hidden = self.decoder.forward(batch_input, sorted_lens,
                                                    encoded_output_v,
                                                    encoded_hidden,
                                                    encoded_output_k,
                                                    self.attention_v,
                                                   
                                                    self.attention_k)

        return target_score, hidden

    def predict(self, batch_img,
                batch_k, k_lens, k_lens_sort,
                # batch_pre, pre_lens, pre_lens_sort,
                batch_input, sorted_lens, inverted_ks_idx):
        assert not self.training
        decoded, hidden = self.forward(batch_img,
                                        batch_k, k_lens, k_lens_sort,
                                        # batch_pre, pre_lens, pre_lens_sort,
                                        batch_input, sorted_lens, inverted_ks_idx)
        return decoded, hidden

    def predict_softmax_score(self, batch_img,
                batch_k, k_lens, k_lens_sort,
                # batch_pre, pre_lens, pre_lens_sort,
                batch_input, sorted_lens, inverted_ks_idx):
        assert not self.training
        decoded, hidden = self.forward(batch_img,
                                        batch_k, k_lens, k_lens_sort,
                                        # batch_pre, pre_lens, pre_lens_sort,
                                        batch_input, sorted_lens, inverted_ks_idx)
        target_score = [F.softmax(x, dim=1) for x in decoded]
        target_score = torch.cat(target_score)
        return target_score, hidden

    def evaluate_perplexity(self, batch_img,
                            batch_k, k_lens, k_lens_sort,
                            # batch_pre, pre_lens, pre_lens_sort,
                            batch_input, sorted_lens, inverted_ks_idx,
                            batch_target, loss_fn):
        # Enter evaluation model s.t. drop/batch norm are off
        assert not self.training
        target_score, _ = self.forward(batch_img,
                                        batch_k, k_lens, k_lens_sort,
                                        # batch_pre, pre_lens, pre_lens_sort,
                                        batch_input, sorted_lens, inverted_ks_idx)
        ppl = []
        start = 0
        for length in sorted_lens.tolist():
            score = target_score[start:start + length]
            target = batch_target[start:start + length]
            loss = loss_fn(score, target)
            ppl.append(2 ** loss.item())
            start += length
        return ppl
