import torch
import torch.nn as nn

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, vocab_size=248, embed_size=100, h_size=100, n_layers=1, dropout=0.5):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, h_size, n_layers, dropout=dropout)
        self.linear = nn.Linear(h_size, vocab_size)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
#         if tie_weights:
#             if nhid != ninp:
#                 raise ValueError('When using the tied flag, nhid must be equal to emsize')
#             self.decoder.weight = self.encoder.weight

        self.init_weights()
        self.n_layers = n_layers
        self.h_size = h_size

    def init_weights(self):
        initrange = 0.1
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, captions, hidden):
        "hidden = (hidden, cell)"
        emb = self.embed(captions)
        emb = self.drop(emb)
        output, hidden = self.lstm(emb, hidden)
        output = self.drop(output)
        decoded = self.linear(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, batch_size):
        # 아래를 n_layer 개수만큼...
        weight = next(self.parameters())
        output = (weight.new_zeros(self.n_layers, batch_size, self.h_size),
                weight.new_zeros(self.n_layers, batch_size, self.h_size))
#         print("init_hidden", output[0].shape, output[1].shape)
        return output