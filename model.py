import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class NIP(nn.Module):
    def __init__(self, args):
        super(NIP, self).__init__()
        self.batch_size = args.batch_size
        self.window_len = args.window_len
        self.kernel_size = args.kernel_size
        self.lstm_hidden_size = args.lstm_hidden_size
        self.gru_hiden_size = args.gru_hidden_size
        self.mlp_hidden = args.mlp_hidden

        Ci = 1
        Co = args.kernel_num
        Ks = args.kernel_size
        self.convs = nn.ModuleList([nn.Conv2d(Ci, Co, kernel_size=(K, args.news_dim)) for K in Ks])
        self.dropout = nn.Dropout(args.dropout)
        self.fc1 = nn.Linear(len(Ks) * Co, args.cnn_out_size)

        self.lstm = nn.LSTM(args.cnn_out_size, args.lstm_hidden_size, batch_first=True)

        # MLP
        self.mlp_hidden.insert(0, self.lstm_hidden_size)
        self.mlp = nn.Sequential()
        for i in range(len(self.mlp_hidden)-1):
            self.mlp.add_module('mlp'+str(i), nn.Linear(self.mlp_hidden[i], self.mlp_hidden[i+1]))
            self.mlp.add_module('activation'+str(i), nn.Sigmoid())
        self.mlp.add_module('mlp'+str(i), nn.Linear(self.mlp_hidden[i], args.num_classes))
        self.mlp.add_module('softmax', nn.Softmax())


    def forward(self, x, p):
        # x: (batch_size, seq_len, feature_dim)
        x = x.unsqueeze(1)  # (batch_size, 1, seq_len, feature_dim)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # [(batch_size, Co, seq_len)]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(batch_size, Co)]*len(Ks)

        x = torch.cat(x, 1)  # (batch_size, len(Ks) * Co)
        x = self.fc1(self.dropout(x))  # (batch_size, cnn_out)

        x = x.unsqueeze(1)  # (batch_size, 1, cnn_out)
        x, _ = self.lstm(x) # (batch_size, lstm_out)

        output = self.mlp(x.squeeze(1))    # (batch_size, num_classes)

        return output