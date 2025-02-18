
import numpy as np
import torch
import torch.nn as nn

class T_CPC(nn.Module):
    def __init__(self, timestep, batch_size, seq_len):

        
        super().__init__()
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.timestep = timestep
        
        self.encoder = nn.Sequential(
            nn.Conv1d(16, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),  # Augmentation
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )  # [bs, 64, seq_len]
        
        self.gru = nn.GRU(64, 32, num_layers=1, bidirectional=False,
                          batch_first=True)  # last layer=tanh, c=[-1,1]
        self.W = nn.ModuleList([nn.Linear(32, 64) for _ in range(timestep)])
        self.log_softmax = nn.LogSoftmax(dim=1)
       
        def _weights_init(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # initialize gru
        for layer_p in self.gru._all_weights:
            for p in layer_p:
                if 'weight' in p:
                    nn.init.kaiming_normal_(self.gru.__getattr__(p), mode='fan_out', nonlinearity='relu')

        self.apply(_weights_init)
    
    def init_hidden(self, batch_size, use_gpu=True):
        if use_gpu:
            return torch.zeros(1, batch_size, 32).cuda()
        else:
            return torch.zeros(1, batch_size, 32)

    def forward(self, x, hidden):
        # input sequence is [N, C, L], e.g. size [bs, 16, 200]
        batch_size = x.size(0)
      
        # encode sequence x
        z = self.encoder(x)  # encoded sequence is [N, C, L], e.g. size [bs, 64, 200]
        
        # sample t between [0.4*seq_len, seq_len-timestep]
        t = torch.randint(int(0.4 * z.size(2)), z.size(2) - self.timestep, size=(1,)).long()
        
        # calculate c_t: take all z_<=t and use them as input for the GRU
        z = z.transpose(1, 2)  # reshape to [N, L, C] for GRU, e.g. size [bs, 200, 64]
        forward_seq = z[:, :t+1, :]  # e.g. size [bs, t, 64]
        output, hidden = self.gru(forward_seq, hidden)  # output, e.g. size [bs, t, 32]
        c_t = output[:, t, :].view(batch_size, 32)  # c_t, e.g. size [bs, 32]
       
        nce = 0
        for k in range(self.timestep):
            linear = self.W[k]
            z_tk = z[:, t+k+1].view(batch_size, 64)  # z_t+k, e.g. size [bs, 64]
            scores = linear(c_t) @ z_tk.T  # bilinear score: z_t+k * Wk * c_t, e.g. size [bs, bs]
            nce += self.log_softmax(scores).diag().sum() # nce is a tensor

        nce /= -1. * batch_size * self.timestep  # average over timestep and batch
        
        y = torch.arange(batch_size).to(x.device)
        accuracy = scores.argmax(1).eq_(y).float().mean()
        accuracy = accuracy * 100

        return nce, accuracy, hidden, c_t

    def predict(self, x, hidden):
        # input sequence is [N, C, L], e.g. size [bs, 16, 200]
        z = self.encoder(x)
        # encoded sequence is [N, C, L], e.g. size [bs, 64, 200]
        # reshape to [N, L, C] for GRU, e.g. size [bs, 200, 64]
        z = z.transpose(1, 2)
        output, hidden = self.gru(z, hidden)  # output, e.g. size [bs, 200, 32]

        return output, hidden
        # return output[:,-1,:], hidden # only return the last
