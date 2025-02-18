import numpy as np
import torch
import torch.nn as nn



class S_CPC(nn.Module):
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
        )  
   
      

        # GRU for temporal encoding

        self.gru = nn.GRU(64, 32, num_layers=1, bidirectional=False, batch_first=True) 

        self.W = nn.ModuleList([nn.Linear(32, 64) for _ in range(self.timestep)])  

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

         # Initialize weights
        self.apply(_weights_init)



    def init_hidden(self, batch_size, use_gpu=True):
        if use_gpu:
            return torch.zeros(1, batch_size, 32).cuda()
        else:
            return torch.zeros(1, batch_size, 32)

    def forward(self, x, hidden):
        batch_size = x.size(0)
        
        # spectral data
        x_fft = torch.fft.fft(x, dim=2)  # Apply FAST FOURIER TRANSFORM along the time dimension
        x_fft = torch.abs(x_fft)  # Use the magnitude of the FFT result (ignoring phase)
      

        z = self.encoder(x_fft)  # Apply the encoder


        # Transpose z for GRU input
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
    
# accuracy is based on correctly matching positive samples with their corresponding augmented pair in the batch


# The outputs from both temporal and spectral encoders can then be fused (e.g., via concatenation or bilinear transformation) before the NCE loss is computed.

    def predict(self, x, hidden):
       
        x_fft = torch.fft.fft(x, dim=2)  # applied along the time dimension
        x_fft = torch.abs(x_fft)         # magnitude 

       

        z = self.encoder(x_fft)
        z = z.transpose(1, 2)

        output, hidden = self.gru(z, hidden)
        
        return output, hidden