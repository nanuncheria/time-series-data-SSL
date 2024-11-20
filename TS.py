
import numpy as np
import torch
import torch.nn as nn



class TS(nn.Module):
    def __init__(self, timestep, batch_size, seq_len):
        super().__init__()
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.timestep = timestep

        # Temporal Encoder (similar to the original CPC encoder)
        self.temporal_encoder = nn.Sequential(
            nn.Conv1d(16, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )  # [bs, 64, seq_len]
   
        # Spectral Encoder 
        self.spectral_encoder = nn.Sequential(
            nn.Conv1d(16, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )  # [bs, 64, seq_len]

        # GRU for temporal encoding

        ## GRU needs to process the concatenation of temporal and spectral encodings(combined features of 128 channels)
        self.gru = nn.GRU(128, 32, num_layers=1, bidirectional=False, batch_first=True) 

        self.W = nn.ModuleList([nn.Linear(32, 128) for _ in range(self.timestep)])  # 32 is the GRU hidden size, 128 is the combined (temporal + spectral) feature size


        # Bilinear fusion of temporal and spectral features
        self.bilinear_fusion = nn.Linear(64 * 2, 128) 

        # Log-Softmax for contrastive loss
        self.log_softmax = nn.LogSoftmax(dim=1)

        # Initialize weights
        self.apply(self._weights_init)

    def _weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def init_hidden(self, batch_size, use_gpu=True):
        if use_gpu:
            return torch.zeros(1, batch_size, 32).cuda()
        else:
            return torch.zeros(1, batch_size, 32)

    def forward(self, x, hidden):
        batch_size = x.size(0)
        
        # Temporal encoding for input sequence x
        z_temporal = self.temporal_encoder(x)  # [N, 64, seq_len]

        # Spectral encoding (apply FFT and then spectral encoder)

        x_fft = torch.fft.fft(x, dim=2)  # Apply FAST FOURIER TRANSFORM along the time dimension
        x_fft = torch.abs(x_fft)  # Use the magnitude of the FFT result (ignoring phase)
        z_spectral = self.spectral_encoder(x_fft)  # Apply the spectral encoder


        # Concatenate temporal and spectral encodings : fused version!!
        z = torch.cat((z_temporal, z_spectral), dim=1)  # [N, 128, seq_len]

        # Transpose z for GRU input
        z = z.transpose(1, 2)  # [bs, seq_len, 128]

        # Sample t between [0.4*seq_len, seq_len - timestep]
        t = torch.randint(int(0.4 * z.size(2)), z.size(2) - self.timestep, size=(1,)).long()

        


        # Pass z up to time t through GRU to get c_t (context vector)
        forward_seq = z[:, :t+1, :]  # [N, t, 128]
        output, hidden = self.gru(forward_seq, hidden)  # [N, t, 32]
        c_t = output[:, t, :].view(batch_size, 32)  # [N, 32]

        # Calculate contrastive loss (NCE)
        nce = 0
        for k in range(self.timestep):
            linear = self.W[k]
            z_tk = z[:, t+k+1].view(batch_size, 128)  # [N, 128] (temporal + spectral features)
            scores = linear(c_t) @ z_tk.T  # Bilinear score: z_t+k * Wk * c_t, [N, N]
            nce += self.log_softmax(scores).diag().sum()  # Sum diagonal (positive samples)

        nce /= -1. * batch_size * self.timestep  # Average over timestep and batch

        # Calculate accuracy
        y_true = torch.arange(batch_size).to(x.device)
        accuracy = scores.argmax(1).eq_(y_true).float().mean() * 100  # Accuracy in percentage

        #predictions = scores.argmax(dim=1)  # Get predicted class indices
        #correct_predictions = (predictions == y_true).float()  # Boolean array of correct predictions

        # Calculate accuracy
        #accuracy = correct_predictions.sum() / batch_size  # Correct predictions divided by total
        #accuracy *= 100


        return nce, accuracy, hidden, c_t
    
# accuracy is based on correctly matching positive samples with their corresponding augmented pair in the batch


# The outputs from both temporal and spectral encoders can then be fused (e.g., via concatenation or bilinear transformation) before the NCE loss is computed.

    def predict(self, x, hidden):
        # Temporal encoding
        z_t = self.temporal_encoder(x)
        
        # Spectral encoding
        x_fft = torch.fft.fft(x, dim=2)
        x_fft = torch.abs(x_fft)
        z_s = self.spectral_encoder(x_fft)

        # Bilinear fusion
        fusion = torch.cat([z_t, z_s], dim=1)
        fusion_t = fusion.transpose(1, 2)
        # print('concatenated feature size:', fusion.size())
    # ensure that it is (batch_size, seq_len, 128).



        fused_rep = self.bilinear_fusion(fusion_t)
        
        #fused_rep = fused_rep.transpose(1, 2)
        output, hidden = self.gru(fused_rep, hidden)
        
        return output, hidden