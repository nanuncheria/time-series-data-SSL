
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
    




#######__ADDED__#################################################

import wandb
import torch.nn.functional as F
import torch.optim as optim
import torch.fft
import time
from tqdm import tqdm


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5,6"



def train_one_epoch(model, loader, device, use_cuda, optimizer):
    
    wandb.watch(model, log_freq=10)

    model.train()
    # loader = tqdm(loader)

    loss_step = []
    acc_step = []
   

    for i, (data, acc, label, _) in enumerate(loader):


        label = label.to(device)
        acc_onehot = F.one_hot(acc, 16)

        # unsqueeze(2) adds an extra dimension to data at position 2 --> element wise multiplication
        data = data.unsqueeze(2).float() * acc_onehot.float()
        data = data.permute(0, 2, 1).to(device)  # add channel dimension
        # permute changes the order of dimensions of 'data'

        hidden = model.init_hidden(len(data), True)   # torch.zeros(1, batch_size=len(data), self.h2) 
        # accuracy, loss, hidden, _ = 
        loss, accuracy, hidden, c_t = model(data, hidden.float())

        loss_step.append(loss.item())
        acc_step.append(accuracy.item())


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        

    loss_epoch = torch.tensor(loss_step).mean()
    acc_epoch = torch.tensor(acc_step).mean()   

    wandb.log({ 'train loss': loss_epoch, 'train accuracy': acc_epoch})

    
    return loss_epoch, acc_epoch


def validate(model, loader, device, use_cuda):

    wandb.watch(model, log_freq=10)

    model.eval()
    
    test_loss_step = []
    test_acc_step = []
    
    with torch.no_grad():

        for i, (data, acc, label, patient_id) in enumerate(loader):
       

            label = label.to(device)

            # acc_onehot = F.one_hot(acc.view(1, -1), 16).view(acc.size(0), acc.size(1), 16)
            acc_onehot = F.one_hot(acc, 16)
            data = data.unsqueeze(2).float() * acc_onehot.float()
            data = data.permute(0, 2, 1).to(device)  # add channel dimension
           

            hidden = model.init_hidden(len(data), True)   # torch.zeros(1, batch_size=len(data), self.h2) 
            # accuracy, test_loss, hidden, _ = 
            test_loss, test_accuracy, hidden, c_t = model(data, hidden.float())


            test_loss_step.append(test_loss.item())
            test_acc_step.append(test_accuracy.item())

            


      
   
    test_loss_epoch = torch.tensor(test_loss_step).mean()
    test_acc_epoch = torch.tensor(test_acc_step).mean()

    wandb.log({'test loss': test_loss_epoch, 'test accuracy': test_acc_epoch})
   
    return test_loss_epoch, test_acc_epoch



import matplotlib.pyplot as plt
from sklearn.decomposition import PCA



def PCAgraph(model, loader, idx):
        
    model.eval()

    with torch.no_grad():

            data, acc, label, patient_id = next(iter(loader))

        # *********************************** PCA ***********************************
          
            acc_onehot = F.one_hot(acc, 16)
            data = data.unsqueeze(2).float() * acc_onehot.float()
            data = data.permute(0, 2, 1)
            
           
            if idx == 'same':
                idx = (patient_id == 145) + (patient_id == 106) + (patient_id == 116) + (patient_id == 176)
            elif idx == 'id > 100':
                idx = patient_id >100
            else:
                print('Error: idx undefinable')
            

            data_idx = data[idx]
            data_idx = data_idx.float().cuda()
            label_idx = label[idx]

            
            

            hidden = model.init_hidden(len(data_idx), True)
            output, _ = model.predict(data_idx, hidden)
            c_t = output[:, -1].squeeze_()
         
            #The label_idx variable is directly taken from label without any filtering, which means it contains labels for all samples

            # pca = PCA(n_components=3)
            pca = PCA(n_components=2)
            principal_components = pca.fit_transform(c_t.cpu()) #.detach().numpy())
            label_idx = label_idx.cpu().numpy()
            plt.figure(figsize=(5, 5))
            plt.rc('axes', labelsize=18) 
            plt.rc('xtick', labelsize=15) 
            plt.rc('ytick', labelsize=15)
            plt.scatter(principal_components[label_idx == 0, 0], principal_components[label_idx == 0, 1],  # selects first pc for sample with label 0 , then selects the second pc for ''
                        s=40, color='#2283a2')
            plt.scatter(principal_components[label_idx == 1, 0], principal_components[label_idx == 1, 1],
                        s=40, color='#f66209')
            plt.xlabel('First Principle Component')
            plt.ylabel('Second Principle Component')
            plt.legend(['non-Afib', 'Afib'], prop={'size': 15})

            plt.show()


def train(num_epochs, model, optimizer, train_loader, test_loader, device, use_cuda, test_every=1):
    start = time.time()
    ### START CODE HERE ### (≈ 19 lines of code)
    train_acc_hist = []
    test_acc_hist = []
    train_loss_hist = []
    test_loss_hist = []
    epoch_hist = []
    best_test_loss = 1000
    best_test_acc = 0

    
    train_loss, train_acc = validate(model, train_loader, device, use_cuda)
    test_loss, test_acc = validate(model, test_loader, device, use_cuda)
    print(f'Init: Train Acc:{train_acc:5.2f}% | Test Acc: {test_acc:5.2f}% | Train Loss: {train_loss:4.2f} | Test Loss: {test_loss:4.2f}')

    
    for epoch in range(num_epochs):
        
        # Train 1 epoch
        train_loss, train_acc = train_one_epoch(model, train_loader, device, use_cuda, optimizer)
       
        
        # Validate on train and Test set
        if epoch % test_every == 0 or epoch == num_epochs - 1:
            
            # again as discussed on 30.10, there is 
            #train_loss, train_acc = train_one_epoch(model, train_loader, device, use_cuda, optimizer)
            PCAgraph(model, test_loader, config['idx'])
            
# Sometimes, after completing an epoch, you might want to evaluate the model on the same training data but with the model in model.eval() mode. 
# This does not update the model's parameters but helps you monitor if the model is overfitting (i.e., performing too well on the training data compared to the test data).


            #train_loss = train_one_epoch(model, train_loader, device, use_cuda, optimizer)
            #_, train_acc = validate(model, train_loader, device, use_cuda)
       
            test_loss, test_acc = validate(model, test_loader, device, use_cuda)
            
            # Print epoch results to screen 
            print(f'Ep {epoch+1:2d}/{num_epochs:2d}: Train Acc:{train_acc:5.2f}% | Test Acc: {test_acc:5.2f}% | Train Loss: {train_loss:4.2f} | Test Loss: {test_loss:4.2f}')
            
           

            # Track stats
            train_loss_hist.append(train_loss)
            train_acc_hist.append(train_acc)
            test_loss_hist.append(test_loss)
            test_acc_hist.append(test_acc)
            epoch_hist.append(epoch)
            
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': test_loss,
                    }, f'best_model_min_test_loss.pth')
            
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': test_loss,
                    }, f'best_model_max_test_acc.pth')

    print(f"Total training time was {(time.time() - start) / 60:.1f}m.")

   
    return train_loss_hist, test_loss_hist, train_acc_hist, test_acc_hist, epoch_hist


def plot_stats(train_loss_hist, test_loss_hist, train_acc_hist, test_acc_hist, epoch_hist):
   
    plt.figure(figsize=(16,5))

    plt.subplot(1,2,1)
    plt.title("Accuracy")
    plt.plot(epoch_hist, train_acc_hist, label='Train acc.')
    plt.plot(epoch_hist, test_acc_hist, label='Test acc.')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.axhline(y=95, color='red', label="Target acc.")
    plt.ylim(top=100)
    plt.legend()

    plt.subplot(1,2,2)
    plt.title("Loss")
    plt.plot(epoch_hist, train_loss_hist, label='Train loss')
    plt.plot(epoch_hist, test_loss_hist, label='Test loss')
    plt.ylim(bottom=0)
    plt.ylabel("Loss")
    plt.xlabel('Epochs')
    plt.legend()

    wandb.log({'TS graph': wandb.Image(plt)})
  
    plt.show()
    plt.close()



#########



# capture a dictionary of hyperparameters with config

epochs = [20,50,100]
batch_size = 64
dataset = 'train_balancedwork.pickle, test.pickle'
traindl_bs = 64
testdl_bs = 'len(test_data)'   #### need to be kept





from torch.utils.data import DataLoader, Dataset
import pickle




class Wearable(Dataset):
    def __init__(self, path, train=True):
        if train:
            with open(os.path.join(path, 'train_balancedwork.pickle'), 'rb') as file:
                self.examples = pickle.load(file)
        else:
            with open(os.path.join(path, 'test.pickle'), 'rb') as file:
                self.examples = pickle.load(file)

    def __getitem__(self, item):
        return self.examples[item]

    def __len__(self):
        return len(self.examples)


path = '/home/ria/MMBSintern/bachelor/exercise/data'
train_data = Wearable(path, True)
test_data = Wearable(path, False)



train_loader = DataLoader(train_data, batch_size=traindl_bs, shuffle=True, num_workers=0)
test_loader = DataLoader(test_data, batch_size=len(test_data), shuffle=False, num_workers=0) 



#from utils import ScheduledOptim

use_cuda = torch.cuda.is_available()
#print('use_cuda is', use_cuda)
device = torch.device("cuda" if use_cuda else "cpu")


#from TS import TS



    

#model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#print(f'# Model summary #\n {str(model)}\n')
#print(f'===>  total number of parameters: {model_params}\n')

index = ['same','id > 100']

for idx in index:
    for num_epochs in epochs:

        test_every = num_epochs/10


        model = TS(timestep=3, batch_size=batch_size, seq_len=200).to(device)
    
    # optimizer = ScheduledOptim(
        #optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
        #           betas=(0.9, 0.98), eps=1e-09, weight_decay=1e-4, amsgrad=True),
        #n_warmup_steps=50)


  # The n_warmup_steps parameter suggests that it uses some form of learning rate warmup strategy.
  # The learning rate should be managed internally by ScheduledOptim

        optimizer = optim.AdamW(model.parameters())



        config = dict(num_epochs = num_epochs,
                    batch_size = batch_size,
                    #dataset = dataset,
                    test_every=test_every,
                    idx = idx,
                    stride = 1)

        
        # Pass the config dictionary when you initialize W&B
        run = wandb.init(project="Exercise TS final", config = config)

        stats = train(config['num_epochs'], model, optimizer, train_loader, test_loader, device, use_cuda, config['test_every'])


        plot_stats(*stats)

        wandb.finish()






