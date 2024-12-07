from CPC import CPC


import wandb
import torch
import torch.nn.functional as F
import torch.optim as optim

import torch.nn as nn
import time
import matplotlib.pyplot as plt



import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5,6"

from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Dataset

import pickle


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

        

            acc_onehot = F.one_hot(acc, 16)
            data = data.unsqueeze(2).float() * acc_onehot.float()
            data = data.permute(0, 2, 1).to(device)  
           

            hidden = model.init_hidden(len(data), True)   
            # accuracy, test_loss, hidden, _ = 
            test_loss, test_accuracy, hidden, c_t = model(data, hidden.float())


            test_loss_step.append(test_loss.item())
            test_acc_step.append(test_accuracy.item())

        

      
   
    test_loss_epoch = torch.tensor(test_loss_step).mean()
    test_acc_epoch = torch.tensor(test_acc_step).mean()

    wandb.log({'test loss': test_loss_epoch, 'test accuracy': test_acc_epoch})
   
    return test_loss_epoch, test_acc_epoch






def PCAgraph(model, loader, log_graph=False):
        
    model.eval()

    with torch.no_grad():

            data, acc, label, patient_id = next(iter(loader))

        # *********************************** PCA ***********************************
          
            acc_onehot = F.one_hot(acc, 16)
            data = data.unsqueeze(2).float() * acc_onehot.float()
            data = data.permute(0, 2, 1)
            
         
            # idx = (patient_id == 145) + (patient_id == 106) + (patient_id == 116) + (patient_id == 176)
            
            idx = patient_id >100
           


            data_idx = data[idx]
            data_idx = data_idx.float().cuda()
            label_idx = label[idx]

            
            

            hidden = model.init_hidden(len(data_idx), True)
            output, _ = model.predict(data_idx, hidden)
            c_t = output[:, -1].squeeze_()
         
           
            if log_graph:
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
                wandb.log({"CPC train PCA graph": wandb.Image(plt)})
                plt.close()


def train(num_epochs, model, optimizer, train_loader, test_loader, device, use_cuda, test_every=1):
    start = time.time()
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
        train_loss = train_one_epoch(model, train_loader, device, use_cuda, optimizer)
       
        
        # Validate on train and test set
        if epoch % test_every == 0 or epoch == num_epochs - 1:
            
            
            train_loss, train_acc = train_one_epoch(model, train_loader, device, use_cuda, optimizer)
            
            if epoch == num_epochs - 1:
                PCAgraph(model, test_loader,log_graph=True)
            else:
                PCAgraph(model, test_loader,log_graph=False)
            
# after completing an epoch, evaluate the model on the same training data but with the model in model.eval() mode. 
# This does not update the model's parameters but helps you monitor if the model is overfitting (i.e., performing too well on the training data compared to the test data).

       
            test_loss, test_acc = validate(model, test_loader, device, use_cuda)
            
            
            print(f'Ep {epoch+1:2d}/{num_epochs:2d}: Train Acc:{train_acc:5.2f}% | Test Acc: {test_acc:5.2f}% | Train Loss: {train_loss:4.2f} | Test Loss: {test_loss:4.2f}')
            
           

            # Track stats
            train_loss_hist.append(train_loss)
            train_acc_hist.append(train_acc)
            test_loss_hist.append(test_loss)
            test_acc_hist.append(test_acc)
            epoch_hist.append(epoch)
            
    

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
  
    plt.show()

    wandb.log({'CPC graph': wandb.Image(plt)})

    plt.close()


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
    

def main():

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    path = '/home/ria/MMBSintern/bachelor/exercise/data' 
    train_data = Wearable(path, True)
    test_data = Wearable(path, False)



    traindl_bs = 64 
    train_loader = DataLoader(train_data, batch_size=traindl_bs, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=len(test_data), shuffle=False, num_workers=0) 



    num_epochs = 10
    batch_size = [64,128,256,512,1024]

    dataset = 'train_balancedwork.pickle, test.pickle'
    testdl_bs = 'len(test_data)'   #### need to be kept


   
    for bs in batch_size:

        test_every = num_epochs/10

        wandb.finish() 
        
        ######
        model = CPC(timestep=3, batch_size=bs, seq_len=200).to(device)
        optimizer = optim.AdamW(model.parameters())
        ######

        config = dict(num_epochs = num_epochs,
                        batch_size = batch_size,
                        #dataset = dataset,
                        test_every=test_every,
                        stride = 1)

            
        # Pass the config dictionary when you initialize W&B
        run = wandb.init(project="CPC train renew", config = config)

        stats = train(config['num_epochs'], model, optimizer, train_loader, test_loader, device, use_cuda, config['test_every'])

        torch.save(model.encoder.state_dict(), 'CPC_encoder_weights.pth')  # Save encoder weights to enable 


        plot_stats(*stats)


if __name__ == "__main__":
    main()