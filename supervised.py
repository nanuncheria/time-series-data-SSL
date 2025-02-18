import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import time

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4,6"

import wandb

from T_CPC import T_CPC
from TS_CPC import TS_CPC
from S_CPC import S_CPC
from linearclassifier_all import Wearable

import numpy as np
import itertools

import csv
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, roc_auc_score 


#criterion = nn.BCELoss()
#optimizer = optim.AdamW()


def sup_train(model, loader, optimizer, criterion, device):
    
    model.train()


    all_c_t = []  # To store all the hidden states (features)
    all_labels = []  

    total_loss = 0
    batch_count = 0

    for i, (data, acc, label, _) in enumerate(loader):
        
        data = data.to(device)
        label = label.float()
        label = label.to(device)

        acc_onehot = F.one_hot(acc, 16).to(device)
        data = data.unsqueeze(2).float() * acc_onehot.float()
        data = data.permute(0, 2, 1).to(device)

        hidden = model.init_hidden(len(data), True).to(device)

        output, _ = model.predict(data, hidden)

        # output shape = [batch_size, sequence_length, feature_dim]
        # output[:,-1] selects the last time step across all the sequences, resulting in a shape of [batch_size, feature_dim].

        ### Condensed representation of the input sequence. Extracted by the CPC model, information that correlates with labels to enable classification
        c_t = output[:, -1].mean(dim=1)  # (c_t): torch.Size([256, 32]) --> torch.Size([256])
        c_t = c_t.unsqueeze(1)  # Shape: [batch_size, 1]

        
        # (label): torch.Size([256]) – single value (binary) for each sample in the batch
        label = label.unsqueeze(1) # shape [batch_size, 1]


        ### Supervised Learning 
        loss = criterion(c_t, label)
    
        total_loss += loss.item() 
        batch_count +=1

        all_c_t.append(c_t.detach())
        all_labels.append(label.detach())

        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        

    average_loss = total_loss/batch_count

    X_train = torch.cat(all_c_t, dim=0).cpu().numpy()  # Features for the entire dataset
    Y_train = torch.cat(all_labels, dim=0).cpu().numpy()  # Labels for the entire dataset


    return X_train, Y_train, average_loss

   

def sup_test(model, loader, criterion, device):

    model.eval()

    all_c_t = []  
    all_labels = []  


    total_loss=0
    batch_count=0

    with torch.no_grad():  # we dont want the gradient: saves computation and protects from any errors
        
        for i, (data, acc, label, _) in enumerate(loader):
        
            data = data.to(device)
            label = label.float()
            label = label.to(device)

            acc_onehot = F.one_hot(acc, 16).to(device)
            data = data.unsqueeze(2).float() * acc_onehot.float()
            data = data.permute(0, 2, 1).to(device)

            hidden = model.init_hidden(len(data), True).to(device)

            output, _ = model.predict(data, hidden)
            c_t = output[:, -1].mean(dim=1) 
            c_t = c_t.unsqueeze(1) 

            label = label.unsqueeze(1) 


            loss = criterion(c_t, label)
            total_loss += loss.item() 
            batch_count +=1

            all_c_t.append(c_t.detach())
            all_labels.append(label.detach())


    average_loss = total_loss / batch_count
      
    X_test = torch.cat(all_c_t, dim=0).cpu().numpy() 
    Y_test = torch.cat(all_labels, dim=0).cpu().numpy()  


    return X_test, Y_test, average_loss
          


def linearclassifier(X_train, Y_train, X_test, Y_test, log_graph=False):


    Y_train = Y_train.flatten()
    Y_test = Y_test.flatten()

    # Create logistic regression object (for binary classification)
    classifier = LogisticRegression()

    classifier.fit(X_train,Y_train)

    Y_pred = classifier.predict(X_test) 
    probs = classifier.predict_proba(X_test)  

    # Accuracy and classification report
    accuracy = accuracy_score(Y_test, Y_pred)
    report = classification_report(Y_test, Y_pred, output_dict=True)  # Use output_dict=True to get a dictionary

    
    #print("Classification Report:")
    #print(report)
    #print("Unique values in test_label:", np.unique(Y_test))
    #print("Unique values in y_pred:", np.unique(Y_pred))

    
    wandb.log({
        "precision_non-AFib": report['0.0']['precision'],
        "recall_non-AFib": report['0.0']['recall'],
        "f1_non-AFib": report['0.0']['f1-score'],
        "precision_AFib": report['1.0']['precision'],
        "recall_AFib": report['1.0']['recall'],
        "f1_AFib": report['1.0']['f1-score'],
        "accuracy": report['accuracy']
    })



    
    roc_auc = roc_auc_score(Y_test, probs[:, 1])  # Use probabilities of the positive class


    wandb.log({f"ROC-AUC Score": roc_auc})



        
    if log_graph:
        plt.figure(figsize=(5,5))

        # Convert X_test to CPU if on gpu
        if isinstance(X_test, torch.Tensor) and X_test.is_cuda:
            X_test = X_test.cpu()
            
        # Scatter plot for the test data points

        plt.hist(X_test[Y_test == 0.0], bins=30, alpha=0.5, label="Class 0 (Test)", color='#2283a2')
        plt.hist(X_test[Y_test == 1.0], bins=30, alpha=0.5, label="Class 1 (Test)", color='#f05f70')
        plt.legend(['non-AF', 'AF'], prop={'size': 15})

        plt.title("Linear Classifier classification")
        plt.show()
        wandb.log({"classification": wandb.Image(plt)})
        plt.close()



    return accuracy, roc_auc



            

  
    

def train(num_epochs, model, train_loader, test_loader, optimizer, criterion, device, test_every=1):
    start = time.time()


    accuracy_list = []
    epoch_list = []
    train_loss_list = []
    test_loss_list = []


    history= []
    for epoch in range(num_epochs):

       
        _, _, train_loss = sup_train(model, train_loader, optimizer, criterion, device)
        train_loss_list.append(train_loss)
        
        # Validate on train and test set
        if epoch % test_every == 0 or epoch == num_epochs - 1:


           # is repetitive to have training done again
            train_c_t, train_label, _ = sup_test(model, train_loader, criterion, device)
            test_c_t, test_label, test_loss = sup_test(model, test_loader, criterion,device)
            test_loss_list.append(test_loss)

            # Reshape c_t to [batch_size, 1] and label to [batch_size] if necessary
            train_c_t = train_c_t.reshape(-1, 1)  # Ensure it's 2D (batch_size, 1)
            test_c_t = test_c_t.reshape(-1, 1)    # Ensure it's 2D (batch_size, 1)
            train_label = train_label.reshape(-1) # Ensure it's 1D (batch_size,)
            test_label = test_label.reshape(-1)   # Ensure it's 1D (batch_size,)

            # Pass the NumPy arrays to your linear classifier
            if epoch == num_epochs - 1:
                accuracy, roc_auc = linearclassifier(train_c_t, train_label, test_c_t, test_label, log_graph=True)
            else:
                accuracy, roc_auc = linearclassifier(train_c_t, train_label, test_c_t, test_label, log_graph=False)
                    
            accuracy_list.append(accuracy * 100)
            epoch_list.append(epoch)
            print (f'for Epoch {epoch}, Accuracy: {accuracy*100}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

            history.append((epoch, roc_auc))

            wandb.log({"epoch": epoch, "accuracy": accuracy, 'train_loss': train_loss, 'test_loss' :test_loss})


            
    plt.plot (epoch_list, accuracy_list)
    plt.plot(range(num_epochs), train_loss_list, label="Train Loss")
    plt.plot(epoch_list, test_loss_list, label="Test Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Model Accuracy over Epochs')



    plt.figure(figsize=(16,5))

    plt.subplot(1,2,1)
    plt.title("Accuracy")
    plt.plot(epoch_list, accuracy_list, label='acc(Y_test, Y_pred)')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.axhline(y=95, color='red', label="Target acc.")
    plt.ylim(top=100)
    plt.legend()

    plt.subplot(1,2,2)
    plt.title("Loss")
    plt.plot(range(num_epochs), train_loss_list, label="Train Loss")
    plt.plot(epoch_list, test_loss_list, label="Test Loss")
    plt.ylim(bottom=0)
    plt.ylabel("Loss")
    plt.xlabel('Epochs')
    plt.legend()

    wandb.log({'Linear Classificaton': wandb.Image(plt)})
  
    plt.show()
    plt.close()# Close the plot to free memory

    print(f"Total training time was {(time.time() - start) / 60:.1f}m.")


    return history



def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # batch_size = 64



    path = '/home/ria/MMBSintern/bachelor/exercise/data' 

    train_data = Wearable(path, True)
    test_data = Wearable(path, False)


   

    # Hyperparameters

    results= []

    modelname=['T_CPC', 'TS_CPC', 'S_CPC']
    n_epo=10

    #lrl = [1e-3,1e-7]
    #batch_sizes = [256,1024]
    lrl = [1e-3]
    batch_sizes = [1024]

    # Generate all combinations 
    combinations = list(itertools.product(modelname, batch_sizes, lrl))

    # Run each combination with each value of num_epochs

    for modelname, bs, lrl in combinations:

        train_loader = DataLoader(train_data, batch_size=bs, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_data, batch_size=len(test_data), shuffle=False, num_workers=0) 

        
        
        test_every = n_epo / 10
        lr = (bs/1024) * lrl
    
        print(f"Running with num of epochs: {n_epo}, model: {modelname}, learning rate: {lr}, batch size: {bs}")
        
    
        if modelname == 'T_CPC':
            model = T_CPC(timestep=3, batch_size=bs, seq_len=200).to(device)

        elif modelname == 'TS_CPC':
            model = TS_CPC(timestep=3, batch_size=bs, seq_len=200).to(device)

        elif modelname == 'S_CPC':
            model = S_CPC(timestep=3, batch_size=bs, seq_len=200).to(device)

        else:
            raise ValueError(f"Unknown model name: {modelname}")

        optimizer = optim.AdamW(model.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss()

    
        config = dict(num_epoch = n_epo,
                test_every = test_every,
                model = modelname,
                learning_r = lr,
                batch_size = bs)
            
        
        wandb.init(project="SUPERVISED_learning", 
            name=f"{n_epo, modelname, lr, bs}",
            config=config,
            reinit=True)


        history = train(config['num_epoch'], model, train_loader, test_loader, optimizer, criterion, device, config['test_every'])
        
        for epoch, roc_auc in history: 
            results.append([epoch, modelname, bs, roc_auc])
        

        wandb.finish() 


    df = pd.DataFrame(results, columns=['n_Epochs','Model','Batch Size','ROC_AUC'])
    csv_filename = 'SUPERVISED.csv'

    df.to_csv(csv_filename, index=False)
    print(f"Results saved to {csv_filename}")


    # Load CSV and plot results
    df = pd.read_csv(csv_filename)


    original = ['#1f77b4', '#ff7f0e', '#2ca02c'] 
    


    plt.figure(figsize=(8, 5))
    
    for i, model in enumerate(df["Model"].unique()):
        subset = df[df["Model"] == model]  
        plt.plot(subset["n_Epochs"], subset["ROC_AUC"], marker='o', linestyle='-', 
                color=original[i % len(original)], linewidth=5, markersize=10, label=model)

    # Labels and title
    plt.xlabel("Epochs", fontsize=16)
    plt.ylabel("ROC-AUC", fontsize=16)
    plt.title("ROC_AUC for different Models", fontsize=16)

    # Show grid lines for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)


    # Add legend
    plt.legend(title="Model", fontsize=14)

    # Save and show the plot
    plt.savefig("SUPERVISED.png")  
    plt.show()

            

# Execute main() only when the script is run directly
if __name__ == "__main__":
    main()