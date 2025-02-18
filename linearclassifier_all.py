import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pickle
import time


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4,6"


import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, roc_auc_score 
import numpy as np

import wandb

import itertools

import csv
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from T_CPC import T_CPC
from TS_CPC import TS_CPC
from S_CPC import S_CPC


def train_extract(model, loader, device, optimizer=None):


    model.train()


    all_c_t = []  # To store all the hidden states (features)
    all_labels = []  

    total_loss = 0
    batch_count = 0
   

    for i, (data, acc, label, patient_id) in enumerate(loader):

        data = data.to(device)
        label = label.to(device)

        acc_onehot = F.one_hot(acc, 16).to(device)

        data = data.unsqueeze(2).float() * acc_onehot.float()
        data = data.permute(0, 2, 1).to(device)  
            
            

        hidden = model.init_hidden(len(data), True).to(device)
        loss, _, _, _ = model(data, hidden.float())  

        total_loss += loss.item() 
        batch_count +=1

        output, _ = model.predict(data, hidden)

        # THIS IS A DIFFERENT C_T that is not dependent on randomly chosen t value (--> change in size)
        c_t = output[:, -1].squeeze_()
        # output retains temporal info for all timesteps
        # output[:,-1] extracts hidden state at the last timestep of the GRU output. Because GRUs propagate info sequentially,the last state has theoretically 'seen' all the previous timesteps.
        # it is a single representation for the entire sequence
            
        

        all_c_t.append(c_t.detach())
        all_labels.append(label.detach())


        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    average_loss = total_loss/batch_count

    # Concatenate all the batches together
    X_train = torch.cat(all_c_t, dim=0).cpu().numpy()  # Features for the entire dataset
    Y_train = torch.cat(all_labels, dim=0).cpu().numpy()  # Labels for the entire dataset
    

    #print(f"Data device: {data.device}, Label device: {label.device}, Hidden device: {hidden.device}, Model device: {next(model.parameters()).device}")


    return X_train, Y_train, average_loss





def test_extract(model, loader, device):
    
    model.eval()

    all_c_t = []  
    all_labels = []  


    total_loss=0
    batch_count=0


    with torch.no_grad():
   

        for i, (data, acc, label, patient_id) in enumerate(loader):

            data = data.to(device)
            label = label.to(device)

            acc_onehot = F.one_hot(acc, 16).to(device)

            data = data.unsqueeze(2).float() * acc_onehot.float()
            data = data.permute(0, 2, 1).to(device)  
       
            
            # idx = (patient_id == 145) + (patient_id == 106) + (patient_id == 116) + (patient_id == 176)
            
            # idx = patient_id >100
           
            
            
            # data_idx = data[idx]
            # data_idx = data_idx.float().to(device)
            # label_idx = label[idx]



            hidden = model.init_hidden(len(data), True).to(device)
            loss, _, _, _ = model(data, hidden.float())  

            total_loss += loss.item() 
            batch_count +=1

            output, _ = model.predict(data, hidden)

            # THIS IS A DIFFERENT C_T that is not dependent on randomly chosen t value (--> change in size)
            c_t = output[:, -1].squeeze_()
            
            
        

            all_c_t.append(c_t.detach())
            all_labels.append(label.detach())

    


    average_loss = total_loss / batch_count

    # Concatenate all the batches together
    X_test = torch.cat(all_c_t, dim=0).cpu().numpy() 
    Y_test = torch.cat(all_labels, dim=0).cpu().numpy()  


    #print(f"Data device: {data.device}, Label device: {label.device}, Hidden device: {hidden.device}, Model device: {next(model.parameters()).device}")
    
    return X_test, Y_test, average_loss



class ClassifierNN(nn.Module):
    def __init__(self, ini):
        super().__init__()  # parent class nn.Module is properly intialized before adding layers
        self.classifi = nn.Sequential(
            nn.Linear(ini,64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64,2)
        )

    def forward(self,x):
        return self.classifi(x)




def linearclassifier(type, X_train, Y_train, X_test, Y_test, log_graph=False):

    if type == 'LogisticRegression':
    # ** press shift + tab to unindent
        Y_train = Y_train.flatten()
        Y_test = Y_test.flatten()

        # Create logistic regression object (for binary classification)
        classifier = LogisticRegression()

        classifier.fit(X_train,Y_train)

        Y_pred = classifier.predict(X_test)   
        probs = classifier.predict_proba(X_test)
    

    # paper : fully connected network with two hidden layers
    elif type == 'ClassifierNN':

        device = 'cuda' if torch.cuda.is_available() else 'cpu'


        Y_train = torch.tensor(Y_train.flatten()).to(device)
        X_train = torch.tensor(X_train).to(device)
        X_test = torch.tensor(X_test).to(device)
            


        classifier = ClassifierNN(X_train.shape[1]).to(device)
        lossfc = nn.CrossEntropyLoss()
        optimizer = optim.Adam(classifier.parameters())


        for epoch in range(64):
            classifier.train()
            optimizer.zero_grad()
            logits = classifier(X_train)
            loss = lossfc(logits, Y_train)
            loss.backward()
            optimizer.step()

        
        classifier.eval()
        with torch.no_grad():
            logits = classifier(X_test)
            probs = torch.softmax(logits, dim=1).cpu().numpy()           # Convert to probabilities
            Y_pred = probs.argmax(axis=1)



############################################################
    # Accuracy and classification report
    accuracy = accuracy_score(Y_test, Y_pred)
    #print(f"Accuracy: {accuracy:.2f}")


    report = classification_report(Y_test, Y_pred, output_dict=True)  # Use output_dict=True to get a dictionary

    
    wandb.log({
        "precision_non-AFib": report['0']['precision'],
        "recall_non-AFib": report['0']['recall'],
        "f1_non-AFib": report['0']['f1-score'],
        "precision_AFib": report['1']['precision'],
        "recall_AFib": report['1']['recall'],
        "f1_AFib": report['1']['f1-score'],
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

        plt.scatter(X_test[Y_test == 0][:, 0], X_test[Y_test == 0][:, 1], color='#2283a2', label="Class 0 (Test)", marker='o')
        plt.scatter(X_test[Y_test == 1][:, 0], X_test[Y_test == 1][:, 1], color='#f66209', label="Class 1 (Test)", marker='x')
        plt.legend(['non-AF', 'AF'], prop={'size': 15})

        plt.title("Linear Classifier classification")
        plt.show()
        wandb.log({"classification": wandb.Image(plt)})
        plt.close()



    return accuracy, roc_auc



            

  
    

def train(num_epochs, model, type, optimizer, train_loader, test_loader, device, test_every=1):

    history=[]

    start = time.time()
    accuracy_list = []
    epoch_list = []
    train_loss_list = []
    test_loss_list = []

    
    for epoch in range(num_epochs):

        _, _, train_loss = train_extract(model, train_loader, device, optimizer)
        train_loss_list.append(train_loss)
        
        # Validate on train and test set
        if epoch % test_every == 0 or epoch == num_epochs - 1:
            
           # is repetitive to have training done again
            train_c_t, train_label, _ = test_extract(model, train_loader, device)
            test_c_t, test_label, test_loss = test_extract(model, test_loader, device)
            test_loss_list.append(test_loss)

            # Pass the NumPy arrays to your linear classifier
            if epoch == num_epochs - 1:
                accuracy, roc_auc = linearclassifier(type, train_c_t, train_label, test_c_t, test_label, log_graph=True)
            else:
                accuracy, roc_auc = linearclassifier(type, train_c_t, train_label, test_c_t, test_label, log_graph=False)
                    
            accuracy_list.append(accuracy * 100)
            epoch_list.append(epoch)
            print (f'for Epoch {epoch}, Accuracy: {accuracy*100}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')


            wandb.log({"epoch": epoch, "accuracy": accuracy, 'train_loss': train_loss, 'test_loss' :test_loss})

   

            history.append((epoch,roc_auc))
            
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
    


# move all executable codes into main() function. leave function definitions and imports, outside above
# code inside main() function would not execute when script is imported into another script. it would only run when the script is run directly


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # batch_size = 64



    path = '/home/ria/MMBSintern/bachelor/exercise/data' 

    train_data = Wearable(path, True)
    test_data = Wearable(path, False)


   

    # Hyperparameters

    results = []

    modelname=['T_CPC', 'TS_CPC', 'S_CPC']
    type=['LogisticRegression', 'ClassifierNN']
    n_epo=10

    #lrl = [1e-3,1e-7]
    #batch_sizes = [256,1024]

    lrl = [1e-3]
    batch_sizes = [1024]

    # Generate all combinations 
    combinations = list(itertools.product(modelname, type, batch_sizes, lrl))

    # Run each combination with each value of num_epochs

    for modelname, type, bs, lrl in combinations:

        train_loader = DataLoader(train_data, batch_size=bs, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_data, batch_size=len(test_data), shuffle=False, num_workers=0) 

        
        
        test_every = n_epo / 10
        lr = (bs/1024) * lrl
    
        print(f"Running with num of epochs: {n_epo}, model: {modelname}, classification:{type}, learning rate: {lr}, batch size: {bs}")
        
    
        if modelname == 'T_CPC':
            model = T_CPC(timestep=3, batch_size=bs, seq_len=200).to(device)

        elif modelname == 'TS_CPC':
            model = TS_CPC(timestep=3, batch_size=bs, seq_len=200).to(device)

        elif modelname == 'S_CPC':
            model = S_CPC(timestep=3, batch_size=bs, seq_len=200).to(device)

        else:
            raise ValueError(f"Unknown model name: {modelname}")

        optimizer = optim.AdamW(model.parameters(), lr=lr)

    
        config = dict(num_epoch = n_epo,
                test_every = test_every,
                model = modelname,
                classification = type,
                learning_r = lr,
                batch_size = bs)
            
        
        wandb.init(project="UNSUPERVISED_learning", 
            name=f"{n_epo, modelname, type, lr, bs}",
            config=config,
            reinit=True)


        history = train(config['num_epoch'], model, config['classification'], optimizer, train_loader, test_loader, device, config['test_every'])
        
        for epoch, roc_auc in history:  
            results.append([epoch, modelname, type, bs, roc_auc])


        wandb.finish() 


    df = pd.DataFrame(results, columns=['n_Epochs','Model','Classification','Batch Size','ROC_AUC'])
    csv_filename = 'UNSUPERVISED.csv'

    df.to_csv(csv_filename, index=False)
    print(f"Results saved to {csv_filename}")


    # Load CSV and plot results
    df = pd.read_csv(csv_filename)

    df["Label"] = df["Model"] + " | " + df["Classification"]

    original = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
    lighter = [mcolors.to_rgba(c, alpha=0.5) for c in original]  # Lighter shades with transparency

    all=[]
    for i in range(len(original)):
        all.append(original[i])
        all.append(lighter[i])
        


    plt.figure(figsize=(10, 7))
    
    for i, label in enumerate(df["Label"].unique()):
        subset = df[df["Label"] == label]  
        plt.plot(subset['n_Epochs'], subset["ROC_AUC"], marker='o', linestyle='-', 
                color=all[i % len(all)], linewidth=5, markersize=10, label=label)

    # Labels and title
    plt.xlabel("Epochs", fontsize=16)
    plt.ylabel("ROC-AUC", fontsize=16)
    plt.title("ROC_AUC for different Model and Classification Combinations", fontsize=16)

    # Show grid lines for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)


    # Add legend
    plt.legend(title="Model | Classification", fontsize=12, loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()

    # Save and show the plot
    plt.savefig("UNSUPERVISED.png")  
    plt.show()


            

# Execute main() only when the script is run directly
if __name__ == "__main__":
    main()