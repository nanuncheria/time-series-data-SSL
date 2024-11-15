import torch
import torch.nn.functional as F
import torch.optim as optim

import time
import matplotlib.pyplot as plt



import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5,6"


import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np



def train_extract(model, loader, device, optimizer):


    model.train()


    all_c_t = []  # To store all the hidden states (features)
    all_labels = []  # To store all the labels
   

    for i, (data, acc, label, patient_id) in enumerate(loader):

        data = data.to(device)
        label = label.to(device)

        acc_onehot = F.one_hot(acc, 16).to(device)

        data = data.unsqueeze(2).float() * acc_onehot.float()
        data = data.permute(0, 2, 1).to(device)  # add channel dimension
        # permute changes the order of dimensions of 'data'




        hidden = model.init_hidden(len(data), True).to(device)   
       

        loss, _, hidden, c_t = model(data, hidden.float())   
            
        

        all_c_t.append(c_t.detach())
        all_labels.append(label.detach())



        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


            # Concatenate all the batches together
    X_train = torch.cat(all_c_t, dim=0).cpu().numpy()  # Features for the entire dataset
    Y_train = torch.cat(all_labels, dim=0).cpu().numpy()  # Labels for the entire dataset
    

    #print(f"Data device: {data.device}, Label device: {label.device}, Hidden device: {hidden.device}, Model device: {next(model.parameters()).device}")


    return X_train, Y_train





def test_extract(model, loader, device, idx):
    

    model.eval()

    all_c_t = []  # To store all the hidden states (features)
    all_labels = []  # To store all the labels


    with torch.no_grad():
   

        for i, (data, acc, label, patient_id) in enumerate(loader):

            data=data.to(device)
            label = label.to(device)

            acc_onehot = F.one_hot(acc, 16).to(device)

            data = data.unsqueeze(2).float() * acc_onehot.float()
            data = data.permute(0, 2, 1).to(device)  # add channel dimension
            # permute changes the order of dimensions of 'data'

#---------         
            if idx == 'same':
                idx = (patient_id == 145) + (patient_id == 106) + (patient_id == 116) + (patient_id == 176)
            elif idx == 'id > 100':
                idx = patient_id >100
            else:
                print('Error: idx undefinable')
            
            
            data_idx = data[idx]
            data_idx = data_idx.float().to(device)
            label_idx = label[idx]




#------
            hidden = model.init_hidden(len(data_idx), True)   # previously data

            loss, accuracy, hidden, c_t = model(data_idx, hidden.float())   # previously data


            
            all_c_t.append(c_t.detach())
            all_labels.append(label_idx.detach())   # previously label


                 # Concatenate all the batches together
    X_test = torch.cat(all_c_t, dim=0).cpu().numpy()  # Features for the entire dataset
    Y_test = torch.cat(all_labels, dim=0).cpu().numpy()  # Labels for the entire dataset


    #print(f"Data device: {data.device}, Label device: {label.device}, Hidden device: {hidden.device}, Model device: {next(model.parameters()).device}")
    
    return X_test, Y_test


from sklearn.metrics import classification_report, roc_auc_score 

def linearclassifier(function, X_train, Y_train, X_test, Y_test):

    if function == 'all considered':

        Y_train = Y_train.flatten()
        Y_test = Y_test.flatten()

        # Create logistic regression object (for binary classification)
        classifier = LogisticRegression()

        classifier.fit(X_train,Y_train)

        y_pred = classifier.predict(X_test)


        # Accuracy and classification report
        accuracy = accuracy_score(Y_test, y_pred)
        #print(f"Accuracy: {accuracy:.2f}")

    
        report = classification_report(Y_test, y_pred, output_dict=True)  # Use output_dict=True to get a dictionary

        # Log metrics to W&B
        wandb.log({
            "precision_non-AFib": report['0']['precision'],
            "recall_non-AFib": report['0']['recall'],
            "f1_non-AFib": report['0']['f1-score'],
            "precision_AFib": report['1']['precision'],
            "recall_AFib": report['1']['recall'],
            "f1_AFib": report['1']['f1-score'],
            "accuracy": report['accuracy']
        })
    
        auc_score = roc_auc_score(Y_test, classifier.predict_proba(X_test)[:, 1])
        wandb.log({"ROC-AUC Score": auc_score})
        

        plt.figure(figsize=(5,5))


        # Scatter plot for the test data points
        plt.scatter(X_test[Y_test == 0][:, 0], X_test[Y_test == 0][:, 1], color='#2283a2', label="Class 0 (Test)", marker='o')
        plt.scatter(X_test[Y_test == 1][:, 0], X_test[Y_test == 1][:, 1], color='#f66209', label="Class 1 (Test)", marker='x')


        plt.title("Logistic Regression")
        plt.show()
        plt.close()

        return accuracy


    elif function == '2PC/2D considered':
    
    # reduce the number of features plotted to be 2
    # since when plotting a decision boundary, it requires the feature space to be two-dimensional


        Y_train = Y_train.flatten()
        Y_test = Y_test.flatten()

        # Reduce dimensionality to 2 features using PCA
        pca = PCA(n_components=2)
        X_train_2D = pca.fit_transform(X_train)
        X_test_2D = pca.transform(X_test)

        # Create logistic regression object (for binary classification)
        classifier = LogisticRegression()

        # Train the model using the reduced 2D training sets
        classifier.fit(X_train_2D, Y_train)
    

        # Make predictions using the reduced 2D testing set
        y_pred = classifier.predict(X_test_2D)
        


        # Accuracy 
        accuracy = accuracy_score(Y_test, y_pred)
        #print(f"Accuracy: {accuracy:.2f}")

    
    
        

        plt.figure(figsize=(5,5))


        # Scatter plot for the test data points
        plt.scatter(X_test[Y_test == 0][:, 0], X_test[Y_test == 0][:, 1], color='#2283a2', label="Class 0 (Test)", marker='o')
        plt.scatter(X_test[Y_test == 1][:, 0], X_test[Y_test == 1][:, 1], color='#f66209', label="Class 1 (Test)", marker='x')


        # Plot the decision boundary : draw a straight line based on model coefficients

        x_values = np.linspace(X_train[:, 0].min(), X_train[:, 0].max(), 100)

        # The equation for the decision boundary is derived from the logistic regression formula: 
        # w1 * x1 + w2 * x2 + b = 0
        # we solve for x2(y-values) to get a straight line: x2 = -(w1 * x1 + b)/w2
    # This is the line separating the two classes.

        y_values = -(classifier.coef_[0][0] * x_values + classifier.intercept_[0]) / classifier.coef_[0][1]
        plt.plot(x_values, y_values, label="Decision Boundary", color='#28a745')


        plt.xlabel('first Principal Component')
        plt.ylabel('second Principal Component')
        plt.legend()

        plt.title("Logistic Regression - Decision Boundary in 2D")
        plt.show()
        plt.close()


        return accuracy
    

    else:
        print('Error: linear classifier undefinable')



def train(num_epochs, model, optimizer, function, train_loader, test_loader, idx, device, test_every=1):
    start = time.time()
    accuracy_list = []
    epoch_list = []

    
    for epoch in range(num_epochs):
       
        train_c_t, train_label = train_extract(model, train_loader, device, optimizer)
        # Validate on train and test set
        if epoch % test_every == 0 or epoch == num_epochs - 1:
            
           # is repetitive to have training done again
            #train_c_t, train_label = train_extract(model, train_loader, device, optimizer)
            test_c_t, test_label = test_extract(model, test_loader, device, idx)


            # Pass the NumPy arrays to your linear classifier
            accuracy = linearclassifier(function, train_c_t, train_label, test_c_t, test_label)
           
            accuracy_list.append(accuracy * 100)
            epoch_list.append(epoch)
            print (f'for Epoch {epoch}, Accuracy: {accuracy*100}')


            wandb.log({"epoch": epoch, "accuracy": accuracy})


            
    plt.plot (epoch_list, accuracy_list)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Model Accuracy over Epochs')
    plt.show()
    #plt.savefig(f"{config['num_epochs']},{config['function']},{config['idx']}.png") # Save the plot as an image file 
    wandb.log({f"{config['num_epochs']},{config['function']},{config['idx']}": wandb.Image(plt)})
    plt.close() # Close the plot to free memory

    print(f"Total training time was {(time.time() - start) / 60:.1f}m.")







import wandb
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5,6"


from torch.utils.data import DataLoader, Dataset
import pickle




#use_cuda = torch.cuda.is_available()
#print('use_cuda is', use_cuda)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


from CPC import CPC

model = CPC(timestep=3, batch_size=64, seq_len=200).to(device)

    
optimizer = optim.AdamW(model.parameters())


#from utils import ScheduledOptim

  
# optimizer = ScheduledOptim(optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
            # betas=(0.9, 0.98), eps=1e-09, weight_decay=1e-4, amsgrad=True),n_warmup_steps=50)


model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#print(f'# Model summary #\n {str(model)}\n')
#print(f'===>  total number of parameters: {model_params}\n')


# capture a dictionary of hyperparameters with config





import itertools

batch_size = 64


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




train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(test_data, batch_size=len(test_data), shuffle=False, num_workers=0) 


function = ['2PC/2D considered', 'all considered']
index = ['same', 'id > 100']
epochs = [10, 50, 100]

# Generate all combinations of function and index
combinations = list(itertools.product(function, index))

# Run each combination with each value of num_epochs
for func, idx in combinations:
    for num_epochs in epochs:
        test_every = num_epochs/10
        print(f"Running with function: {func}, idx: {idx}, epochs: {num_epochs}")
        # Call your function here, e.g., train_model(func, idx, epochs)
        # 
        # 
        # 
      
        config = dict( num_epochs = num_epochs,
               test_every = test_every,
               batch_size = batch_size,
               function = func,
               idx = idx)
        
        
        wandb.init(project="LinearClassification!!", 
            name=f"{num_epochs,func,idx}",
            config=config,
            reinit=True)


        stats = train(config['num_epochs'], model, optimizer, config['function'], train_loader, test_loader, config['idx'], device, config['test_every'])
        
        wandb.finish() 