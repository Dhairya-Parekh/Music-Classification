import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset,DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,precision_recall_fscore_support, confusion_matrix,fbeta_score


'''
Prints validation Accuracy,Precision,Recall, other metrics 
'''
def metrics(true_values,pred_values):
    accuracy=accuracy_score(true_values,pred_values)
    precision,recall,f1_score,support = precision_recall_fscore_support(true_values,pred_values,average='weighted')
    f2_score = fbeta_score(true_values,pred_values,beta=2.0,average='weighted')
    f_5_score = fbeta_score(true_values,pred_values,beta=0.5,average='weighted')
    print("Accuracy: ",accuracy)
    print("Precision : ",precision )
    print("Recall : ",recall)
    print("F1 Score : ",f1_score)
    print("F2 Score : ",f2_score)
    print("F 0.5 Score : ",f_5_score)
    genre_list = ['blues','classical','country','disco','hiphop','jazz','metal','pop','reggae','rock']
    genre_list = sorted(genre_list)
    conf_matrix = confusion_matrix(true_values,pred_values,labels=range(10),normalize="true")
    plot_confusion_matrix(genre_list,conf_matrix)

'''    
Plots Confusion Matrix
'''
def plot_confusion_matrix(genre_list, mat):
    conf_matrix = np.copy(mat)
    conf_matrix = conf_matrix / np.sum(conf_matrix, axis=1)
    box = plt.subplots(figsize=(30,15))[1]
    box.xaxis.set_ticks_position('top')
    box.xaxis.set_label_position('top')
    fmt = '.2f'
    plt.imshow(conf_matrix, interpolation='nearest')
    tick_marks = np.arange(len(genre_list))
    plt.yticks(tick_marks, genre_list)
    plt.xticks(tick_marks, genre_list)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            color = "black" if conf_matrix[i,j]>0.5 else "white"
            plt.text(j, i, format(conf_matrix[i, j], fmt), horizontalalignment="center",color=color)
    plt.ylabel('True Genre',fontsize=22)
    plt.xlabel('Predicted Genre', fontsize=22)
    plt.savefig('confusion_matrix2.png')

class MusicDataset(Dataset):
    def __init__(self,X,Y):
        self.X = X
        self.Y = Y
    
    def __len__(self):
        return len(self.Y)
    
    def __getitem__(self, index):
        return self.X[index], self.Y[index]

class lstm_model(nn.Module):
    def __init__(self,input_dim,hidden_size,num_layers,batch_size) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.lstm = nn.LSTM(input_size=input_dim,hidden_size=hidden_size, num_layers = num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size,50)
        self.fc2 = nn.Linear(50,25)
        self.fc3 = nn.Linear(25,10)
        self.tanh = nn.Tanh()
        
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)
        
    
    def forward(self,x):
        h_0 = Variable(torch.zeros(self.num_layers,self.batch_size,self.hidden_size)) 
        c_0 = Variable(torch.zeros(self.num_layers,self.batch_size,self.hidden_size))
        out,_ = self.lstm(x,(h_0,c_0))
        out = self.fc1(out[:,-1,:])
        out = self.tanh(out)
        out = self.fc2(out)
        out = self.tanh(out)
        out = self.fc3(out)
        return out


def train(model,train_dataloader,val_dataloader,epochs=80):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_func = torch.nn.CrossEntropyLoss()
    for epoch in range(epochs):
        running_loss = 0
        for x,y in train_dataloader:
            optimizer.zero_grad()
            v = model(x)
            loss = loss_func(v,y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print("epoch: ",epoch," loss: ",running_loss)
        
    correct_preds = 0
    total_preds = 0
    Y_val=[]
    Y_pred=[]
    with torch.no_grad():
        for x,y in val_dataloader:
            v = model(x)
            y_preds = torch.argmax(v,axis=1)
            Y_val+=y.tolist()
            Y_pred+=y_preds.tolist()
    metrics(Y_val,Y_pred)
    #print("accuracy: ",correct_preds/total_preds)
        
        
        

if __name__ == '__main__':
    df = pd.read_csv("../Dataset/features_3_sec.csv")
    df.drop(['filename','length'],axis=1,inplace=True)
    labels = df['label'].unique()
    cols = df.columns
    for col in cols:
        if col.endswith("var"):
            df.drop([col],axis=1,inplace=True)
    Y = df['label'][df.index%10==0]
    mapping = {}
    for i,label in enumerate(labels):
        mapping[i] = label
        Y[Y==label] = i
    df.drop(['label'],axis=1,inplace=True)
    X = df.to_numpy().reshape(len(Y),10,-1)
    Y = Y.to_numpy().reshape(len(Y))
    indx = np.random.choice(len(Y),len(Y),replace=True)
    X = X[indx]
    Y = Y[indx]
    X = torch.tensor(X.astype(np.float32))
    Y = torch.tensor(Y.astype(np.uint8))
    X = (X - torch.mean(X))/torch.std(X)
    tc = int(0.9*len(Y))
    X_trn = X[:tc,:,:]
    Y_trn = Y[:tc]
    X_val = X[tc:,:,:]
    Y_val = Y[tc:]
    
    train_dataset = MusicDataset(X_trn,Y_trn)
    val_dataset = MusicDataset(X_val,Y_val)
    
    batch_size = 16
    train_dataloader = DataLoader(train_dataset,batch_size,drop_last=True)
    val_dataloader = DataLoader(val_dataset,batch_size,drop_last=True)
    
    model = lstm_model(input_dim=X_trn.shape[2],hidden_size=100,num_layers=3,batch_size=batch_size)
    train(model,train_dataloader,val_dataloader,150)
    
    
    