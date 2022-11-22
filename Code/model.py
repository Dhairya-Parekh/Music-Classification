import pandas as pd
import torch
import torch.nn as nn
import torch.functional as F
import numpy as np

class MC(nn.Module):
    def __init__(self,input_size) -> None:
        super().__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size,50)
        self.fc2 = nn.Linear(50,50)
        self.fc3 = nn.Linear(50,10)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)
        
    def forward(self,x):
        output = self.fc3(self.tanh(self.fc2(self.tanh(self.fc1(x)))))
        return output

def train(model,X_trn,Y_trn,X_val,Y_val,epochs=80):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_func = torch.nn.CrossEntropyLoss()
    count = len(X_trn)
    for epoch in range(epochs):
        running_loss = 0
        for i  in range(count):
            optimizer.zero_grad()
            v = model(X_trn[i])
            loss = loss_func(v,Y_trn[i])
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print("epoch: ",epoch," loss: ",running_loss)
        
    y_preds = torch.empty(size=(len(X_val),))
    with torch.no_grad():
        for i in range(len(X_val)):
            v = model(X_val[i])
            y_preds[i] = torch.argmax(v)
    print("accuracy: ",sum(y_preds==Y_val)/len(Y_val))
       
    
    
if __name__ == '__main__':
    
    df_lstm = pd.read_csv("/Users/nsr/Desktop/sem 5/AI ML Lab/Music-Classification/Dataset/features_3_sec.csv")
    df_svm = pd.read_csv("/Users/nsr/Desktop/sem 5/AI ML Lab/Music-Classification/Dataset/features_30_sec.csv")
    
    
    
    df_svm.drop(['filename','length'],axis=1,inplace=True)
    cols = df_svm.columns
    for col in cols:
        if col.endswith("var"):
            df_svm.drop([col],axis=1,inplace=True)
    df_svm = df_svm.sample(frac=1)
    labels = df_svm['label'].unique()
    Y = df_svm['label']
    df_svm.drop(['label'],axis=1,inplace=True)
    # normalization
    mean = df_svm.mean(axis=0)
    sd = df_svm.std(axis=0)
    df_svm = (df_svm-mean)/sd
    
    mapping = {}
    for i,label in enumerate(labels):
        mapping[i] = label
        Y[Y==label] = i
    
    Y = torch.tensor(Y)
    X = torch.tensor(df_svm.values.astype(np.float32))
    
    tc = int(0.9*len(Y))
    X_trn = X[:tc,:]
    Y_trn = Y[:tc]
    X_val = X[tc:,:]
    Y_val = Y[tc:]
    
    model = MC(X_trn.shape[1])
    train(model,X_trn,Y_trn,X_val,Y_val,60)
    
    