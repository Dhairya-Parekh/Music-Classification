import pandas as pd
import torch
import torch.nn as nn
import torch.functional as F
import numpy as np


class MC(nn.Module):
    def __init__(self,input_size,output_size) -> None:
        super().__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size,200)
        self.fc2 = nn.Linear(200,100)
        self.fc3 = nn.Linear(100,output_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        return self.softmax(self.fc3(self.sigmoid(self.fc2(self.fc1(x)))))
        # return self.softmax(self.relu(self.fc1(x)))

def train(model,X_trn,Y_trn,X_val,Y_val,epochs=80):
    epochs = 10
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
            y_preds[i] = torch.amax(v)
    print("accuracy: ",sum(y_preds==Y_val)/len(Y_val))
       
    
    
if __name__ == '__main__':
    
    df_lstm = pd.read_csv("/Users/nsr/Desktop/sem 5/AI ML Lab/Music-Classification/Dataset/features_3_sec.csv")
    df_svm = pd.read_csv("/Users/nsr/Desktop/sem 5/AI ML Lab/Music-Classification/Dataset/features_30_sec.csv")
    
    
    
    df_svm.drop(['filename','length'],axis=1,inplace=True)
    df_svm = df_svm.sample(frac=1)
    labels = df_svm['label'].unique()
    Y = df_svm['label']
    df_svm.drop(['label'],axis=1,inplace=True)
    mapping = {}
    for i,label in enumerate(labels):
        mapping[i] = label
        Y[Y==label] = i
    # print(df_svm)
    Y = torch.tensor(Y)
    X = torch.tensor(df_svm.values.astype(np.float32))
    # print(Y.shape,X.shape)
    tc = int(0.8*len(Y))
    X_trn = X[:tc,:]
    Y_trn = Y[:tc]
    X_val = X[tc:,:]
    Y_val = Y[tc:]
    # print(Y_trn,X_trn)
    model = MC(X_trn.shape[1],10)
    train(model,X_trn,Y_trn,X_val,Y_val,80)
    
    