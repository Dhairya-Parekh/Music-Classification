'''
Feed forward Neural Network for Music Genre Classification AI-ML Project
Dhairya Parekh(200050097)| Utkarsh Pratap Singh(200050146)| Naman Singh Rana(200050083)| Aditya Kadoo(200050055)
'''
import pandas as pd
import torch
import torch.nn as nn
import torch.functional as F
import numpy as np
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
    plt.savefig('confusion_matrix.png')


class MC(nn.Module):
    def __init__(self,input_size) -> None:
        super().__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size,60)
        self.fc2 = nn.Linear(60,30)
        self.fc3 = nn.Linear(30,10)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.drop1 = nn.Dropout(0.05)
        self.drop2 = nn.Dropout(0.05)
        
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)

        
    def forward(self,x):
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.fc2(x)
        x = self.tanh(x)
        x = self.fc3(x)
        return x

'''      
Training neural network
'''
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
    metrics(Y_val.tolist(),y_preds.tolist())



       
    
    
if __name__ == '__main__':
    
    df_lstm = pd.read_csv("../Dataset/features_3_sec.csv")
    df_svm = pd.read_csv("../Dataset/features_30_sec.csv")
    
    
    
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

    '''
    Split data into train and validation sets
    '''
    tc = int(0.9*len(Y))
    X_trn = X[:tc,:]
    Y_trn = Y[:tc]
    X_val = X[tc:,:]
    Y_val = Y[tc:]
    
    model = MC(X_trn.shape[1])
    train(model,X_trn,Y_trn,X_val,Y_val,80)
    
    