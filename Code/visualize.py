import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,precision_recall_fscore_support, confusion_matrix,fbeta_score

'''
Prints validation Accuracy,Precision,Recall, other metrics 
'''
def metrics(true_values,pred_values,name):
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
    plot_confusion_matrix(genre_list,conf_matrix,name)

'''    
Plots Confusion Matrix
'''
def plot_confusion_matrix(genre_list, mat,name):
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
    plt.savefig('../Metrics/confusion_matrix_'+name+'.png')

def plot_loss(test_err,name):
    xvals = np.arange(0,len(test_err))
    plt.plot(xvals, test_err)
    plt.xlabel("epochs")
    plt.ylabel("test error")
    plt.savefig('../Metrics/loss_'+name+'.png') 