# -*- coding: utf-8 -*-
"""
Created on Sat Jan 14 12:41:43 2023

@author: alexa
"""
import numpy as np
import os
import pandas as pd

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader, TensorDataset


#testare
def testare():
    predictii = []
    test_labels = []
    model.eval()
    for batch in test_loader:
        batch_data = batch[0].to(device)
        batch_labels = batch[1].to(device)

        current_predict = model.forward(batch_data)
        current_predict = current_predict.cpu()
        batch_labels = batch_labels.cpu()
    
        current_predict = np.argmax(current_predict.detach().numpy(),axis=1)
        predictii = np.concatenate((predictii,current_predict))
        test_labels = np.concatenate([test_labels, batch_labels])

    acc = np.sum(predictii==test_labels)/len(predictii)
    print( 'Acuratetea pe setul de testare este {}%'.format(acc*100) )
    return acc



#load csv file as dataframe

emotii = pd.read_csv("fer2013.csv")

#filter angry and happy classes (0=Angry 3=Happy)
emotii_filtrat = emotii[(emotii['emotion'] == 0) | (emotii['emotion'] == 3)]

#count_test = emotii_filtrat['emotion'].value_counts()
#See the class division
count_test1 = emotii_filtrat['Usage'].value_counts()

#Merge PublicTest with PrivateTest in Test to have 80% train and 20%test

emotii_filtrat = emotii_filtrat.replace('PublicTest','Test')
emotii_filtrat = emotii_filtrat.replace('PrivateTest', 'Test')
emotii_filtrat = emotii_filtrat.replace('Training', 'Train')

#count_test2 = emotii_filtrat['Usage'].value_counts()

#Rename colums 
emotii_filtrat= emotii_filtrat.rename(columns={"emotion": "classes","pixels": "images"})

#Make 1 the identifier for class "happy"
emotii_filtrat['classes'] = emotii_filtrat['classes'].replace(3,1)

#count_test3 = emotii_filtrat['classes'].value_counts()

#Save as csv without Usage column 
emotii_test = emotii_filtrat[emotii_filtrat['Usage']=='Test']
emotii_test = emotii_test.drop(columns=["Usage"])

if os.path.isfile('test_data.csv') == False:
  emotii_test.to_csv('test_data.csv', index=False)


emotii_train = emotii_filtrat[emotii_filtrat['Usage']=='Train']
emotii_train = emotii_train.drop(columns=["Usage"])

if os.path.isfile('train_data.csv') == False:
    emotii_train.to_csv('train_data.csv', index=False)


#Show an image from the images column 

emotion_label = {0: 'Anger', 1:'Happy'}

fig = plt.figure(1, (12,12))
k=0

for label in sorted(emotii_train.classes.unique()):
    for j in range(4):
        px = emotii_train[emotii_train.classes==label].images.iloc[k]
        px = np.array(px.split(' ')).reshape(48,48).astype('float32')
        
        k += 1
        ax = plt.subplot(4,4,k)
        ax.imshow(px, cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(emotion_label[label])
        plt.tight_layout()

fig = plt.figure(2, (12,12))
k=0

for label in sorted(emotii_test.classes.unique()):
    for j in range(4):
        px = emotii_test[emotii_test.classes==label].images.iloc[k]
        px = np.array(px.split(' ')).reshape(48,48).astype('float32')
        
        k += 1
        ax = plt.subplot(4,4,k)
        ax.imshow(px, cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(emotion_label[label])
        plt.tight_layout()
        

#From dataframe to np.array
train = emotii_train['images'].apply(lambda x: np.array(x.split()).reshape(1, 48, 48).astype('float32'))
train = np.stack(train, axis=0)
#print(train.shape)

img_labels_train = emotii_train.classes
#print(img_labels_train.shape)

#TODO this for test
test = emotii_test['images'].apply(lambda x: np.array(x.split()).reshape(1, 48, 48).astype('float32'))
test = np.stack(test, axis=0)
#print(test.shape)

img_labels_test = emotii_test.classes
#print(img_labels_test.shape)

# Transform the data to tensors
tensor_img_train = torch.Tensor(train)
tensor_lb_train = torch.Tensor(img_labels_train.to_numpy()).long()

tensor_img_test = torch.Tensor(test)
tensor_lb_test = torch.Tensor(img_labels_test.to_numpy()).long()

# Create the train and test datasets
train_dataset = TensorDataset(tensor_img_train, tensor_lb_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = TensorDataset(tensor_img_test, tensor_lb_test)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

#architecture
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = "cpu"
print(device)



# Instantiem reteaua
model = models.efficientnet_b0(pretrained=True)

#num_features = model.fc.in_features
#model.fc = nn.Linear(num_features, 2)

#print(model.classifier[0])
#print(type(model.classifier[1]))
#print(model.features[0])
model.features[0][0]=nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
model.classifier[1] = nn.Linear(in_features=1280, out_features=2)
model.to(device)
#print(model)
# Specificarea functiei loss
loss_function = nn.CrossEntropyLoss()
#loss_function = nn.BCELoss()
# Specificarea optimizatorului
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

#antrenare
nr_epoci = 1
train_acc = []
test_acc = []
model = model.train()

for ep in range(nr_epoci):
    predictii = []
    etichete = []
    running_corects = 0
    for i, batch in enumerate(train_loader):
        #print(i)
        batch_data = batch[0].to(device)
        batch_labels = batch[1].to(device)
        # Se calculeaza predictia retelei pentru datele curente (forward pass/ propagare inainte)
        current_predict = model.forward(batch_data)

        # Se calculeaza valoarea momentana a functiei loss
        loss = loss_function(current_predict, batch_labels) 
        
        # Se memoreaza predictiile si etichetele aferente batch-ului actual (pentru calculul acuratetii)
        current_predict = current_predict.cpu()
        batch_labels = batch_labels.cpu()
        current_predict = np.argmax(current_predict.detach().numpy(), axis=1)
        predictii = np.concatenate((predictii,current_predict))
        etichete = np.concatenate((etichete,batch_labels))
        
        # Antrenarea propriu-zisa
        
            # 1. Se sterg toti gradientii calculati anteriori, pentru toate variabilele antrenabile
            # deoarece, metoda <backward> acumuleaza noile valori, in loc sa le inlocuiasca.
        optimizer.zero_grad()
            # 2. Calculul tuturor gradientilor. Backpropagation
        loss.backward()
            # 3. Actualizarea tuturor ponderilor, pe baza gradientilor.
        optimizer.step()
        
        # Get the training accuracy
    running_corects += sum(predictii == etichete)
    acc_train = running_corects/ len(predictii)
    train_acc.append(acc_train)
    print( 'Acuratetea pe setul de antrenare la epoca {} este {}%'.format(ep+1,acc_train*100) )
    
    # Check the performance on the test set
    acc_test = testare()
    test_acc.append(acc_test)
    model.train()
          
    # Save the model after each epoch
    torch.save(model.state_dict(), f'checkpoint_epoch{ep}.pth')  



# Plot the accuracy
fig = plt.figure(4)
plt.plot(train_acc, label='train')
plt.plot(test_acc, label='test')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


acc_test = testare()


