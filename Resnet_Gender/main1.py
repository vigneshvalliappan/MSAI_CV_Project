import pandas as pd
import numpy as np
import cv2
from matplotlib import pyplot as plt
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from random import randint
import time
from net import *


fold_0 = pd.read_csv('AdienceBenchmarkGenderAndAgeClassification/fold_0_data.txt',sep='\t')
fold_1 = pd.read_csv('AdienceBenchmarkGenderAndAgeClassification/fold_1_data.txt',sep='\t')
fold_2 = pd.read_csv('AdienceBenchmarkGenderAndAgeClassification/fold_2_data.txt',sep='\t')
fold_3 = pd.read_csv('AdienceBenchmarkGenderAndAgeClassification/fold_3_data.txt',sep='\t')
fold_4 = pd.read_csv('AdienceBenchmarkGenderAndAgeClassification/fold_4_data.txt',sep='\t')
df_combined = pd.concat([fold_0,fold_1,fold_2,fold_3,fold_4],ignore_index=True)
print(f'Number of rows: {len(df_combined)}')
df_combined.info()
df_combined.head()

# new column to specify the file path
df_combined['filepath'] = 'AdienceBenchmarkGenderAndAgeClassification/faces/' + df_combined['user_id'] + '/coarse_tilt_aligned_face.' + df_combined['face_id'].astype('str') + '.' + df_combined['original_image']
df_combined['filepath']

# filtered of unlabelled data
print(f'Total Number of records: {len(df_combined)}')
df_gender = df_combined.copy()
df_gender = df_gender[df_gender['gender'] != 'u']
df_gender = df_gender[df_gender['gender'].notna()]
print(f'Total Number of records with Gender Labels: {len(df_gender)}')

# gender class
print('Gender Class (before labelling): ')
print(df_gender.gender.value_counts())

# convert class to int
def labelling_gender(gender_range):
    if gender_range == 'm':
        result = 0
    elif gender_range == 'f':
        result = 1
    else:
        print(gender_range)
    return result

df_gender['label_gender'] = df_gender['gender'].apply(labelling_gender)
print('Gender Class (after labelling): ')
print(df_gender.label_gender.value_counts())

df_gender.reset_index(inplace=True)
df_gender.drop(columns=['index'],inplace=True)


import torchvision
#from torchvision.io import read_image
from torch.utils.data import Dataset
from skimage import io
from PIL import Image

class FaceDataset(Dataset):
    def __init__(self, df, model='gender', transform=None):
        self.df = df
        self.model = model
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self,idx):
        img_path = self.df['filepath'][idx]
        image = Image.open(img_path)
        if self.model == 'gender':
            label = self.df['label_gender'][idx]
        elif self.model == 'age':
            label = self.df['label_age'][idx]
        else:
            print('Please specify "gender" or "age".')
            label = None
        if self.transform:
            image = self.transform(image)
        return (image, label)


transform = transforms.Compose(
    [
    transforms.Resize(230),
    transforms.RandomResizedCrop((224),scale=(0.6,1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    ]
)



dataset_gender = FaceDataset(df_gender,model='gender',transform=transform)
dataset_gender_len = len(dataset_gender)
print(dataset_gender_len)
print([int(dataset_gender_len*0.7),dataset_gender_len - int(len(dataset_gender)*0.7)])
train_gender, test_gender = torch.utils.data.random_split(dataset_gender,[int(dataset_gender_len*0.7),dataset_gender_len - int(len(dataset_gender)*0.7)])

from torch.utils.data import DataLoader

train_gender_loader = DataLoader(train_gender,batch_size=32,shuffle=True)
test_gender_loader = DataLoader(test_gender,batch_size=32,shuffle=True)



#train_features, train_labels = next(iter(train_gender_loader))
#print(f"Feature batch shape: {train_features.size()}")
#print(f"Labels batch shape: {train_labels.size()}")
#img = train_features[0].squeeze()

net_gender = ResNet(ResidualBlock, [2, 2, 2, 2])

criterion = nn.CrossEntropyLoss()
my_lr=0.001 
bs= 32
device= torch.device("cuda:2")
net_gender = net_gender.to(device)


def get_error( scores , labels ):

    bs=scores.size(0)
    predicted_labels = scores.argmax(dim=1)
    indicator = (predicted_labels == labels)
    num_matches=indicator.sum()
    
    return 1-num_matches.float()/bs    
    
def eval_on_test_set():

    running_error=0
    num_batches=0

    for i in range(0,len(test_gender),bs):

        minibatch_data, minibatch_label=  next(iter(test_gender_loader))

        minibatch_data=minibatch_data.to(device)
        minibatch_label=minibatch_label.to(device)
        
        inputs = minibatch_data

        scores=net_gender( inputs ) 

        error = get_error( scores , minibatch_label)

        running_error += error.item()

        num_batches+=1

    total_error = running_error/num_batches
    print( 'error rate on test set =', total_error*100 ,'percent')
    
    
start=time.time()

for epoch in range(1,20):
    net_gender.train()
    # divide the learning rate by 2 at epoch 10, 14 and 18
    if epoch==10 or epoch == 14 or epoch==18:
        my_lr = my_lr / 2
    
    # create a new optimizer at the beginning of each epoch: give the current learning rate.   
    optimizer=torch.optim.Adam( net_gender.parameters() , lr=my_lr )
        
    # set the running quatities to zero at the beginning of the epoch
    running_loss=0
    running_error=0
    num_batches=0
 
    for count in range(0,len(train_gender),bs):
    
        # Set the gradients to zeros
        optimizer.zero_grad()
        
        # create a minibatch       
        minibatch_data, minibatch_label = next(iter(train_gender_loader))

        minibatch_data=minibatch_data.to(device)
        minibatch_label=minibatch_label.to(device)
        
        # normalize the minibatch (this is the only difference compared to before!)
        inputs = minibatch_data
        
        # tell Pytorch to start tracking all operations that will be done on "inputs"
        inputs.requires_grad_()

        # forward the minibatch through the net 
        scores=net_gender( inputs ) 

        # Compute the average of the losses of the data points in the minibatch
        loss =  criterion( scores , minibatch_label) 
        
                
        # backward pass to compute dL/dU, dL/dV and dL/dW   
        loss.backward()

        # do one step of stochastic gradient descent: U=U-lr(dL/dU), V=V-lr(dL/dU), ...
        optimizer.step()
        
        # START COMPUTING STATS
        
        # add the loss of this batch to the running loss
        running_loss += loss.detach().item()
        
        # compute the error made on this batch and add it to the running error       
        error = get_error( scores.detach() , minibatch_label)
        running_error += error.item()
        
        num_batches+=1
        # print(f'{num_batches} : {(time.time()-start)/60}')
        
        
        print('[Epoch {0}] num_batch [{1}]  '
                'loss: {loss:.4f}  '
                'error: {error:.4f}  '
                .format(
                epoch, num_batches,
                loss = loss.item(),
                error = error.item(),
                ))
                
                
                
    
    # compute stats for the full training set
    total_loss = running_loss/num_batches
    total_error = running_error/num_batches
    elapsed = (time.time()-start)/60

    print('epoch=',epoch, '\t time=', elapsed,'min','\t lr=', my_lr  ,'\t loss=', total_loss , '\t error=', total_error*100 ,'percent')
    torch.save(net_gender,'net_gender.pth')
    net_gender.eval()
    eval_on_test_set() 
    net_gender.train()
    print(' ')
    


