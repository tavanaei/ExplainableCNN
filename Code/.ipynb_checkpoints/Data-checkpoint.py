from torch.utils import data
import torchvision.transforms as tr
import glob
import numpy as np
import random
import pickle
import torch
import os
import pandas as pd
from PIL import Image
import matplotlib.image as img

class ImageNetLoad(data.Dataset):
    """
    Pytorch Data Loader
    """
    def __init__(self,root,word,_transforms=None,train=True):
        random.seed(1)
        if root[-1]!='/':
            root +='/'
        self.files=[]
        self.labels = []
        self.box = []
        self.tr = _transforms
        
        root+='train/'
        docs = os.listdir(root)
        #docs.remove('.DS_Store')
        for doc in docs:
            tmp = glob.glob(root+doc+'/images/*.jpg')
            self.files+= tmp.copy()
            self.labels+= [word[doc]]*len(tmp)
            del tmp
        c = list(zip(self.files, self.labels))
        random.shuffle(c)
        self.files, self.labels = zip(*c)
        n = len(self.labels)
        if train:
            self.files = self.files[:int(0.8*n)]
            self.labels = self.labels[:int(0.8*n)]
        else:
            self.files = self.files[int(0.8*n):]
            self.labels = self.labels[int(0.8*n):]
            
        
            
    def __getitem__(self,index):
        ind = index%len(self.labels)
        tmp = Image.open(self.files[ind])
        data = self.tr(tmp)
        label = self.labels[ind]
        box = []
        if len(self.box)>1:
            box = self.box[ind]
        return data,label,box
    
    def __len__(self):
        return len(self.files)

def ImageNetLoad2(root,word,train=True):
    """
    Keras Data Loader
    """
    random.seed(1)
    if root[-1]!='/':
        root +='/'
    files=[]
    labels = []   
    root+='train/'
    docs = os.listdir(root)
    for doc in docs:
        tmp = glob.glob(root+doc+'/images/*.jpg')
        files+= tmp.copy()
        labels+= [word[doc]]*len(tmp)
        del tmp
    c = list(zip(files, labels))
    random.shuffle(c)
    files, labels = zip(*c)
    n = len(labels)
    data = []
    for i in range(len(files)):
        data.append(img.imread(files[i]))
    data = np.stack(data)
    labels = np.array(labels)
    print(data.shape)
    print(labels.shape)
    
    return data.astype('float32'),labels.astype('float32')

class ImageNetTest():
    """
    Data Loader for Pytorch and Keras
    """
    def __init__(self,root):
        if root[-1]!='/':
            root+='/'
        f = open(root+'class.dat','rb')
        word = pickle.load(f)

        root+='val/'
        self.files= glob.glob(root+'images/*.jpg')
        self.labels = []
        self.box = []
        df = pd.read_table(root+'val_annotations.txt',header=None,names=['image','class','w1','h1','w2','h2'])
        for i in range(len(self.files)):
            im_name = os.path.basename(self.files[i]).split('.')[0]+'.JPEG'
            row = df[df['image']==im_name]
            self.labels.append(word[row['class'].item()])
            self.box.append((int(row['h1'].item()),int(row['w1'].item()),int(row['h2'].item()),int(row['w2'].item())))
    def torchLoad(self,maxi = 100):
        trans = tr.Compose([tr.ToTensor(),tr.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),])
        data = []
        maxi = min(maxi,len(self.files))
        for i in range(maxi):
            data.append(trans(Image.open(self.files[i])))
        data = torch.stack(data)
        return data,np.array(self.labels)[:maxi],self.files[:maxi]
    def kerasLoad(self,maxi=100):
        data = []
        maxi = min(maxi,len(self.files))
        for i in range(maxi):
            data.append(img.imread(self.files[i]))
        data = np.stack(data)
        return data,np.array(self.labels)[:maxi],self.files[:maxi]

    
    