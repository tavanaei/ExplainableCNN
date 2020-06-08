import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
import pickle
import os
import matplotlib.image as img
from PIL import Image
import torchvision

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
def cifar():
    path = 'TestImages/cifar/'
    model_name = 'Models/Model_Cifar_128.net'
    
    files = os.listdir(path)
    data = []
    for im_name in files:
        if im_name.endswith('.jpg'):
            data.append(img.imread(path+im_name))
    ims = np.stack(data)
    del data
    
    net = torch.load(model_name).to(device)
    transform_test = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
    batch = torch.zeros(ims.shape[0],ims.shape[3],ims.shape[1],ims.shape[2])
    for i in range(len(ims)):
        batch[i] = transform_test(ims[i])
    batch = batch.to(device)  
    output = net(batch)
    maps = net.maps.cpu().detach().numpy()
    heatmap,thrmap = vis(maps,ims)
    return ims,heatmap,thrmap

def tinyimagenet():
    path = 'TestImages/tinyimagenet/'
    model_name = 'Models/Model_TinyImageNet_128.net'
    net = torch.load(model_name).to(device)
    transform_test = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])
    
    files = os.listdir(path)
    data = []
    for im_name in files:
        if im_name.endswith('.jpg'):
            data.append(transform_test(Image.open(path+im_name)))
    ims = torch.stack(data).to(device)
    del data
    output = net(ims)
    maps = net.maps.cpu().detach().numpy()
    data = np.transpose(ims.detach().cpu().numpy()*0.5+0.5,(0,2,3,1))
    heatmap,thrmap = vis(maps,data)
    return data,heatmap,thrmap

def mnist():
    path = 'TestImages/mnist/samples.pk'
    model_name = 'Models/Model_MNIST_32.net'
    f = open(path,'rb')
    data = pickle.load(f)/127.5-1.0
    net = torch.load(model_name).to(device)
    output = net(data.to(device))
    maps = net.maps.cpu().detach().numpy()
    data = data.detach().cpu().numpy()*0.5+0.5
    heatmap,thrmap = vis(maps*(-1),data)
    return data.squeeze(),heatmap,thrmap

def vis(maps,data):
    kernel=np.ones([3,3],dtype=np.uint8)
    heatmaps = []
    thrmaps = []
    for i in range(min(len(maps),100)):
        if len(maps[i].shape)>2:
            tmp = np.transpose(maps[i],(1,2,0)).squeeze()
        else:
            tmp = maps[i].squeeze()
        tmp=(tmp-tmp.min())/(tmp.max()-tmp.min())
        tmp = 1-(tmp)
        heatmaps.append(tmp.copy())
        
        ## Thresholding for Localization (Future work)
        if maps[i].shape[0]>1:
            tmp2 = tmp.copy()
            tmp2[tmp2>0.5]=1
            tmp2[tmp2<=0.5]=0
            tmp2 = cv2.dilate(tmp2,kernel,iterations=1)
            tmp2 = cv2.erode(tmp2,kernel,iterations=1)
            tmp2 = tmp2*255
            gray = tmp2.astype('uint8')
            ret,thresh1 = cv2.threshold(gray,100,255,cv2.THRESH_BINARY)
            contours,hierarchy = cv2.findContours(thresh1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            im = cv2.cvtColor(data[i], cv2.COLOR_RGB2BGR)
            box = None
            c = 0
            for ind, cont in enumerate(contours):
                if cont.shape[0]>c:
                    c = cont.shape[0]
                    box = cont.copy()

            elps = cv2.boundingRect(box)
            cv2.rectangle(im,elps,(0,0,0),1)
            thrmaps.append(im[:,:,::-1])
    return heatmaps,thrmaps


