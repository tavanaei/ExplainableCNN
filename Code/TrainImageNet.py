import argparse
from XAI import *
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torch.utils.data
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from Data import ImageNetLoad as iml
import pickle


parser = argparse.ArgumentParser()
parser.add_argument('-epoch', type=int, default=100, help='training epoch')
parser.add_argument('-lr', type=float, default=0.001, help='learning rate')
parser.add_argument('-wd', type=float, default=0.000001,help='weight decay')
parser.add_argument('-beta', type=float, default=0.9, help='momentum')
parser.add_argument('-cuda', action='store_true', help='Cuda operation')
parser.add_argument('-batch', type=int, default=128, help='batch_size')
parser.add_argument('-lr_decay', type=float, default=.99, help='lr decay')
parser.add_argument('-gpu', type=int, default=1, help='GPU device')

opt = parser.parse_args()
print(opt)
size = 64
path = 'tinyimagenet_data/'
f = open(path+'class.dat','rb')
categories = pickle.load(f)

transform_train = transforms.Compose([
    transforms.RandomCrop(size, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
])

trainset = iml(path,categories,_transforms=transform_train,train=True)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=opt.batch, shuffle=True, num_workers=4)

testset = iml(path,categories,_transforms=transform_test,train=False)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=opt.batch, shuffle=False, num_workers=4)


g_kernels = [3,128]
d_kernels =  [1,64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512] 
g_strides = [2,2,2]
d_strides = [1,2,1,2,1,1,2,1,1,2,1,1,2]
g_patch = 3
d_patch = 3
fc_layers = [2048,200]

net = Model(G_kernels=g_kernels,D_kernels=d_kernels,G_strides=g_strides,D_strides=d_strides,G_patch=g_patch,D_patch=d_patch,fcc_layers=fc_layers,G_pool='average',D_pool='max',dropout=0.3,prob=False)

print(net)
device = torch.device('cuda:'+str(opt.gpu) if opt.cuda else 'cpu')
net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=opt.lr, weight_decay=opt.wd,betas=(opt.beta,opt.beta+0.09))
decay_step = opt.epoch//2
def calculate_accuracy(loader):
    correct = 0.
    total = 0.
    for data in loader:
        images, labels,box = data
        images = images.to(device)
        labels = labels.to(device)
        outputs = net(Variable(images))
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum()

    return 100.0 * correct / total
loss_track = []
acc_track_tr = []
acc_track_ts = []
for epoch in range(opt.epoch):
    running_loss = 0
    for i, data in enumerate(trainloader, 0):
        inputs, labels,box = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        inputs, labels = Variable(inputs), Variable(labels)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()
        running_loss += loss.item()
    running_loss /= len(trainloader)
    
    train_accuracy = calculate_accuracy(trainloader)
    test_accuracy = calculate_accuracy(testloader)
    
    print('Iteration: {} | Loss: {} | Training accuracy: {} | Test accuracy: {}'.format(
        epoch+1, running_loss, train_accuracy, test_accuracy))
    loss_track.append(running_loss)
    acc_track_ts.append(test_accuracy)
    acc_track_tr.append(train_accuracy)
    if epoch>0 and epoch%decay_step==0:
        lr = opt.lr*(0.1**(epoch//decay_step))
        for param_group in optimizer.param_groups:
                param_group['lr'] = lr
model_name = 'Models/ImageNet_128'
torch.save(net.to(torch.device('cpu')),model_name+'.net')
torch.save({'loss':loss_track, 'acc_train':acc_track_tr, 'acc_test':acc_track_ts},model_name+'.dat')


