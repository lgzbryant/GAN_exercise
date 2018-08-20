##welcome kou_lan_zang_pig_HTT
import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch


folder=os.path.exists('images')
if not folder:
    os.makedirs('images')

parser = argparse.ArgumentParser()

parser.add_argument('--epoches', type=int, default=200)
parser.add_argument('--batchsize',type=int,default=64)
parser.add_argument('--lr',type=float,default=0.00002)
parser.add_argument('--b1',type=float,default=0.5)
parser.add_argument('--b2',type=float,default=0.999)
parser.add_argument('--cpus',type=int,default=8)
parser.add_argument('--latent_dim',type=int,default=128)
parser.add_argument('--n_classes',type=int,default=10)
parser.add_argument('--img_size',type=int,default=28)
parser.add_argument('--channels',type=int,default=1)
parser.add_argument('--sample_interval',type=int,default=500)
opt=parser.parse_args()
print(opt)


folder=os.path.exists('data/mnist')
if not folder:
    os.makedirs('data/mnist')

dataloader = torch.utils.data.DataLoader(
    datasets.MNIST('data/mnist', train=True, download=True,
                   transform=transforms.Compose([
                        transforms.Resize(opt.img_size),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                   ])),
    batch_size=opt.batchsize, shuffle=True)


img_shape = (opt.channels, opt.img_size, opt.img_size)

class G(nn.Module):
    def __init__(self):
        super(G,self).__init__()
        self.label_emb=nn.Embedding(opt.n_classes,opt.n_classes)
        self.model=nn.Sequential(
            #batchsize*(100+10)------------>batchsize*256
            nn.Linear(opt.latent_dim+opt.n_classes, 256),
            nn.BatchNorm1d(256,0.8),                   
            nn.LeakyReLU(0.2,inplace=True),
            
            nn.Linear(256, 512),
            nn.BatchNorm1d(512,0.8),
            nn.LeakyReLU(0.2,inplace=True),                  
        
            nn.Linear(512, int(np.prod(img_shape))),
            nn.Tanh()                      
        )
    def forward(self,z,labels):
        #64*10 cat 64*100--------------->64*110
        g_input=torch.cat([self.label_emb(labels),z],-1)
        img=self.model(g_input)
        #batchsize*(img_shape)
        img=img.view(img.size(0),*(1,opt.img_size,opt.img_size))       
        return img
class D(nn.Module):
    def __init__(self):
        super(D,self).__init__()
        #
        self.label_emb=nn.Embedding(opt.n_classes,opt.n_classes)
        
        self.model=nn.Sequential(
            nn.Linear(opt.n_classes + int(np.prod(img_shape)), 512),
            
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(512, 512),
            nn.Dropout(0.4),           
            nn.LeakyReLU(0.2, inplace=True),
                      
            nn.Linear(512, 1)    
        )
    def forward(self,img,labels):
        #batchsize*(img_shape)  cat batchsize*10------------>
        d_input=torch.cat((img.view(img.size(0),-1),self.label_emb(labels)),-1)
        res=self.model(d_input)     
        return res
        
gan_loss=torch.nn.MSELoss()

g=G()
d=D() 

if  torch.cuda.is_available():
    g.cuda()
    d.cuda()
    gan_loss.cuda()
                     
optimizer_g=torch.optim.Adam(g.parameters(),lr=opt.lr,betas=(opt.b1,opt.b2))
optimizer_d=torch.optim.Adam(d.parameters(),lr=opt.lr,betas=(opt.b1,opt.b2))

def save_sample(row_col,now_batch):
    #save (row_col*row_col) digits
    z=Variable(torch.cuda.FloatTensor(np.random.normal(0,1,(row_col**2,opt.latent_dim))))
    list=range(row_col)
    labels=np.array([num for i in list for num in list])
    labels=Variable(torch.cuda.LongTensor(labels))
    
    g_imgs=g(z,labels)
    save_image(g_imgs.data,'images/%d.png'%now_batch,nrow=row_col,normalize=True)


for epoch in range(opt.epoches):
    for i,(imgs,labels) in enumerate(dataloader):   
        #what a prit!!   Not  batchsize=opt.batchsize
        
        #batchsize=opt.batchsize
        batchsize=imgs.shape[0]
                
        # print('==================')
        # if imgs.size(0)==32:
            # print('*'*200000) 
            
        # print(imgs.size())        
        # print('==================')
        
        real=Variable(torch.cuda.FloatTensor(batchsize,1).fill_(1.0),requires_grad=False)
        fake=Variable(torch.cuda.FloatTensor(batchsize,1).fill_(0.0),requires_grad=False)
        
        real_imgs=Variable(imgs.type(torch.cuda.FloatTensor))
        labels=Variable(labels.type(torch.cuda.LongTensor))
        
        #--------------------------
        optimizer_g.zero_grad()
        
        z = Variable(torch.cuda.FloatTensor(np.random.normal(0, 1, (batchsize, opt.latent_dim))))
        g_labels = Variable(torch.cuda.LongTensor(np.random.randint(0, opt.n_classes, batchsize)))
        
        
        g_imgs=g(z,g_labels)
        g_predict=d(g_imgs,g_labels)
        
        g_loss=gan_loss(g_predict,real)
        
        g_loss.backward()
        optimizer_g.step()
        #---------------------------------------
        optimizer_d.zero_grad()
        
        real_predict=d(real_imgs,labels)
        real_loss=gan_loss(real_predict,real)
        
        fake_predict=d(g_imgs.detach(),g_labels)
        fake_loss=gan_loss(fake_predict,fake)
        
        d_loss=(real_loss+fake_loss)/2
        d_loss.backward()
        optimizer_d.step()
        
         
        print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" \
        % (epoch, opt.epoches, i, len(dataloader), d_loss.data[0], g_loss.data[0]))
                                                            

        now_batches = epoch * len(dataloader) + i
        if now_batches % opt.sample_interval == 0:
            save_sample(5, now_batches)
