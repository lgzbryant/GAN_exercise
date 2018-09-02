import argparse
import os 
import numpy as np
import math
import itertools

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

folder = os.path.exists('images')
if not folder:
    os.makedirs('images')    
    
parser = argparse.ArgumentParser()
parser.add_argument('--total_epoches', type=int, default=200)
parser.add_argument('--batchsize', type=int, default=64)
parser.add_argument('--lr', type=float, default=0.0002)
parser.add_argument('--b1', type=float, default=0.5)
parser.add_argument('--b2', type=float, default=0.999)
parser.add_argument('--cpus', type=int, default=8)
parser.add_argument('--latent_dim', type=int, default=10)
parser.add_argument('--img_size',type=int,default=32)
parser.add_argument('--channels', type=int, default=1)
parser.add_argument('--sample_interval', type=int, default=500)
opt=parser.parse_args()
print(opt)



img_shape = (opt.channels, opt.img_size, opt.img_size)
cuda=True if torch.cuda.is_available() else False 

def reparameterization(mu, logvar):
    std = torch.exp(logvar / 2)
    sample_z = Variable(Tensor(np.random.normal(0,1,(mu.size(0),opt.latent_dim))))
    z=sample_z*std + mu
    return z
 
class E(nn.Module):
    def __init__(self):
        super(E, self).__init__()
        
        self.model=nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512,512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True)      
        )
        self.mu=nn.Linear(512, opt.latent_dim)
        self.logvar=nn.Linear(512, opt.latent_dim)
        
    def forward(self, img):
        img_flat= img.view(img.shape[0], -1)
        x=self.model(img_flat)
        mu=self.mu(x)
        logvar=self.logvar(x)
        z=reparameterization(mu, logvar)
        return z

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(opt.latent_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img_flat = self.model(z)
        img = img_flat.view(img_flat.shape[0], *img_shape)
        return img
       
        
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.model=nn.Sequential(
            nn.Linear(opt.latent_dim, 512),
            nn.LeakyReLU(0.2, inplace= True),
            nn.Linear(512,256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256,1),
            nn.Sigmoid()
            
        )
    def forward(self, z):
        predict=self.model(z)
        return predict
           
gan_loss = torch.nn.BCELoss()
pix_loss = torch.nn.L1Loss()

encoder=E()
decoder=Decoder()
discriminator=Discriminator()

if cuda:
    encoder.cuda()
    decoder.cuda()
    discriminator.cuda()
    pix_loss.cuda()
    gan_loss.cuda()
    
folder = os.path.exists('../../data/mnist')
if not folder:  
    os.makedirs('../../data/mnist')
dataloader=torch.utils.data.DataLoader(
    datasets.MNIST('../../data/mnist',train=True, download=True,
        transform=transforms.Compose([          
                  transforms.Resize(opt.img_size),
                  transforms.ToTensor(),
                  transforms.Normalize((0.5, 0.5, 0.5), (0.5,0.5, 0.5))
                  ])),
        batch_size=opt.batchsize, shuffle=True
)
        

optimizer_G = torch.optim.Adam(itertools.chain(encoder.parameters(), decoder.parameters()),
                              lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor=torch.cuda.FloatTensor if cuda else torch.FloatTensor

def sample_image(rows, now_batch):
    z= Variable(Tensor(np.random.normal(0, 1, (rows**2, opt.latent_dim))))
    imgs=decoder(z)
    save_image(imgs.data, 'imgs/%d.jpg'%now_batch,nrow=rows, normalize=True)
    

for epoch in range(opt.total_epoches):
    for i, (imgs, _) in enumerate(dataloader):
    # for i (imgs, _) in enumerate(dataloader):
        
        real_target=Variable(Tensor(imgs.shape[0],1).fill_(1.0), requires_grad=False)
        fake_target=Variable(Tensor(imgs.shape[0],1).fill_(0.0), requires_grad=False)
        
        real_imgs=Variable(imgs.type(Tensor))
  
        # -----------------
        #  Train Generator
        # -----------------       
        optimizer_G.zero_grad()
        
        encoder_z=encoder(real_imgs)
        decoder_imgs=decoder(encoder_z)
        
        g_loss=gan_loss(discriminator(encoder_z), real_target)+10*pix_loss(decoder_imgs, real_imgs)
                
        g_loss.backward()
        
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()
        
        z=Variable(Tensor(np.random.normal(0,1, (imgs.shape[0], opt.latent_dim))))
        real_loss=gan_loss(discriminator(z), real_target)
        fake_loss=gan_loss(discriminator(encoder_z.detach()), fake_target)
        
        d_loss=0.5*(real_loss+fake_loss)
        d_loss.backward()
        
        optimizer_D.step()            
            
        print('Epoch %d/%d | batch %d/%d | D loss: %f | G loss: %f' %(epoch, opt.total_epoches, i, len(dataloader), d_loss.data[0], g_loss.data[0]))
            
        now_batch=epoch*len(dataloader)+i
        if now_batch % opt.sample_interval == 0:
            sample_image(rows=10, now_batch=now_batch)
        
