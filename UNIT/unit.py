import argparse
import os
import numpy as np
import math
import itertools
import datetime
import time
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from models import *
from datasets import *

import torch.nn as nn
import torch.nn.functional as F
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--start_epoch', type=int, default=0, help='epoch to start training from')
parser.add_argument('--total_epoches', type=int, default=200, help='number of epochs of training')
parser.add_argument('--dataset_name', type=str, default="2138_cycle", help='name of the dataset')
parser.add_argument('--batchsize', type=int, default=1, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0001, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--decay_epoch', type=int, default=100, help='epoch from which to start lr decay')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--img_height', type=int, default=256, help='size of image height')
parser.add_argument('--img_width', type=int, default=256, help='size of image width')
parser.add_argument('--channels', type=int, default=3, help='number of image channels')
parser.add_argument('--sample_interval', type=int, default=100, help='interval between sampling images from generators')
parser.add_argument('--checkpoint_interval', type=int, default=-1, help='interval between saving model checkpoints')
parser.add_argument('--n_downsample', type=int, default=2, help='number downsampling layers in encoder')
parser.add_argument('--dim', type=int, default=64, help='number of filters in first encoder layer')
opt = parser.parse_args()
print(opt)

cuda = True if torch.cuda.is_available() else False

# Create sample and checkpoint directories
os.makedirs('images/%s' % opt.dataset_name, exist_ok=True)
os.makedirs('saved_models/%s' % opt.dataset_name, exist_ok=True)

# Losses
criterion_GAN = torch.nn.MSELoss()
criterion_pixel = torch.nn.L1Loss()

# Calculate output of image discriminator (PatchGAN)
patch = (1, opt.img_height // 2**4, opt.img_width // 2**4)

# Dimensionality (channel-wise) of image embedding
shared_dim = opt.dim * 2**opt.n_downsample


shared_E = ResidualBlock(features = shared_dim)
E_a = E(dim=opt.dim, n_downsample=opt.n_downsample, shared_block=shared_E)
E_b = E(dim=opt.dim, n_downsample=opt.n_downsample, shared_block=shared_E)

shared_G = ResidualBlock(features=shared_dim)
G_a2b = G(dim = opt.dim, n_upsample= opt.n_downsample, shared_block=shared_G)
G_b2a= G(dim = opt.dim, n_upsample= opt.n_downsample, shared_block=shared_G)


D_a=D()
D_b=D()


if cuda:
    E_a = E_a.cuda()
    E_b = E_b.cuda()
    G_a2b = G_a2b.cuda()
    G_b2a = G_b2a.cuda()
    D_a = D_a.cuda()
    D_b = D_b.cuda()
    criterion_GAN.cuda()
    criterion_pixel.cuda()

if opt.start_epoch != 0:
    # Load pretrained models
    E_a.load_state_dict(torch.load('saved_models/%s/E_a_%d.pth' % (opt.dataset_name, opt.epoch)))
    E_b.load_state_dict(torch.load('saved_models/%s/E_b_%d.pth' % (opt.dataset_name, opt.epoch)))
    G_a2b.load_state_dict(torch.load('saved_models/%s/G_a2b_%d.pth' % (opt.dataset_name, opt.epoch)))
    G_b2a.load_state_dict(torch.load('saved_models/%s/G_b2a_%d.pth' % (opt.dataset_name, opt.epoch)))
    D_a.load_state_dict(torch.load('saved_models/%s/D_a_%d.pth' % (opt.dataset_name, opt.epoch)))
    D_b.load_state_dict(torch.load('saved_models/%s/D_b_%d.pth' % (opt.dataset_name, opt.epoch)))
else:
    # Initialize weights
    E_a.apply(weights_init_normal)
    E_b.apply(weights_init_normal)
    G_a2b.apply(weights_init_normal)
    G_b2a.apply(weights_init_normal)
    D_a.apply(weights_init_normal)
    D_b.apply(weights_init_normal)

# Loss weights
lambda_0 = 10   # GAN
lambda_1 = 0.1  # KL (encoded images)
lambda_2 = 100  # ID pixel-wise
lambda_3 = 0.1  # KL (encoded translated images)
lambda_4 = 100  # Cycle pixel-wise

# Optimizers
optimizer_G = torch.optim.Adam(itertools.chain(E_a.parameters(), E_b.parameters(), G_a2b.parameters(), G_b2a.parameters()),
                                lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_a = torch.optim.Adam(D_a.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_b = torch.optim.Adam(D_b.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# Learning rate update schedulers
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opt.total_epoches, opt.start_epoch, opt.decay_epoch).step)
lr_scheduler_D_a = torch.optim.lr_scheduler.LambdaLR(optimizer_D_a, lr_lambda=LambdaLR(opt.total_epoches, opt.start_epoch, opt.decay_epoch).step)
lr_scheduler_D_b = torch.optim.lr_scheduler.LambdaLR(optimizer_D_b, lr_lambda=LambdaLR(opt.total_epoches, opt.start_epoch, opt.decay_epoch).step)

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

# Image transformations
transforms_ = [ transforms.Resize(int(opt.img_height*1.12), Image.BICUBIC),
                transforms.RandomCrop((opt.img_height, opt.img_width)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]

# Training data loader
dataloader = DataLoader(ImageDataset("../../data/%s" % opt.dataset_name, transforms_=transforms_, unaligned=True),
                        batch_size=opt.batchsize, shuffle=True, num_workers=opt.n_cpu)
# Test data loader
val_dataloader = DataLoader(ImageDataset("../../data/%s" % opt.dataset_name, transforms_=transforms_, unaligned=True, mode='test'),
                        batch_size=5, shuffle=True, num_workers=1)

def sample_images(now_batch):
    
    images=next(iter(val_dataloader))
    A=Variable(images['A'].type(Tensor))
    B=Variable(images['B'].type(Tensor))
    _, z_A=E_a(A)
    _, z_B=E_b(B)
    fake_A = G_b2a(z_B)
    fake_B = G_a2b(z_A)
    images_sample = torch.cat((A.data, fake_B.data,
                               B.data, fake_A.data), 0 )
    save_image(images_sample, 'images/%s/%s.png'%(opt.dataset_name, now_batch), nrow=5, normalize=True)
    

def compute_k1(mu):
    mu_2 = torch.pow(mu, 2)
    loss=torch.mean(mu_2)
    return loss

prev_time=time.time()
for epoch in range(opt.start_epoch, opt.total_epoches):
    for i,batch in enumerate(dataloader):
        
        real_A= Variable(batch['A'].type(Tensor))
        real_B= Variable(batch['B'].type(Tensor))
        
        real_target = Variable(Tensor(np.ones((real_A.size(0), *patch))), requires_grad=False)
        fake_target = Variable(Tensor(np.zeros((real_A.size(0), *patch))), requires_grad=False)
     
        # -------------------------------
        #  Train Encoders and Generators
        # -------------------------------
        optimizer_G.zero_grad()
        
        mu_A, z_A = E_a(real_A)
        mu_B, z_B = E_b(real_B)
        
        recon_A = G_b2a(z_A)
        recon_B = G_a2b(z_B)
        
        fake_A = G_b2a(z_B)
        fake_B = G_a2b(z_A)
        
        
        mu_fake_A, z_fake_A = E_a(fake_A)
        mu_fake_B, z_fake_B = E_b(fake_B)
        
        cycle_A = G_b2a(z_fake_B)
        cycle_B = G_a2b(z_fake_A)
        
        loss_GAN_1 = lambda_0 * criterion_GAN(D_a(fake_A), real_target)
        loss_GAN_2 = lambda_0 * criterion_GAN(D_b(fake_B), real_target)
        
        loss_KL_1 = lambda_1 * compute_k1(mu_A)
        loss_KL_2 = lambda_1 * compute_k1(mu_B)
        
        loss_ID_1 = lambda_2 * criterion_pixel(recon_A, real_A)
        loss_ID_2 = lambda_2 * criterion_pixel(recon_B, real_B)
        
        loss_KL_1_ = lambda_3* compute_k1(mu_fake_A)
        loss_KL_2_ = lambda_3 * compute_k1(mu_fake_B)
        
        loss_cyc_1 = lambda_4 * criterion_pixel(cycle_A, real_A)
        loss_cyc_2 = lambda_4 * criterion_pixel(cycle_B, real_B)

        # Total loss
        loss_G =    loss_KL_1 + \
                    loss_KL_2 + \
                    loss_ID_1 + \
                    loss_ID_2 + \
                    loss_GAN_1 + \
                    loss_GAN_2 + \
                    loss_KL_1_ + \
                    loss_KL_2_ + \
                    loss_cyc_1 + \
                    loss_cyc_2

        loss_G.backward()
        optimizer_G.step()

        # -----------------------
        #  Train D_a
        # -----------------------

        optimizer_D_a.zero_grad()
        loss_D_a = criterion_GAN(D_a(real_A), real_target) + \
                   criterion_GAN(D_a(fake_A.detach()), fake_target)    
   
        
        loss_D_a.backward()
        optimizer_D_a.step()

        # -----------------------
        #  Train D_b
        # -----------------------
        optimizer_D_b.zero_grad()
        
        loss_D_b = criterion_GAN(D_b(real_B), real_target) + \
                   criterion_GAN(D_b(fake_B.detach()), fake_target)
        loss_D_b.backward()
        optimizer_D_b.step()

        # ================================================
        now_batch = epoch * len(dataloader) + i
        batches_left = opt.total_epoches * len(dataloader) - now_batch
        time_left = datetime.timedelta(seconds = batches_left*(time.time() - prev_time))
        prev_time = time.time()
        
        print('[epoch %d/%d] [batch %d/%d] [D loss: %f] [G loss %f] [time_left: %s]'%
                (epoch, opt.total_epoches, i, len(dataloader), (loss_D_a+loss_D_b).item(), loss_G.item(), time_left))
                
        if now_batch % opt.sample_interval ==0:
            sample_images(now_batch)
              

    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_a.step()
    lr_scheduler_D_b.step()

    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(E_a.state_dict(), 'saved_models/%s/E_a_%d.pth' % (opt.dataset_name, epoch))
        torch.save(E_b.state_dict(), 'saved_models/%s/E_b_%d.pth' % (opt.dataset_name, epoch))
        torch.save(G_a2b.state_dict(), 'saved_models/%s/G_a2b_%d.pth' % (opt.dataset_name, epoch))
        torch.save(G_b2a.state_dict(), 'saved_models/%s/G_b2a_%d.pth' % (opt.dataset_name, epoch))
        torch.save(D_a.state_dict(), 'saved_models/%s/D_a_%d.pth' % (opt.dataset_name, epoch))
        torch.save(D_b.state_dict(), 'saved_models/%s/D_b_%d.pth' % (opt.dataset_name, epoch))
