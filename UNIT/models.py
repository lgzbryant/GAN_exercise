import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable    
import numpy as np

def weights_init_normal(m):
    classname = m.__class__.__name__
    print(classname)
    if classname.find('Conv')!=-1:
        torch.nn.init.normal_(m.weight.data,0.0, 0.02)
    elif classname.find('BatchNorm2d')!=-1:
        torch.nn.init.normal_(m.weight.data,1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
class LambdaLR():
    def __init__(self, total_epoches, offset, decay_start_epoch):
        assert((total_epoches - decay_start_epoch)>0), 'Decay must start before the training session ends'
        self.total_epoches = total_epoches
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch
        
    def step(self, epoch):
        return 1.0 - max(0, epoch+self.offset - self.decay_start_epoch)/(self.total_epoches - self.decay_start_epoch)

    
class ResidualBlock(nn.Module):
    def __init__(self, features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(features, features, 3),
                        nn.InstanceNorm2d(features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(features, features, 3),
                        nn.InstanceNorm2d(features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


 
class E(nn.Module):
    def __init__(self, in_channels=3, dim=64, n_downsample=2, shared_block=None):
        super(E, self).__init__()
        
        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, dim, 7),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        
        for _ in range(n_downsample):
          layers += [
              nn.Conv2d(dim, dim*2, 4, stride=2, padding=1),
              nn.InstanceNorm2d(dim*2),
              nn.ReLU(inplace=True)
          ]
          dim*=2
        for _ in range(3):
            layers += [ResidualBlock(dim)]
        self.model_blocks = nn.Sequential(*layers)
        self.shared_block = shared_block
    def reparameterization(self, mu):
        Tensor = torch.cuda.FloatTensor if mu.is_cuda else torch.FloatTensor
        z=Variable(Tensor(np.random.normal(0, 1, mu.shape)))
        return z+mu
        
    def forward(self, x):
        x = self.model_blocks(x)
        mu= self.shared_block(x)
        z=self.reparameterization(mu)
        return mu ,z
             
class G(nn.Module):
    def __init__(self, out_channels=3, dim=64, n_upsample=2, shared_block=None):
        super(G, self).__init__()
        
        self.shared_block=shared_block
        
        layers=[]
        dim=dim*2**n_upsample
        
        for _ in range(3):
            layers+=[ResidualBlock(dim)]
            
        for _ in range(n_upsample):
            layers+=[
                nn.ConvTranspose2d(dim, dim//2, 4, stride=2, padding=1),
                nn.InstanceNorm2d(dim//2),
                nn.LeakyReLU(0.2, inplace=True)
            ]
            dim//=2
        
        layers+=[
            nn.ReflectionPad2d(3),
            nn.Conv2d(dim, out_channels, 7),
            nn.Tanh()
        ]
        self.model_blocks = nn.Sequential(*layers)
        
        
    def forward(self, x):
        x=self.shared_block(x)
        x=self.model_blocks(x)
        
        return x

class D(nn.Module):
    def __init__(self, in_channels=3):
        super(D, self).__init__()
        
        def  d(in_channels, out_channels, normlize=True):
            layers=[
                nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1)
            ]
            if normlize:
                layers.append(nn.InstanceNorm2d(out_channels))
                
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
            
        self.model = nn.Sequential(
            *d(in_channels, 64, normlize=False),
            *d(64, 128),
            *d(128, 256),
            *d(256, 512),
            nn.Conv2d(512, 1,3, padding=1))
            
    def forward(self, x):
        return self.model(x)