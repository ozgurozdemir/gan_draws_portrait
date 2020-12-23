# Src: https://github.com/nightldj/behance_release/blob/master/net_utils.py

import torch
import torchvision
import torch.nn as nn

import numpy as np
import time
import random


# Mask Module
class Mask(nn.Module):
    def __init__(self, in_channel, out_channel, unit=128, activation=None):
        super(Mask, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel 
        self.unit  = unit
        self.activation = activation
        
        self.block0 = nn.Sequential(
            nn.Conv2d(self.in_channel, self.unit, kernel_size=1, padding=0, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(self.unit),
            
            nn.ReflectionPad2d((1,1,1,1)),
            nn.Conv2d(self.unit, self.unit, kernel_size=3, padding=0, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(self.unit),
        )
        
        self.last = nn.Sequential(nn.Conv2d(self.unit, self.out_channel, kernel_size=1, padding=0)) 
        if self.activation == 'sigmoid':
          self.last.add_module("activation", nn.Sigmoid())
        elif self.activation == 'tanh':
          self.last.add_module("activation", nn.Tanh())

    def forward(self, x):
      out = self.block0(x)
      out = self.last(out)
      return out


# Decoder Module
class Decoder(nn.Module):
    def __init__(self, out_channel=3, units=64, activation='sigmoid'):
        super(Decoder, self).__init__()
        self.out_channel = out_channel 
        self.units  = units
        self.activation = activation
        
        # [512, 256, up]
        self.block0 = nn.Sequential(
            nn.ReflectionPad2d((1,1,1,1)),
            nn.Conv2d(self.units * 8, self.units * 8, kernel_size=3, padding=0, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(self.units * 8),
            
            nn.ReflectionPad2d((1,1,1,1)),
            nn.Conv2d(self.units * 8, self.units * 4, kernel_size=3, padding=0, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(self.units * 4),

            nn.ConvTranspose2d(self.units * 4, self.units * 4, kernel_size=4, stride=2, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # [256, 256, 256, 128, up]
        self.block1 = nn.Sequential(
            nn.ReflectionPad2d((1,1,1,1)),
            nn.Conv2d(self.units * 4, self.units * 4, kernel_size=3, padding=0, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(self.units * 4),

            nn.ReflectionPad2d((1,1,1,1)),
            nn.Conv2d(self.units * 4, self.units * 4, kernel_size=3, padding=0, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(self.units * 4),
            
            nn.ReflectionPad2d((1,1,1,1)),
            nn.Conv2d(self.units * 4, self.units * 4, kernel_size=3, padding=0, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(self.units * 4),

            nn.ReflectionPad2d((1,1,1,1)),
            nn.Conv2d(self.units * 4, self.units * 2, kernel_size=3, padding=0, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(self.units * 2),

            nn.ConvTranspose2d(self.units * 2, self.units * 2, kernel_size=4, stride=2, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
        ) 

        # [128, 64, up]
        self.block2 = nn.Sequential(
            nn.ReflectionPad2d((1,1,1,1)),
            nn.Conv2d(self.units * 2, self.units * 2, kernel_size=3, padding=0, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(self.units * 2),

            nn.ReflectionPad2d((1,1,1,1)),
            nn.Conv2d(self.units * 2, self.units, kernel_size=3, padding=0, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(self.units),
            
            nn.ConvTranspose2d(self.units, self.units, kernel_size=4, stride=2, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
        ) 
        # [64]
        self.block3 = nn.Sequential(
            nn.ReflectionPad2d((1,1,1,1)),
            nn.Conv2d(self.units, self.units, kernel_size=3, padding=0, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(self.units)
        )

        self.last = nn.Sequential(nn.Conv2d(self.units, self.out_channel, kernel_size=1, padding=0)) 
        if self.activation == 'sigmoid':
          self.last.add_module("activation", nn.Sigmoid())
        elif self.activation == 'tanh':
          self.last.add_module("activation", nn.Tanh())

    def forward(self, x):
      out = self.block0(x)
      out = self.block1(out)
      out = self.block2(out)
      out = self.block3(out)
      out = self.last(out)
      return out
      

# Generator Module
class Generator(nn.Module):
    def __init__(self, enc_pretrained=False, units=64, mask_unit=128, 
                 dec_activation='sigmoid', mask_activation=None):
      
        super(Generator, self).__init__()

        self.enc_pretrained = enc_pretrained
        self.units = units
        self.mask_unit = mask_unit
        self.dec_activation = dec_activation
        self.mask_activation = mask_activation
        
        # For content representation, only output of last layers of encoder is used.
        self.base_dep = {14, 16} 
        
        # For gram (style) representation, output of all layers of encoder is used.
        self.gram_dep = {2, 4, 7, 9, 12, 14, 17, 19}
        
        # For output representation, output of only last layer of encoder is used. 
        self.perc_dep = 20

        self.encoder = self.create_vgg_encoder(self.enc_pretrained)
        self.mask = Mask(in_channel=self.units * 16, out_channel=self.units * 8, unit=self.mask_unit,
                         activation=self.mask_activation)
        self.decoder = Decoder(out_channel=3, units=self.units, activation=self.dec_activation)

    def forward(self, x1, x2):
      base1, perc1, _ = self.get_base_perc_gram(x1, gram_flag=False)
      base2, _, gram2 = self.get_base_perc_gram(x2, gram_flag=True)

      base1 = base1.detach()
      base2 = base2.detach()

      adin12 = self.adin_transform(base1, base2)
      mask = self.mask(torch.cat([base1, base2], dim=1))      
      
      code = mask*base1 + (1-mask)*adin12
      mixture = self.decoder(code)
      return mixture, perc1, gram2, mask

    # Encoder module, pretrained VGG-16 network
    def create_vgg_encoder(self, pretrained):
      encoder = torchvision.models.vgg16(pretrained=pretrained)
      return encoder.features[:21]


    # AdaIN transform
    def adin_transform(self, base, base2):  
      #whitening
      batch_s, channel, width, height = base.size()
      base_vec = base.view(batch_s, channel, width * height)  #vectorize feature map
      mu = torch.mean(base_vec, dim=2, keepdim=True) #get mean
      ss = torch.std(base_vec, dim=2, keepdim=True)
      b = (base_vec - mu)/torch.clamp(ss, min=1e-6)  #normalize

      #color transfer
      base_vec2 = base2.view(batch_s, channel, width * height)  #vectorize feature map
      mu2 = torch.mean(base_vec2, dim=2, keepdim=True) #get mean
      ss2 = torch.std(base_vec2, dim=2, keepdim=True)

      bvst = b*ss2 + mu2

      return bvst.view(batch_s, channel, width, height)

    # Calculate content and style representations
    def get_base_perc_gram(self, img, gram_flag=True, blur_flag=False):
      code = img; bases = []; grams = []

      if 0 in self.base_dep:
          bases.append(img)
      if gram_flag and 0 in self.gram_dep:
          grams.append(self.get_gram(img))
          
      for i in range(len(self.encoder)):
          code = self.encoder[i](code)
          
          # Content representations
          if (i+1) in self.base_dep:
              bases.append(code)
          # Gram matrices
          if gram_flag and (i+1) in self.gram_dep:
              grams.append(self.get_gram(code))
          # Output layer
          if (i+1) == self.perc_dep:
              out = code
      
      # Downsampling
      if i > 0 or 0 not in self.base_dep:
          bases = [nn.functional.avg_pool2d(b, kernel_size=2, stride=2) for b in bases] 
      if blur_flag:
          out = nn.functional.avg_pool2d(out, kernel_size=2, stride=2)
      base = torch.cat(bases, dim=1)
      return base,out,grams


    def get_gram(self, ftr, use_norm=True):
      b, c, h, w = ftr.size()  # batch, channel, height, width
      features = ftr.view(b, c, h * w)  # flatten
      G = torch.bmm(features, features.transpose(1,2))  # compute the gram product
      if use_norm:
          return G.div(c*h*w)
      else:
          return G.div(b)