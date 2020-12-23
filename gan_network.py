# Src: https://github.com/nightldj/behance_release/blob/master/train_mask.py

import torch
import torchvision
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt

import time
import random
import os

from discriminator import *
from generator import *
from loss_function import *


# GAN Module
class GAN(nn.Module):
    def __init__(self, num_style, num_content, depth=3, units=64, use_cuda=True, 
                 disc_use_sigmoid=False, disc_use_projection=True, disc_use_content=False,
                 gen_train_dec=True, gen_enc_pretrained=False, gen_mask_unit=128, 
                 gen_dec_activation='sigmoid', gen_mask_activation='tanh',
                 w_gan_loss=1.0, w_style_loss=1.0,
                 w_perceptron=1.0, w_gram=1.0, w_cycle=0.0, 
                 blur_perc=False, use_mse_loss=False):  

        super(GAN, self).__init__()

        self.units = units
        self.depth = depth
        self.use_cuda = use_cuda

        # Discriminator parameters
        self.num_style = num_style
        self.num_content = num_content
        self.disc_use_sigmoid = disc_use_sigmoid
        self.disc_use_projection = disc_use_projection
        self.disc_use_content = disc_use_content

        self.discriminator = Discriminator(self.num_style, self.num_content, 
                                           self.depth, self.units, 
                                           self.disc_use_sigmoid, 
                                           self.disc_use_projection,
                                           self.disc_use_content)
        
        # Generator parameters
        self.gen_mask_unit = gen_mask_unit
        self.gen_enc_pretrained = gen_enc_pretrained
        self.gen_dec_activation = gen_dec_activation
        self.gen_mask_activation = gen_mask_activation
        self.gen_train_dec = gen_train_dec

        self.generator = Generator(self.gen_enc_pretrained, self.units, 
                                   self.gen_mask_unit, self.gen_dec_activation,
                                   self.gen_mask_activation)

        # Freeze generator
        self.update_requires_grad(self.generator.encoder, update=False) 
        if not self.gen_train_dec: 
          self.update_requires_grad(self.generator.decoder, update=False) 

        # Optimizers and losses
        self.w_gan_loss = w_gan_loss
        self.w_style_loss = w_style_loss
        self.w_perceptron = w_perceptron
        self.w_cycle = w_cycle
        self.w_gram = w_gram
        self.blur_perc = blur_perc
        self.use_mse_loss = use_mse_loss
        self.loss_function = LossFunction(self.w_gan_loss, 
                                          self.w_style_loss,
                                          self.use_cuda)


    def update_requires_grad(self, net, update=False):
      for layer in net.modules():
        layer.requires_grad=update


    def train_discriminator(self, inputs, labels, style_inputs, style_labels,
                            style_loss, binary_loss, use_real):
      # Freeze encoder
      self.update_requires_grad(self.generator.mask, update=False)
      if self.gen_train_dec: 
        self.update_requires_grad(self.generator.decoder, update=False) 
      
      # Unfreeze discriminator
      self.update_requires_grad(self.discriminator, update=True)

      # Forward propogation
      mix_img, _, _, _ = self.generator(inputs, style_inputs)
      mix_bcout, mix_scout, mix_ccout = self.discriminator(mix_img.detach(), 
                                                           y_style=style_labels,
                                                           y_content=labels)
      style_bcout, style_scout, _ = self.discriminator(style_inputs, 
                                                       y_style=style_labels)

      self.loss_function.reset()

      # Style Classification Loss
      if self.loss_function.w_style > 0:
        self.loss_function.add_ce(mix_scout, style_labels, self.loss_function.w_style) 
        self.loss_function.add_ce(style_scout, style_labels, self.loss_function.w_style)
        style_loss += sum(self.loss_function.loss[:2])

      # Adversarial Loss
      if self.loss_function.w_gan > 0:
        if use_real:
          self.loss_function.add_gan_real(mix_bcout, self.loss_function.w_gan) 
          self.loss_function.add_gan_fake(style_bcout, self.loss_function.w_gan)
        else:
          self.loss_function.add_gan_fake(mix_bcout, self.loss_function.w_gan) 
          self.loss_function.add_gan_real(style_bcout, self.loss_function.w_gan)
        binary_loss += sum(self.loss_function.loss[-2:])

      # Sum total loss
      if self.loss_function.loss is not None and len(self.loss_function.loss) > 0:
        loss = self.loss_function.return_loss()
      else:
        loss = []
      return loss, binary_loss, style_loss


    def forward(self, inputs, labels, style_inputs, style_labels):
      labels = labels.view(labels.shape[0])
      style_labels = style_labels.view(style_labels.shape[0])
    
      mix_img, fout, gout, mout = self.generator(inputs, style_inputs)
      bcout, scout, ccout = self.discriminator(mix_img, y_style=style_labels,
                                               y_content=labels)
      
      # Style & Gan Losses
      self.loss_function.reset()
      if self.loss_function.w_style > 0:
        self.loss_function.add_ce(scout, style_labels, self.loss_function.w_style)
      if self.loss_function.w_gan > 0:
        self.loss_function.add_gan_real(bcout, self.loss_function.w_gan)
      
      # Perceptron & Gram Loss
      if self.w_gram > 0 or self.w_perceptron > 0 or self.w_cycle > 0:
        mix_base, mix_feat, mix_gram = self.generator.get_base_perc_gram(mix_img, 
                                                                         blur_flag=self.blur_perc)

        # Calculate Cycyle
        if self.w_cycle > 0:
          mix_cycle = self.generator.mask(torch.cat([mix_base.detach(), mix_base.detach()], dim=1)) 

        # Distance (norm) function
        dist_loss = self.loss_function.add_mse if self.use_mse_loss else self.loss_function.add_l1
        
        # Content & Gram & Cycle Losses
        dist_loss(mix_feat, fout.detach(), self.w_perceptron) # content loss
        for i in range(len(mix_gram)):
          dist_loss(mix_gram[i], gout[i].detach(), self.w_gram) # gram loss
        if self.w_cycle > 0:
          dist_loss(mout, mix_cycle.detach(), self.w_cycle) # cycle loss
            
      # Sum total loss
      if self.loss_function.loss is not None and len(self.loss_function.loss) > 0:
        loss = self.loss_function.loss
      else:
        loss = []

      return loss, mix_img
      
    # save & load model
    def save_model(self, gen_opt, disc_opt, epoch, gen_loss, path):
        torch.save({'epochs': epoch,
            'mask': self.generator.mask.state_dict(),
            'decoder': self.generator.decoder.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'gen_optimizer': gen_opt.state_dict(),
            'disc_optimizer': disc_opt.state_dict(),
            'gen_loss': gen_loss}, path)

    def load_model(self, gen_opt, disc_opt, path):
      checkpoint = torch.load(path)
      self.generator.mask.load_state_dict(checkpoint['mask'])
      self.generator.decoder.load_state_dict(checkpoint['decoder'])
      self.discriminator.load_state_dict(checkpoint['discriminator'])
      gen_opt.load_state_dict(checkpoint['gen_optimizer'])
      disc_opt.load_state_dict(checkpoint['disc_optimizer'])
    
      return gen_opt, disc_opt, checkpoint['epochs'], checkpoint['gen_loss']
      