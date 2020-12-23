# Src: https://github.com/nightldj/behance_release/blob/master/my_discriminator.py

import torch
import torchvision
import torch.nn as nn

import numpy as np
import time
import random


class Discriminator(nn.Module):
    def __init__(self, num_style, num_content, depth=3, units=64, use_sigmoid=False,
                 use_projection=True, use_content=False):
        super(Discriminator, self).__init__()
        self.num_style = num_style
        self.num_content = num_content
        self.depth = depth
        self.units  = units
        self.use_sigmoid = use_sigmoid
        self.use_projection = use_projection
        self.use_content = use_content
        
        # patch discriminator
        self.main = nn.Sequential(
            # input layer
            nn.Conv2d(self.depth, self.units, kernel_size=4, stride=2, padding=1,
                      bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            # first layer
            nn.Conv2d(self.units, self.units * 2, kernel_size=4, stride=2, padding=1,
                      bias=True),
            nn.BatchNorm2d(self.units * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # second layer
            nn.Conv2d(self.units * 2, self.units * 4, kernel_size=4, stride=2, padding=1,
                      bias=True),
            nn.BatchNorm2d(self.units * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # third layer
            nn.Conv2d(self.units * 4, self.units * 8, kernel_size=4, stride=2, padding=1,
                      bias=True),
            nn.BatchNorm2d(self.units * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # binary classifier
        self.binary_classifier = nn.Sequential(
            nn.Conv2d(self.units * 8, 1, kernel_size=3, stride=1, padding=1)
        )
        if self.use_sigmoid:
          self.binary_classifier.add_module("sigmoid", nn.Sigmoid())

        # style and content classifiers
        if self.use_projection:
          self.style_projection = nn.Embedding(self.num_style, self.units * 8)
          self.style_projection.weight.data.fill_(0)  
          if self.use_content:
            self.content_projection = nn.Embedding(self.num_content, self.units * 8)
            self.content_projection.weight.data.fill_(0)
        
        self.style_classifier = nn.Sequential(nn.Linear(512*4*4, self.num_style))
        if self.use_content:
          self.content_classifier = nn.Sequential(nn.Linear(512*4*4, self.num_content))

    def forward(self, x, y_style = None, y_content = None):
      out = self.main(x)
      binary_out = self.binary_classifier(out)

      if self.use_projection:
          
        # style embedding 
        if y_style is not None:
          style_emb = self.style_projection(y_style)
          style_project = out * style_emb.view(style_emb.size(0), style_emb.size(1), 1, 1) 
          binary_out += torch.mean(style_project, dim=1, keepdim=True)

        # content embedding
        if self.use_content and y_content is not None:
          content_emb = self.content_projection(y_content)
          content_project = out * content_emb.view(content_emb.size(0), content_emb.size(1), 1, 1) 
          binary_out += torch.mean(content_project, dim=1, keepdim=True)

      out = nn.functional.avg_pool2d(out, kernel_size=2)
      out = out.view(out.size(0), -1) # flatten

      style_out = self.style_classifier(out)
      content_out = self.content_classifier(out) if self.use_content else None

      return binary_out, style_out, content_out
