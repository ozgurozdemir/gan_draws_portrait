# Src: https://github.com/nightldj/behance_release/blob/master/make_loss.py

import torch
import torchvision
import torch.nn as nn


class LossFunction(nn.Module):
  def __init__(self, w_gan=1.0, w_style=1.0, use_cuda=True):
    super(LossFunction, self).__init__()
    self.mse = nn.MSELoss()
    self.ce = nn.CrossEntropyLoss()
    self.bce = nn.BCEWithLogitsLoss()
    self.l1 = nn.L1Loss()
    self.use_cuda = use_cuda
    self.w_gan = w_gan
    self.w_style = w_style

    self.real_lbl = 1.0
    self.fake_lbl = 0.0
    self.loss = []

  def reset(self):
    self.loss = []

  def add_mse(self, input, target, weight=1.0):
    loss = weight*self.mse(input, target)
    self.loss.append(loss)

  def add_l1(self, input, target, weight=1.0):
    loss = weight*self.l1(input, target)
    self.loss.append(loss)

  def add_ce(self, input, target, weight=1.0):
    loss = weight*self.ce(input, target)
    self.loss.append(loss)

  def add_gan_real(self, input, weight=1.0):
    real_labels = torch.FloatTensor(input.size()).fill_(self.real_lbl)
    real_labels = torch.autograd.Variable(real_labels, requires_grad=False)
    if self.use_cuda:
      real_labels = real_labels.cuda()

    loss = weight*self.bce(input, real_labels)
    self.loss.append(loss)

  def add_gan_fake(self, input, weight=1.0):
    fake_labels = torch.FloatTensor(input.size()).fill_(self.fake_lbl)
    fake_labels = torch.autograd.Variable(fake_labels, requires_grad=False)
    if self.use_cuda:
      fake_labels = fake_labels.cuda()

    loss = weight*self.bce(input, fake_labels)
    self.loss.append(loss)

  def return_loss(self):
    return sum(self.loss)