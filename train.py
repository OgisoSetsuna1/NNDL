import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
from torchvision.utils import save_image
import os
import csv

import cdcgan
import cgan
import data

def train(args):
    src = args.src
    epochs = args.epochs
    batch_size = args.batch_size
    glr = args.glr
    dlr = args.dlr
    valid_tag = args.valid_tag
    fake_tag = args.fake_tag
    gd_factor = args.gd_factor
    device = args.device
    sample_num = args.sample_num
    sample_src = args.sample_src
    net = args.net
    
    dataset = data.ImageLoader(src)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    loss_function = nn.BCELoss().to(device)
    if net == 0:
        g_model = cgan.Generator().to(device)
        d_model = cgan.Discriminator().to(device)
    elif net == 1:
        g_model = cdcgan.Generator().to(device)
        d_model = cdcgan.Discriminator().to(device)
    else:
        raise ValueError('Net type error!')
    
    g_optimizer = optim.Adam(g_model.parameters(), lr=glr, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(d_model.parameters(), lr=dlr, betas=(0.5, 0.999))

    log_file = open('./log.csv', 'w', newline='')
    writer = csv.writer(log_file)
    writer.writerow(['epoch', 'g_loss', 'd_loss'])

    for epoch in range(epochs):
        g_loss_list = []
        d_loss_list = []
        for batch_idx, (imgs, hair_label, eye_label) in enumerate(dataloader):
            imgs = imgs.to(device)
            valid = Variable(torch.FloatTensor(imgs.size(0), 1).fill_(valid_tag).to(device), requires_grad=False)
            fake = Variable(torch.FloatTensor(imgs.size(0), 1).fill_(fake_tag).to(device), requires_grad=False)

            # Training Generator
            g_optimizer.zero_grad()
            z = Variable(torch.FloatTensor(np.random.normal(0, 1, (imgs.size(0), 100))).to(device))
            z_hair_label = Variable(torch.FloatTensor(np.random.normal(0, 1, (imgs.size(0), 12))).to(device))
            z_eye_label = Variable(torch.FloatTensor(np.random.normal(0, 1, (imgs.size(0), 10))).to(device))
            fake_imgs = g_model(z, z_hair_label, z_eye_label)
            g_loss = loss_function(d_model(fake_imgs, z_hair_label, z_eye_label), valid)
            g_loss_list.append(g_loss.item())
            g_loss.backward(retain_graph=True)
            g_optimizer.step()

            # Training Discriminator
            if batch_idx % gd_factor == 0:
                d_optimizer.zero_grad()
                imgs_hair_label = F.one_hot(torch.as_tensor(hair_label), num_classes=12).to(device)
                imgs_eye_label = F.one_hot(torch.as_tensor(eye_label), num_classes=10).to(device)
                d_loss_1 = loss_function(d_model(imgs, imgs_hair_label, imgs_eye_label), valid)
                d_loss_2 = loss_function(d_model(fake_imgs.detach(), z_hair_label, z_eye_label), fake)
                d_loss = (d_loss_1 + d_loss_2) / 2
                d_loss_list.append(d_loss.item())
                d_loss.backward(retain_graph=True)
                d_optimizer.step()

            print('[Epoch: %d, batch: %d] Generator loss: %.6f Discriminator loss: %.6f' 
                % (epoch + 1, batch_idx, g_loss.item(), d_loss.item()))

        writer.writerow((epoch + 1, np.mean(g_loss_list), np.mean(d_loss_list)))
        save_image(fake_imgs.data[:(sample_num * sample_num)], 
            os.path.join(sample_src, '%d.png' % (epoch + 1)), nrow=sample_num, normalize=True)
    
    torch.save(g_model.state_dict(), './g_model.pt')
    torch.save(d_model.state_dict(), './d_model.pt')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training Parser')
    parser.add_argument('--src', type=str, default='./data', help='Path of training dataset')
    parser.add_argument('--sample-src', type=str, default='./sample', help='Path of saving sample pictures')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=256, help='Mini-Batch size')
    parser.add_argument('--glr', type=float, default=1e-3, help='Generator earning rate')
    parser.add_argument('--dlr', type=float, default=1e-3, help='Discriminator earning rate')
    parser.add_argument('--valid-tag', type=float, default=1.0, help='tag for valid samples')
    parser.add_argument('--gd-factor', type=int, default=1, help='G-D Training Iteration Factor')
    parser.add_argument('--fake-tag', type=float, default=0.0, help='tag for fake samples')
    parser.add_argument('--device', type=str, default='cuda', help='Training device')
    parser.add_argument('--sample-num', type=int, default=5, help='Samples in a picture(rooted)')
    parser.add_argument('--net', type=int, default=0, help='0 for cGAN, 1 for cDCGAN')

    args = parser.parse_args()
    train(args)