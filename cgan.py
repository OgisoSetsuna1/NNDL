import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feature, out_feature, normalize=True):
            layers = [nn.Linear(in_feature, out_feature)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feature, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))

            return layers
        
        self.model = nn.Sequential(
            *block(100 + 22, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, 64 * 64 * 3),
            nn.Tanh()
        )

    def forward(self, z, hair_label, eye_label):
        in_ = torch.cat((z, hair_label, eye_label), -1)
        img = self.model(in_)
        img = img.view(-1, *(3, 64, 64))

        return img

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(64 * 64 * 3 + 22, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img, hair_label, eye_label):
        in_ = torch.cat((img.view(-1, 64 * 64 * 3), hair_label, eye_label), -1)
        v = self.model(in_)

        return v

if __name__ == '__main__':
    g_model = Generator().cuda()
    d_model = Discriminator().cuda()
    
    print(summary(g_model, [(100, ), (12, ), (10, )]))
    print(summary(d_model, [(3, 64, 64), (12, ), (10, )]))