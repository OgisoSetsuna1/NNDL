import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            nn.ConvTranspose2d(100 + 22, 128 * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(128 * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(128 * 8, 128 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128 * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(128 * 4, 128 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128 * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(128 * 2, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif classname.find('BatchNorm') != -1:
                nn.init.normal_(m.weight.data, 1, 0.02)
                nn.init.constant_(m.bias.data, 0)

    def forward(self, z, hair_label, eye_label):
        in_ = torch.cat((z, hair_label, eye_label), -1).unsqueeze(-1).unsqueeze(-1)
        img = self.model(in_)

        return img

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.part1 = nn.Sequential(
            nn.Conv2d(3, 128, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128 * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128 * 2, 128 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128 * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128 * 4, 128 * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128 * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128 * 8, 128, 4, 1, 0, bias=False),
        )

        self.part2 = nn.Sequential(
            nn.Linear(128 + 22, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, img, hair_label, eye_label):
        v = self.part1(img).squeeze(-1).squeeze(-1)
        v = torch.cat((v, hair_label, eye_label), -1)
        v = self.part2(v)

        return v

if __name__ == '__main__':
    g_model = Generator().cuda()
    d_model = Discriminator().cuda()
    
    print(summary(g_model, [(100, ), (12, ), (10, )]))
    print(summary(d_model, [(3, 64, 64), (12, ), (10, )]))