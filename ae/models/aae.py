"""
Adversarial Autoencoder (https://arxiv.org/abs/1511.05644) using Wasserstein loss (https://arxiv.org/abs/1701.07875)

Code logic is adopted from https://github.com/maitek/waae-pytorch/blob/master/WAAE.py

The encoder and decoder is also similar to STS paper (https://arxiv.org/abs/1611.07932)
"""

import torch
from torch import nn, optim
from torch.nn import functional as F


class Encoder(nn.Module):
    def __init__(self, vector_length, bottle_neck=True):
        super(Encoder, self).__init__()
        self.bottle_neck = bottle_neck
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32 * 32
            nn.Conv2d(10, 20, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16 * 16
            nn.Conv2d(20, 50, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 8 * 8
            nn.Conv2d(50, 100, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2))  # 4 * 4
        if self.bottle_neck:
            self.bottle_down = nn.Sequential(
                nn.Conv2d(100, 200, 3, 1, 1),
                nn.ReLU(),
                nn.MaxPool2d(2),  # 2 * 2
                nn.Conv2d(200, 400, 3, 1, 1),
                nn.ReLU(),
                nn.MaxPool2d(2),  # 1 * 1
                nn.Conv2d(400, vector_length, 1, 1, 0))
        self.fc1 = nn.Linear(1600, vector_length)

    def forward(self, x):
        feature = self.encoder(x)
        if self.bottle_neck:
            z_out = self.bottle_down(feature)
            z = z_out  # need transpose
        else:
            z = self.fc1(feature.view(-1, 1600))
        recon_x = self.decoder(feature2)
        return z


class Decoder(nn.Module):
    def __init__(self, vector_length, bottle_neck=True):
        super(Decoder, self).__init__()

        if bottle_neck:
            self.bottle_up = nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(vector_length, 400, 1, 1, 0),
                nn.Upsample(scale_factor=2, mode='bilinear'),
                nn.Conv2d(400, 200, 3, 1, 1))  # 2 * 2,
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.ReLU())

            self.fc2 = nn.Linear(vector_length, 1600)
            self.decoder = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='bilinear'),
        nn.Conv2d(100, 50, 3, 1, 1),
        nn.ReLU(),
        nn.Upsample(scale_factor=2, mode='bilinear'),
        nn.Conv2d(50, 20, 3, 1, 1),
        nn.ReLU(),
        nn.Upsample(scale_factor=2, mode='bilinear'),
        nn.Conv2d(20, 10, 3, 1, 1),
        nn.ReLU(),
        nn.Upsample(scale_factor=2, mode='bilinear'),
        nn.Conv2d(10, 1, 3, 1, 1),
        nn.Sigmoid())


    def forward(self, x):
        if bottle_neck:
            z = z  # need transpose
            feature = self.bottle_up(z)
        else:
            feature = F.relu(self.fc2(z))
            feature = feature.view(-1, 100, 4, 4)  # B, C, H, W
        recon_x = self.decoder(feature)
        return recon_x


class Discriminator(nn.Module):
    def __init__(self, vector_length):
        super(Discriminator, self).__init__()
        self.classfier = nn.Sequential(
            nn.Linear(vector_length, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.classifier(x)


def loss_function(recon_x, x, func="mse"):
    if func == "mse":
        loss = F.mse_loss(recon_x, x)
    elif func == "bce":
        loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    return loss


def train(epoch, Q, P, D, train_loader, Q_optimizer, P_optimizer, D_optimizer, args):
    Q.train()
    P.train()
    D.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)

        """ Reconstruction phase"""
        Q_optimizer.zero_grad()
        P_optimizer.zero_grad()
        z_sample = Q(data)
        recon_batch = P(z_sample)
        recon_loss = loss_function(recon_batch, data, "mse")
        recon_loss.backward()

        P_optimizer.step()
        Q_optimizer.step()

        """ Regularization phase"""
        D_optimizer.zero_grad()
        for _ in range(5):
            z_real = torch.randn(args.batch_size, args.vector_length)
            z_real = z_real.cuda()

            z_fake = z_fake.view(args.batch_size, -1)
            D_real = D(z_real)
            D_fake = D(z_fake)

            D_loss = -(torch.mean(D_real) - torch.mean(D_fake))  # Wasserstein loss
            # should modified D, original repo is wrong

            D_loss.backward()
            D_optimizer.step()

            for p in D.paramters():  # weight clipping
                p.data.clamp_(-0.01, 0.01)

        # Generator
    z_fake = Q(data).view(batch_size, -1)
    D_fake = D(z_fake)
    G_loss = -torch.mean(D_fake)
    G_loss.backward()
    Q_optimizer.step()

    # save model

    if batch_idx % args.log_interval == 0:
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data),
                                                                       len(train_loader.dataset),
                                                                       100. * batch_idx / len(train_loader),
                                                                       loss.item() / len(data)))


print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))


def parse_args():
    parser = argparse.ArgumentParser(description='WAAE for shape manifold')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--vector_length', type=int, default=20, help='length of the shape vector')
    parser.add_argument('--bottle_neck', type=bool, default=False, help='vector is convolved or flatten')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    return args


if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if args.cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    # train_loader
    # add noise to input 0.5 * N(0,1)

    Q = Encoder(args.vector_length, args.bottle_neck).to(device)
    P = Decoder(args.vector_length, args.bottle_neck).to(device)
    D = Discriminator(args.vector_length).to(device)

    Q_optimizer = optim.Adam(Q.parameters(), lr=1e-3)
    P_optimizer = optim.Adam(P.parameters(), lr=1e-3)
    D_optimizer = optim.Adam(D.parameters(), lr=1e-4)

    for epoch in range(1, args.epochs + 1):
        train(epoch, Q, P, D, train_loader, Q_optimizer, P_optimizer, D_optimizer, args)
