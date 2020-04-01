"""
The main structure is adopted from STS (https://arxiv.org/abs/1611.07932)
The original autoencoder is changed to variational ae.
The original VAE please refer to https://arxiv.org/abs/1312.6114

The structure of this code is adapted from pytorch official tutorial https://github.com/pytorch/examples/blob/master/vae/main.py
"""

import torch
from torch import nn
from torch.nn import functional as F


class VAE(nn.Module):
    def __init__(self, vector_length):
        super(VAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(10, 20, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(20, 50, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(50, 100, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc11 = nn.Linear(1600, vector_length)
        self.fc12 = nn.Linear(1600, vector_length)
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

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        feature = self.encoder(x)
        mu = self.fc11(feature.view(-1, 1600))
        logvar = self.fc12(feature.view(-1, 1600))
        z = self.reparameterize(mu, logvar)
        feature2 = F.relu(self.fc2(z))
        recon_x = self.decoder(feature2.view(-1, 100, 4, 4))  # B, C, H, W
        return recon_x, z, mu, logvar


"""
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 64*64), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def train(epoch, model, train_loader, optimizer, args):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx/len(train_loader), loss.item()/len(data)))
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))

def parse_args():
    parser = argparse.ArgumentParser(description='VAE for shape manifold.')
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
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    return args

if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if args.cuda else "cpu")
    kwargs = {'num_workers':1, 'pin_memory':True} if args.cuda else {}

    # train_loader

    model = VAE(args.vector_length).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, args.epochs + 1):
        train(epoch, model, train_loader, optimizer, args)
"""
