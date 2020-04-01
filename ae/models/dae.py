"""
The main structure is adopted from STS (https://arxiv.org/abs/1611.07932)
The original autoencoder is changed to denoising ae.

The structure of this code is adapted from pytorch official tutorial https://github.com/pytorch/examples/blob/master/vae/main.py
"""

from torch import nn
from torch.nn import functional as F


class DAE(nn.Module):
    def __init__(self, vector_length, bottle_neck=True):
        super(DAE, self).__init__()
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
            self.bottle_up = nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(vector_length, 400, 1, 1, 0),
                nn.Upsample(scale_factor=2, mode='bilinear'),
                nn.Conv2d(400, 200, 3, 1, 1),  # 2 * 2
                nn.ReLU(),
                nn.Upsample(scale_factor=2, mode='bilinear'),
                nn.ReLU())
        self.fc1 = nn.Linear(1600, vector_length)
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
        print(f"input x {x.size()}")
        feature = self.encoder(x)
        print(f"feature {feature.size()}")
        if self.bottle_neck:
            print("Bottle_neck")
            z_out = self.bottle_down(feature)
            z = z_out  # need transpose
            feature2 = self.bottle_up(z_out)
        else:
            print("NO Bottle_neck")
            z = self.fc1(feature.view(-1, 1600))
            feature2 = F.relu(self.fc2(z))
            feature2 = feature2.view(-1, 100, 4, 4)  # B, C, H, W
        recon_x = self.decoder(feature2)
        return recon_x, z


"""
def loss_function(recon_x, x):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 64*64), reduction='sum')
    return BCE

def train(epoch, model, train_loader, optimizer, args):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx/len(train_loader), loss.item()/len(data)))
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))

def parse_args():
    parser = argparse.ArgumentParser(description='DAE for shape manifold.')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log_interval', type=int, default=10, metavar='N',
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
    kwargs = {'num_workers':1, 'pin_memory':True} if args.cuda else {}

    # train_loader
    # add noise to input 0.5 * N(0,1)
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    train_dataset = datasets.ImageFolder(root='/ssd2/wenqiang/ae/data/norm_mask_64', transform=data_transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    model = DAE(args.vector_length, args.bottle_neck).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, args.epochs + 1):
        train(epoch, model, train_loader, optimizer, args)
"""
