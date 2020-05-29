from Dataset import AnimeDataset
from Network import VAE
from Network import latent_dim as LATENT_DIM

from torch.utils.data import DataLoader, Subset
import torch
import torchvision
from torch import nn
from torchsummary import summary
from torchvision.utils import save_image

from torchvision import transforms
from PIL import Image

import os
import time
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = 'data'  # contain image 1.png、2.png...

BATCH_SIZE = 64
IMAGE_SIZE = (64, 64)

# torch.manual_seed(96)
EPOCHS = 20
learning_rate = 1e-4
KL_COEF = 1

dataset = AnimeDataset(DATA_DIR)
print('The number of data:', len(dataset))

train_set = Subset(dataset, range(0, len(dataset) * 8 // 10))
test_set = Subset(dataset, range(len(dataset) * 8 // 10, len(dataset)))
print('train set : {} , test set : {}'.format(len(train_set), len(test_set)))

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
# show_batch(train_loader)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VAE().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)


def plot_loss(history, title):
    fig = plt.figure()
    plt.plot(range(EPOCHS), history)
    plt.title(title)
    plt.xlabel('Number of epochs')
    plt.ylabel('Loss')
    # plt.show()
    fig.savefig('./log/{}_{}_{}epochs.png'.format('VAE', title, EPOCHS))
    plt.close()


# bce_loss = nn.BCELoss(size_average=False)
bce_loss = nn.BCELoss(reduction='sum')
mse_loss = nn.MSELoss(reduction='sum')


# Reconstruction Loss + KL divergence loss
def loss_function(reconstructed_x, x, mu, logvar):
    Reconstruction_Loss = bce_loss(reconstructed_x, x)
    # Reconstruction_Loss = mse_loss(reconstructed_x, x)
    KLD = -0.5 * torch.sum(logvar + 1 - torch.pow(mu, 2) - torch.exp(logvar))

    return Reconstruction_Loss + KL_COEF * KLD


def train():
    model.train()
    loss_sum = 0
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()

        x_rec, mu, logvar = model(data)
        # if batch_idx == 1:
        #     print(data)
        #     print(x_rec)
        loss = loss_function(x_rec, data, mu, logvar)
        loss.backward()

        loss_sum += loss.item()
        optimizer.step()

    return loss_sum / len(train_loader.dataset)


def test():
    model.eval()
    loss_sum = 0
    with torch.no_grad():
        for idx, data in enumerate(test_loader):
            data = data.to(device)
            x_rec, mu, logvar = model(data)
            loss_sum += loss_function(x_rec, data, mu, logvar).item()
            if idx == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n], x_rec.view(BATCH_SIZE, 3, IMAGE_SIZE[0], IMAGE_SIZE[1])[:n]])

    return loss_sum / len(test_loader.dataset), comparison


if __name__ == '__main__':

    os.makedirs('./results', exist_ok=True)
    os.makedirs('./log', exist_ok=True)

    summary(model, (3, IMAGE_SIZE[0], IMAGE_SIZE[1]))
    # print(model)
    # print('VAE TRAINING...')
    # # Training starts
    # loss_history = {'train': [], 'test': []}
    # for e in range(1, EPOCHS + 1):
    #     since = time.time()  # record the epoch start time
    #
    #     train_loss = train()  # train 1 epoch
    #     test_loss, test_result = test()  # test 1 epoch
    #
    #     print('Epoch {} costs {:.2f} secs.'.format('%02d' % e, time.time() - since))
    #     print('train loss: {:.2f}'.format(train_loss))
    #     print('test loss: {:.2f}\n'.format(test_loss))
    #
    #     save_image(test_result.cpu(),
    #                'results/reconstruction_' + '%02d' % e + '.png', nrow=8)
    #     # record the loss
    #     loss_history['train'].append(train_loss)
    #     loss_history['test'].append(test_loss)
    #
    # plot_loss(loss_history['train'], 'Learning Curve')
    # plot_loss(loss_history['test'], 'Test Loss')
    # torch.save(model.state_dict(), '{}_kl{}_{}epochs.pth'.format('VAE', KL_COEF, EPOCHS))
    # training end

    # torch.manual_seed(96)
    # synthesize some examples
    model.load_state_dict(torch.load('VAE_kl{}_20epochs.pth'.format(KL_COEF), map_location=device))
    model.eval()
    with torch.no_grad():
        # make an 8*8 Synthesized samples
        sample = torch.randn(64, LATENT_DIM).to(device)
        sample = model.decoder(sample).cpu()
        save_image(sample.view(64, 3, IMAGE_SIZE[0], IMAGE_SIZE[1]), 'results/sample.png', nrow=8)
        # interpolation of two latent codes z between samples
        sample = next(iter(train_loader))
        # print(sample.shape)
        z_1 = sample[0, :].view(1, 3, 64, 64).to(device)
        z_2 = sample[1, :].view(1, 3, 64, 64).to(device)
        # print(z_1.shape)
        # z_1 = torch.randn(1, LATENT_DIM).to(device)
        # z_2 = torch.randn(1, LATENT_DIM).to(device)
        # print(z_1.shape)
        z_1 = model.encoder(z_1)[:, :512]
        z_2 = model.encoder(z_2)[:, :512]
        interval = 7
        inte_start = z_1
        interpolation = z_1
        for i in range(1, interval + 1):
            inte_start += (i * (z_2 - z_1) / 7)
            interpolation = torch.cat((interpolation, inte_start))
            # print(inte_start.shape, interpolation.shape)

        interpolation = model.decoder(interpolation).cpu()
        # print(interpolation.shape)
        save_image(interpolation, 'results/interpolation.png')
        print('Synthesized images based on the interpolation of two latent codes z Finished')
