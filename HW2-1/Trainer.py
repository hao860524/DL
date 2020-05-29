import torch
from torch import nn
import matplotlib.pyplot as plt
import time


class Trainer:
    def __init__(self, interval, dataloader, net, num_epochs, lr=0.05, momentum=0.9):
        self.interval = interval

        self.train_loader, self.test_loader = dataloader

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = net.to(self.device)
        # self.model.__class__.__name__
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=1e-3)
        self.criterion = nn.CrossEntropyLoss()
        # self.criterion = nn.BCELoss()
        self.epochs = num_epochs

    def plot_acc(self, history, title):
        fig = plt.figure()
        plt.plot(range(self.epochs), history)
        plt.title(title)
        plt.xlabel('Number of epochs')
        plt.ylabel('Accuracy rate')
        # plt.show()
        fig.savefig('./log/{}_L{}_{}_{} epochs.png'
                    .format(self.model.__class__.__name__, self.interval, title, self.epochs))
        plt.close()

    def run(self):
        # print(len(self.test_loader), len(self.test_loader.dataset))
        # print(len(self.train_loader), len(self.train_loader.dataset))
        loss_history = {'train': [], 'test': []}
        accuracy_history = {'train': [], 'test': []}

        for self.epoch in range(self.epochs):
            since = time.time()  # record the epoch start time

            train_loss, train_acc = self.train()  # train 1 epoch
            test_loss, test_acc = self.test()  # test 1 epoch

            print('Epoch {} costs {:.2f} secs.'.format("%03d" % self.epoch, time.time() - since))
            print('train loss: {:.4f} Acc: {:.4f}'.format(train_loss, train_acc))
            print('test loss: {:.4f} Acc: {:.4f}\n'.format(test_loss, test_acc))

            # record the loss
            loss_history['train'].append(train_loss)
            loss_history['test'].append(test_loss)

            # record the accuracy
            accuracy_history['train'].append(train_acc)
            accuracy_history['test'].append(test_acc)

        self.plot_acc(accuracy_history['train'], 'Training Acc')
        self.plot_acc(accuracy_history['test'], 'Test Acc')

        torch.save(self.model.state_dict(), '{}_L{}_{}epochs.pth'
                   .format(self.model.__class__.__name__, self.interval, self.epochs))

    def train(self):
        # open the dropout & batch normalization
        self.model.train()

        loss_sum = 0
        acc = 0
        for seq, label in iter(self.train_loader):
            self.optimizer.zero_grad()
            seq = seq.to(self.device).view(-1, self.interval, 1)
            label = label.to(self.device)
            # zero the parameter gradients
            outputs = self.model(seq)
            # print(outputs.size(), label.squeeze())
            loss = self.criterion(outputs, label.squeeze())  # label shape [N(batch_size)]
            loss_sum += loss.item()
            # compute the accuracy for each batch
            _, preds = torch.max(outputs, 1)
            acc += torch.sum(preds == label.squeeze()).double()
            # print(loss, acc)
            # backward propagation + optimization
            loss.backward()
            self.optimizer.step()

        return (loss_sum / len(self.train_loader)), (acc / len(self.train_loader.dataset))

    @torch.no_grad()  # turn off gradient
    def test(self):
        # close the dropout & batch normalization
        self.model.eval()

        loss_sum = 0
        acc = 0
        for seq, label in iter(self.test_loader):
            seq = seq.to(self.device).view(-1, self.interval, 1)
            label = label.to(self.device)

            outputs = self.model(seq)
            # compute the test loss
            loss = self.criterion(outputs, label.squeeze())  # label shape [N(batch_size)]
            loss_sum += loss.item()
            # compute the accuracy
            _, preds = torch.max(outputs, 1)
            acc += torch.sum(preds == label.squeeze()).double()

        return (loss_sum / len(self.test_loader)), (acc / len(self.test_loader.dataset))
