import torch
from torch import nn
import matplotlib.pyplot as plt
import time


class Trainer:
    def __init__(self, dataloader, net, num_epochs, lr=1e-3, momentum=0.9):

        self.train_loader, self.test_loader = dataloader

        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.model = net.to(self.device)

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=1e-3)
        # define CrossEntropyLoss with weights
        weight = torch.tensor([1, 30, 1], dtype=torch.float)
        # self.criterion = nn.CrossEntropyLoss(weight=weight.to(self.device))
        self.criterion = nn.CrossEntropyLoss()
        self.epochs = num_epochs

    def run(self):

        loss_history = {'train': [], 'test': []}
        accuracy_history = {'train': [], 'test': []}

        for self.epoch in range(self.epochs):
            since = time.time()  # record the epoch start time

            train_loss, train_acc = self.train  # train 1 epoch
            test_loss, test_acc = self.test()  # test 1 epoch

            print('Epoch {} costs {:.2f} secs.'.format("%03d" % self.epoch, time.time() - since))
            print('train loss: {:.4f} Acc: {:.4f}'.format(train_loss, train_acc))
            print('test loss: {:.4f} Acc: {:.4f}\n'.format(test_loss, test_acc))

            # record the loss & acc
            loss_history['train'].append(train_loss)
            loss_history['test'].append(test_loss)

            # record the accuracy
            accuracy_history['train'].append(train_acc)
            accuracy_history['test'].append(test_acc)

        fig = plt.figure()
        plt.plot(range(self.epochs), loss_history['train'], label='train')
        plt.plot(range(self.epochs), loss_history['test'], label='test')
        plt.legend()
        plt.title('Learning Curve')
        plt.xlabel('Number of epochs')
        plt.ylabel('Cross entropy')
        # plt.show()
        fig.savefig('./log/loss_{} epochs_f={}_s={}.png'
                    .format(self.epochs, self.model.kernel_size, self.model.stride))
        plt.close()

        fig = plt.figure()
        plt.plot(range(self.epochs), accuracy_history['train'], label='train')
        plt.plot(range(self.epochs), accuracy_history['test'], label='test')
        plt.legend()
        plt.title('Training Accuracy')
        plt.xlabel('Number of epochs')
        plt.ylabel('Accuracy rate')
        # plt.show()
        fig.savefig('./log/accuracy_{} epochs_f={}_s={}.png'
                    .format(self.epochs, self.model.kernel_size, self.model.stride))
        plt.close()
        torch.save(self.model.state_dict(), 'model_{}epochs_f{}_s{}.pth'
                   .format(self.epochs, self.model.kernel_size, self.model.stride))

    @property
    def train(self):
        # open the dropout & batch normalization
        self.model.train

        loss_sum = 0
        acc = 0
        for img_train, category_train in iter(self.train_loader):
            img_train = img_train.to(self.device)
            category_train = category_train.to(self.device)

            # zero the parameter gradients
            self.optimizer.zero_grad()

            outputs = self.model(img_train)
            # compute the loss for each batch
            loss = self.criterion(outputs, category_train.squeeze())  # label shape [N(batch_size)]
            loss_sum += loss.item()
            # compute the accuracy for each batch
            _, preds = torch.max(outputs, 1)
            acc += torch.sum(preds == category_train.squeeze()).double()
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
        for img_test, category_test in iter(self.test_loader):
            img_test = img_test.to(self.device)
            category_test = category_test.to(self.device)

            outputs = self.model(img_test)
            # compute the test loss
            loss = self.criterion(outputs, category_test.squeeze())  # label shape [N(batch_size)]
            loss_sum += loss.item()
            # compute the accuracy
            _, preds = torch.max(outputs, 1)
            acc += torch.sum(preds == category_test.squeeze()).double()

        return (loss_sum / len(self.test_loader)), (acc / len(self.test_loader.dataset))
