from Preprocess import face_crop
from Dataset import dataset
from Network import Net
from Trainer import Trainer
import os
from torch.utils.data import DataLoader
import torchvision

ROOT_DIR = './problem2-CNN'
train_scv = '/train.csv'
test_scv = '/test.csv'


if __name__ == '__main__':
    # train set
    print('processing training set...')
    train_set = face_crop(ROOT_DIR + train_scv)
    train_data = dataset(train_set)
    # test set
    print('processing test set...')
    test_set = face_crop(ROOT_DIR + test_scv)
    test_data = dataset(test_set)

    train_loader = DataLoader(dataset=train_data, batch_size=10,
                              shuffle=True, num_workers=4)

    test_loader = DataLoader(dataset=test_data, batch_size=10,
                             shuffle=False, num_workers=4)

    mask_Net = Net(ks=3, stride=1)
    mask_Net2 = Net(ks=3, stride=2)
    mask_Net3 = Net(ks=7, stride=1)
    mask_Net4 = Net(ks=7, stride=2)

    os.makedirs('./log', exist_ok=True)
    print('1st experiment (filter size=3*3, stride size=1) starts:')
    Trainer(dataloader=(train_loader, test_loader), net=mask_Net, num_epochs=50).run()
    print('2nd experiment (filter size=3*3, stride size=2) starts:')
    Trainer(dataloader=(train_loader, test_loader), net=mask_Net2, num_epochs=50).run()
    print('3rd experiment (filter size=7*7, stride size=1) starts:')
    Trainer(dataloader=(train_loader, test_loader), net=mask_Net3, num_epochs=50).run()
    print('4th experiment (filter size=7*7, stride size=2) starts:')
    Trainer(dataloader=(train_loader, test_loader), net=mask_Net4, num_epochs=50).run()
    print('Experiment Finished')





