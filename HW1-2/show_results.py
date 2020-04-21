from Preprocess import face_crop
from Preprocess import IMG_SIZE
from Dataset import dataset
from Network import Net
import torch
import cv2
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

ROOT_DIR = './problem2-CNN'
train_scv = '/train.csv'
test_scv = '/test.csv'
# pth = 'model_50epochs_f3_s1.pth'
pth = 'CELoss_weight_model_50epochs_f3_s1.pth' # 改進過的loss train出來的model
visul = ['20200120000931.jpg', '1152x768_246964803156.jpg', '389711_84767ea8af93d57a_o.jpg']


def val(loader, net, para):
    net.load_state_dict(torch.load(para, map_location=device))
    net.eval()

    category = ['good', 'none', 'bad']
    correct = {'good': 0, 'none': 0, 'bad': 0}
    total = {'good': 0, 'none': 0, 'bad': 0}
    acc = {'good': '0', 'none': '0', 'bad': '0'}
    iteration = 0
    for img, label in iter(loader):
        iteration += 1
        if iteration % 100 == 0: print('Iteration', iteration)
        pred_test = net(img.to(device)).to('cpu')
        total[category[label.item()]] += 1

        _, pred = torch.max(pred_test, 1)
        if pred == label.squeeze():
            correct[category[label.item()]] += 1

    for c in category:
        acc[c] = '%.2f' % (correct[c] / total[c] * 100) + '%'
    return acc


if __name__ == '__main__':
    # train acc
    print('processing training data...')
    train_set = face_crop(ROOT_DIR + train_scv)
    train_data = dataset(train_set)
    # test acc
    print('processing test data...')
    test_set = face_crop(ROOT_DIR + test_scv)
    test_data = dataset(test_set)

    train_loader = DataLoader(dataset=train_data, num_workers=4)
    test_loader = DataLoader(dataset=test_data, num_workers=4)

    ##### bulid a validation model #####
    device = torch.device('cuda')
    val_net = Net(ks=3, stride=1).to(device)
    print(val_net)

    train_acc = val(train_loader, val_net, pth)
    test_acc = val(test_loader, val_net, pth)
    print('Train acc', train_acc)
    print('Test acc', test_acc)

    val_net.load_state_dict(torch.load(pth, map_location=device))
    val_net.eval()

    visul_data = pd.read_csv(ROOT_DIR + test_scv)
    for v in visul:
        img = cv2.imread(ROOT_DIR+'/images/'+v)
        start = test_set['name'].index(v)
        step = test_set['name'].count(v)
        # predict each picture patch
        for i in range(step):
            row = visul_data.iloc[start + i]
            input = test_set['image'][start + i]

            input = np.swapaxes(input, 0, 1).reshape(3, IMG_SIZE[1], IMG_SIZE[0]) / 255
            input = torch.FloatTensor(input)
            input = input.view(1, input.size(0), input.size(1), input.size(2))
            output = val_net(input.to(device)).to('cpu')
            _, pred = torch.max(output, 1)
            if pred == 0:
                img = cv2.rectangle(img, (row['xmin'], row['ymin']), (row['xmax'], row['ymax']), (0, 255, 0), 2)
                img = cv2.putText(img, 'good', (row['xmin'], row['ymin']-8), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            if pred == 1:
                img = cv2.rectangle(img, (row['xmin'], row['ymin']), (row['xmax'], row['ymax']), (255, 191, 0), 2)
                img = cv2.putText(img, 'none', (row['xmin'], row['ymin']-8), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 191, 0), 2)
            if pred == 2:
                img = cv2.rectangle(img, (row['xmin'], row['ymin']), (row['xmax'], row['ymax']), (0, 0, 255), 2)
                img = cv2.putText(img, 'bad', (row['xmin'], row['ymin']-8), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        cv2.imwrite('pred_'+v, img)

    print('Finished')