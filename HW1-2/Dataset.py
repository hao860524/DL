from torch.utils.data import Dataset
import torch
from Preprocess import *
from Preprocess import IMG_SIZE


class dataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.imgs = data['image']
        self.labels = data['label']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        img = self.imgs[idx]
        img = np.swapaxes(img, 0, 1).reshape(3, IMG_SIZE[1], IMG_SIZE[0]) / 255
        img = torch.FloatTensor(img)

        label = self.labels[idx]
        if label == 'good':
            l = np.array([0])

        elif label == 'none':
            l = np.array([1])

        else:
            l = np.array([2])
        # l = np.array([label == 'good', label == 'none', label == 'bad']).astype(int)

        category = torch.tensor(l, dtype=torch.long)

        return img, category
