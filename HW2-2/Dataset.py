from torch.utils.data import Dataset
from torchvision import transforms

import os
from PIL import Image


class AnimeDataset(Dataset):
    def __init__(self, path, transform=transforms.Compose([
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor()])
         ):
        """
        Args:
            path (string): Path to the folder
        """
        self.path = path
        self.transform = transform
        self.list_images = list(os.listdir(path))

    def __len__(self):
        return len(list(os.listdir(self.path)))

    def __getitem__(self, idx):
        img_name = self.list_images[idx]

        image = Image.open(os.path.join(self.path, img_name))
        image = self.transform(image)
        return image
