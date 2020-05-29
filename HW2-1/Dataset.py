from torch.utils.data import Dataset
import torch


class torch_dataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        subsequence = torch.FloatTensor(self.data[idx][0])
        label = self.data[idx][1]

        if label:
            l = torch.tensor([1], dtype=torch.long)
        else:
            l = torch.tensor([0], dtype=torch.long)

        return subsequence, l
