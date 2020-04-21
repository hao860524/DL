from torch import nn
from Preprocess import IMG_SIZE


class ConvBlock(nn.Module):
    def __init__(self, cin, cout, ks, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, kernel_size=ks, stride=stride, padding=ks // 2)
        self.bn = nn.BatchNorm2d(cout)
        self.relu = nn.LeakyReLU()  # nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Net(nn.Module):
    def __init__(self, ks, stride):
        super().__init__()
        self.kernel_size = ks
        self.stride = stride

        self.pool_1st = 2
        self.pool_2nd = 2
        self.pool_3rd = 2
        # self.pool_4th = 2
        self.features = nn.Sequential(
            ConvBlock(3, 32, self.kernel_size, self.stride), # shape(32, 128, 128)
            ConvBlock(32, 32, self.kernel_size, self.stride), # shape(32, 128, 128)
            nn.MaxPool2d(self.pool_1st), # shape(32, 64, 64)

            ConvBlock(32, 32, self.kernel_size, 1), # shape(32, 64, 64)
            ConvBlock(32, 32, self.kernel_size, 1), # shape(32, 64, 64)
            nn.MaxPool2d(self.pool_2nd), # shape(32, 32, 32)

            ConvBlock(32, 32, self.kernel_size, 1),  # shape(32, 32, 32)
            ConvBlock(32, 32, self.kernel_size, 1),  # shape(32, 32, 32)
            nn.MaxPool2d(self.pool_3rd)  # shape(32, 16, 16)
        )
        # default stride=1 shape(32, 16, 16)
        #
        #      if stride=2: (32, 4, 4)
        fc_input = 32 * (IMG_SIZE[0] // self.stride // self.stride // self.pool_1st
                         // self.pool_2nd // self.pool_3rd ) ** 2

        self.regression = nn.Sequential(
            #nn.Flatten(),
            Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(fc_input, (fc_input//16)),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear((fc_input//16), 3)
        )

    def forward(self, x):
        feature = self.features(x)
        return self.regression(feature)
