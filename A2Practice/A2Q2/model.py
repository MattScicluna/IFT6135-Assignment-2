import torch.nn as nn


class imCNN3(nn.Module):
    """Convnet Classifier"""

    def __init__(self):
        super(imCNN3, self).__init__()
        self.conv = nn.Sequential(
            # Layer 1
            nn.Conv2d(in_channels=3, out_channels=18, kernel_size=(3, 3), padding=1),
            #nn.Dropout(p=0.5),
            nn.BatchNorm2d(18),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),

            # Layer 2
            nn.Conv2d(in_channels=18, out_channels=36, kernel_size=(3, 3), padding=1),
            #nn.Dropout(p=0.5),
            nn.BatchNorm2d(36),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),

            # Layer 3
            nn.Conv2d(in_channels=36, out_channels=72, kernel_size=(3, 3), padding=1),
            # nn.Dropout(p=0.5),
            nn.BatchNorm2d(72),
            nn.ReLU(),

            # Layer 4
            nn.Conv2d(in_channels=72, out_channels=72, kernel_size=(3, 3), padding=1),
            # nn.Dropout(p=0.5),
            nn.BatchNorm2d(72),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),

            # Layer 5
            nn.Conv2d(in_channels=72, out_channels=144, kernel_size=(3, 3), padding=1),
            # nn.Dropout(p=0.5),
            nn.BatchNorm2d(144),
            nn.ReLU(),

            # Layer 6
            nn.Conv2d(in_channels=144, out_channels=144, kernel_size=(3, 3), padding=1),
            # nn.Dropout(p=0.5),
            nn.BatchNorm2d(144),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
        )

        # Linear units
        self.linear = nn.Sequential(
            nn.Linear(144 * 4 * 4, 144 * 4 * 4),
            nn.BatchNorm1d(144 * 4 * 4),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(144 * 4 * 4, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(500, 2)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 144*4*4)
        x = self.linear(x)
        return x


class imCNN2(nn.Module):
    """Convnet Classifier"""

    def __init__(self):
        super(imCNN2, self).__init__()
        self.conv = nn.Sequential(
            # Layer 1
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=(3, 3), padding=1),
            #nn.Dropout(p=0.5),
            nn.BatchNorm2d(8),
            nn.ReLU(),

            # Layer 2
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), padding=1),
            #nn.Dropout(p=0.5),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),

            # Layer 3
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=1),
            # nn.Dropout(p=0.5),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            # Layer 4
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=1),
            # nn.Dropout(p=0.5),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            # Layer 5
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1),
            # nn.Dropout(p=0.5),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # Layer 6
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1),
            # nn.Dropout(p=0.5),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # Layer 7
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1),
            # nn.Dropout(p=0.5),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),

            # Layer 8
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1),
            #nn.Dropout(p=0.5),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # Layer 9
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1),
            #nn.Dropout(p=0.5),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # Layer 10
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1),
            #nn.Dropout(p=0.5),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        )

        # Linear units
        self.linear = nn.Sequential(
            nn.Linear(64*8*8, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 2),
            #nn.Softmax()
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 64*8*8)
        x = self.linear(x)
        return x


class imCNN1(nn.Module):
    """Convnet Classifier"""

    def __init__(self):
        super(imCNN1, self).__init__()
        self.conv = nn.Sequential(
            # Layer 1
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding=1),
            #nn.Dropout(p=0.5),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),

            # Layer 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1),
            #nn.Dropout(p=0.5),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),

            # Layer 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1),
            # nn.Dropout(p=0.5),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            # Layer 4
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=1),
            # nn.Dropout(p=0.5),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            # Layer 5
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=1),
            # nn.Dropout(p=0.5),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),

            # Layer 6
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=1),
            # nn.Dropout(p=0.5),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            # Layer 7
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1),
            # nn.Dropout(p=0.5),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),

            # Layer 8
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1),
            #nn.Dropout(p=0.5),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            # Layer 9
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1),
            #nn.Dropout(p=0.5),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        )

        # Linear units
        self.linear = nn.Sequential(
            nn.Linear(512*2*2, 512*2*2),
            nn.BatchNorm1d(512*2*2),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512*2*2, 512*2*2),
            nn.BatchNorm1d(512*2*2),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512*2*2, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1000, 2),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 512*2*2)
        x = self.linear(x)
        return x
