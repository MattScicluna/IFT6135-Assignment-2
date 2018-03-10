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
            nn.Linear(144 * 4 * 4, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(),
            nn.Linear(500, 2)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 144*4*4)
        x = self.linear(x)
        return x
