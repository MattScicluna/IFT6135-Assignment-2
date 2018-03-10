import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class imgCNN(nn.Module):
    def __init__(self):
        super(imgCNN, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(3, 20, 3)
        torch.nn.init.xavier_uniform(self.conv1.weight)
        self.bn1 = nn.BatchNorm2d(20)

        self.conv2 = nn.Conv2d(20, 20, 3)
        torch.nn.init.xavier_uniform(self.conv2.weight)

        self.conv3 = nn.Conv2d(20, 40, 3)
        self.bn2 = nn.BatchNorm2d(40)
        torch.nn.init.xavier_uniform(self.conv3.weight)
        self.conv4 = nn.Conv2d(40, 40, 3)

        self.conv5 = nn.Conv2d(40, 80, 3)
        self.bn3 = nn.BatchNorm2d(80)
        self.conv6 = nn.Conv2d(80, 80, 3)
        self.conv7 = nn.Conv2d(80, 80, 3)

        self.conv8 = nn.Conv2d(80, 160, 3)
        self.bn4 = nn.BatchNorm2d(160)

        self.fc1 = nn.Linear(160, 160)
        self.bn5 = nn.BatchNorm1d(160)
        self.fc2 = nn.Linear(160, 50)
        self.bn6 = nn.BatchNorm1d(50)
        self.fc3 = nn.Linear(50, 2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(F.relu(self.bn1(self.conv2(x))))
        x = F.relu(self.bn2(self.conv3(x)))
        x = self.pool(F.relu(self.bn2(self.conv4(x))))
        x = F.relu(self.bn3(self.conv5(x)))
        x = F.relu(self.bn3(self.conv6(x)))
        x = self.pool(F.relu(self.bn3(self.conv7(x))))
        x = F.relu(self.bn4(self.conv8(x)))
        x = x.view(-1, 160)
        x = F.relu(self.bn5(self.fc1(x)))
        x = F.relu(self.bn6(self.fc2(x)))
        x = self.fc3(x)
        return x