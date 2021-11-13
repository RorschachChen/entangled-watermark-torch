from torch import nn
import torch.nn.functional as F

from models.ResNet import ResNet18
from models.backbone import BaseModel


class EWE_2_conv(BaseModel):
    """
    2 conv + 2 fc
    """
    def __init__(self, input_channel, output_channel, factors, metric, temperatures):
        super(EWE_2_conv, self).__init__(input_channel, output_channel, factors, metric, temperatures)
        self.conv1 = nn.Conv2d(in_channels=input_channel, out_channels=32, kernel_size=(5, 5))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3))
        self.fc1 = nn.Linear(1600, 128)
        self.fc2 = nn.Linear(128, output_channel)
        self.layers = [0, 1, 2]

    def forward(self, x, per=False):
        B, H, W, C = x.shape
        if per:
            x = x.permute(0, 3, 1, 2)
        x = self.conv1(x)
        s1 = x
        x = F.relu(x)
        x = F.max_pool2d(x, (2, 2), 2)
        x = F.dropout(x)
        x = self.conv2(x)
        s2 = x
        x = F.relu(x)
        x = F.max_pool2d(x, (2, 2), 2)
        x = F.dropout(x)
        x = x.contiguous().view(B, -1)
        x = self.fc1(x)
        x = F.dropout(x)
        s3 = x
        x = F.relu(x)
        x = self.fc2(x)
        return [s1, s2, s3, x]


class EWE_LSTM(BaseModel):
    def __init__(self, input_channel, output_channel, factors, metric, temperatures):
        super(EWE_LSTM, self).__init__(input_channel, output_channel, factors, metric, temperatures)
        self.fc1 = nn.Linear(input_channel, 128)
        self.fc2 = nn.Linear(128, output_channel)
        self.lstm = nn.LSTM(input_channel, hidden_size=256)
        self.layers = [-4, -3, -2]

    def forward(self, x):
        res = []
        states, x = self.lstm(x)
        res.append(x[int(len(x) // 2)])
        x = x[-1]
        res.append(x)
        x = self.fc1(x)
        res.append(x)
        x = F.relu(x)
        x = self.fc2(x)
        res.append(x)
        return res


class EWE_ResNet(BaseModel):
    __metaclass__ = ResNet18()
    def __init__(self, input_channel, output_channel, factors, metric, temperatures, layers):
        super(EWE_ResNet, self).__init__(input_channel, output_channel, factors, metric, temperatures, layers)
        self.layers = [-1, -2, -3]
        self.resnet = ResNet18()

    def forward(self, x):
        self.resnet.cuda()
        x = x.permute(0, 3, 1, 2)
        res = []
        out = self.resnet.convbnrelu_1(x)
        out = self.resnet.layer1(out)
        out = self.resnet.layer2(out)
        out = self.resnet.layer3(out)
        res.append(out)
        maxlen = len(self.resnet.layer4)
        for i in range(maxlen):
            out = self.resnet.layer4[i](out)
            if i < maxlen - 1:
                res.append(out)
            i += 1
        out = F.avg_pool2d(out, 4)
        res.append(out)
        out = out.view(out.size(0), -1)
        out = self.resnet.linear(out)
        self.activations = res
        return out


class Plain_2_conv(BaseModel):
    def __init__(self, input_channel, output_channel):
        super(Plain_2_conv, self).__init__(input_channel, output_channel)
        self.conv1 = nn.Conv2d(in_channels=input_channel, out_channels=32, kernel_size=(5, 5))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3))
        self.fc1 = nn.Linear(1600, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x, per=False):
        B, H, W, C = x.shape
        if per:
            x = x.permute(0, 3, 1, 2)
        conv1 = self.conv1(x)
        x = F.max_pool2d(conv1, (2, 2), 2)
        x = F.dropout(x)
        conv2 = self.conv2(x)
        x = F.max_pool2d(conv2, (2, 2), 2)
        x = F.dropout(x)
        x = x.contiguous().view(B, -1)
        x = self.fc1(x)
        drop = F.dropout(x)
        x = self.fc2(drop)
        return x


class Plain_LSTM(BaseModel):
    def __init__(self, input_channel, output_channel):
        super().__init__(input_channel, output_channel)
        self.fc1 = nn.Linear(input_channel, 128)
        self.fc2 = nn.Linear(128, output_channel)
        self.lstm = nn.LSTM(input_channel, hidden_size=256)

    def forward(self, x):
        states, x = self.lstm(x)
        x = x[-1]
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


@DeprecationWarning
class Plain_ResNet(BaseModel):
    def __init__(self, input_channel, output_channel):
        super(Plain_ResNet, self).__init__(input_channel, output_channel)
