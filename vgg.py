import torch 
import torch.nn as nn 
import torchvision.datasets as datasets
from torchvision.transforms import transforms 
import torch.nn.functional as F 
import torch.optim as optim

class VGG(nn.Module):
    def __init__(self, num_classes = 1000):
        super(VGG, self).__init__()
        
        self.num_classes = num_classes
        self.relu = nn.ReLU()
        self.conv11 = nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = (3,3), stride = (1,1), padding = (1,1))
        self.conv12 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (3,3), stride = (1,1), padding = (1,1))

        self.conv21 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = (3,3), stride = (1,1), padding = (1,1))
        self.conv22 = nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = (3,3), stride = (1,1), padding = (1,1))


        ##third layer 
        self.conv31 = nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = (3,3), stride = (1,1), padding = (1,1))
        self.conv32 = nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = (3,3), stride = (1,1), padding = (1,1))
        self.conv33 = nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = (3,3), stride = (1,1), padding = (1,1))


        ##fourth layer 
        self.conv41 = nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = (3,3), stride = (1,1), padding = (1,1))
        self.conv42 = nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = (3,3), stride = (1,1), padding = (1,1))
        self.conv43 = nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = (3,3), stride = (1,1), padding = (1,1))


        ##fifth layer
        self.conv51 = nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = (3,3), stride = (1,1), padding = (1,1))
        self.conv52 = nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = (3,3), stride = (1,1), padding = (1,1))
        self.conv53 = nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = (3,3), stride = (1,1), padding = (1,1))


        # max pooling (kernel_size, stride)
        self.pool = nn.MaxPool2d(kernel_size = (2,2), stride = (2,2), padding = (0,0))

        # fully conected layers:
        self.fc1 = nn.Linear(7*7*512, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 1000)

        

    def forward(self, x, training=True):
        x = self.relu(self.conv11(x))
        x = self.relu(self.conv12(x))
        x = self.pool(x)

        x = self.relu(self.conv21(x))
        x = self.relu(self.conv22(x))
        x = self.pool(x)

        x = self.relu(self.conv31(x))
        x = self.relu(self.conv32(x))
        x = self.relu(self.conv33(x))
        x = self.pool(x)

        x = self.relu(self.conv41(x))
        x = self.relu(self.conv42(x))
        x = self.relu(self.conv43(x))
        x = self.pool(x)

        x = self.relu(self.conv51(x))
        x = self.relu(self.conv52(x))
        x = self.relu(self.conv53(x))
        x = self.pool(x)

        x = x.reshape(x.shape[0], -1)

        x = F.relu(self.fc1(x))
        x = F.dropout(x, 0.5, training=training)

        x = F.relu(self.fc2(x))
        x = F.dropout(x, 0.5, training=training)

        x = self.fc3(x)

        return x



model = VGG(num_classes = 1000)
print(model)
x = torch.randn(1,3,224,224)
print(model(x).shape)
        
        




