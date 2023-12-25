import torch
import torch.nn as nn
import torch.nn.functional as F

# define the CNN architecture
class MyModel(nn.Module):
    def __init__(self, num_classes: int = 50, dropout: float = 0.7) -> None:

        super().__init__()

        # (3x224x224) RGB images input
        self.conv1 = nn.Conv2d(3, 6, 3, padding=1) # paddding=1 to keep x,y dimensions stable with kernel_size=3
        # (6x112x112) 
        self.conv1_bn = nn.BatchNorm2d(6)

        self.conv2 = nn.Conv2d(6, 16, 3, padding=1)
        # (16x56x56)
        self.conv2_bn = nn.BatchNorm2d(16)

        self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
        # 32x28x28
        self.conv3_bn = nn.BatchNorm2d(32)

        self.conv4 = nn.Conv2d(32, 64, 3, padding=1)
        # 64x14x14
        self.conv4_bn = nn.BatchNorm2d(64)

        self.conv5 = nn.Conv2d(64, 128, 3, padding=1)
        # 128x7x7
        self.conv5_bn = nn.BatchNorm2d(128)

        # pooling layers to half x,y dimensions per layer
        self.pool = nn.MaxPool2d(2, 2)



        # fc- layers 
        self.fc1 = nn.Linear(128*7*7, 1024)
        self.fc1_bn = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, 512)
        self.fc2_bn = nn.BatchNorm1d(512)

        self.fc3 = nn.Linear(512, num_classes)

        # dropout
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.conv1(x)
        x = self.pool(F.relu(self.conv1_bn(x)))


        x = self.conv2(x)
        x = self.pool(F.relu(self.conv2_bn(x)))

        x = self.conv3(x)
        x = self.pool(F.relu(self.conv3_bn(x)))

        x = self.conv4(x)
        x = self.pool(F.relu(self.conv4_bn(x)))

        x = self.conv5(x)
        x = self.pool(F.leaky_relu(self.conv5_bn(x)))

        # flatten batch_size
        x = x.view(-1, 128*7*7) # adapt dimension 1 to forced dimension 2o of tensor 

        x = self.dropout(x)
        x = self.fc1(x)

        x = F.relu(self.fc1_bn(x))

        x = self.dropout(x)
        x = self.fc2(x)
        x = F.relu(self.fc2_bn(x))

        x = self.dropout(x)
        x = self.fc3(x)
        return x


######################################################################################
#                                     TESTS
######################################################################################
#import pytest


#@pytest.fixture(scope="session")
#def data_loaders():
    #from .data import get_data_loaders

    #return get_data_loaders(batch_size=2)


#def test_model_construction(data_loaders):

    #model = MyModel(num_classes=23, dropout=0.3)

    #dataiter = iter(data_loaders["train"])
    #images, labels = dataiter.next()

    #out = model(images)

    #assert isinstance(out, torch.Tensor), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    #assert out.shape == torch.Size([2, 23]), f"Expected an output tensor of size (2, 23), got {out.shape}"
