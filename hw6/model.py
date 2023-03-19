# from ComputationalGraphPrimer import *
import random
import torch
import torch.nn as nn 
from sklearn.metrics import confusion_matrix
from typing import Any, Callable, List, Optional, Type, Union

MIN_W = 200
MIN_H = 200
ROOT = "."

class ResnetBlock(nn.Module):
    """
    Inspired by the original implementation in pytorch github
    """
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        outplanes: int,
        downsample: Union[str, bool] = None
    ) -> None:
        super().__init__()
        
        norm_layer = nn.BatchNorm2d
        self.downsample = downsample

        self.conv1 = nn.Conv2d(inplanes, outplanes, kernel_size=3, padding=1)
        self.bn1 = norm_layer(outplanes)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(outplanes, outplanes, kernel_size=3, padding=1, stride = (1 if downsample in (False, None) else 2))
        self.bn2 = norm_layer(outplanes)

        if self.downsample == True:
            self.downsample = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=2)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)

        return out

IM_SIZE = 256

class HW5Net(nn.Module):
    """
    Resnet-based encoder that consists of a few downsampling + several Resnet blocks as the backbone and two prediction heads.
    """
    def __init__(self, input_nc, ngf=8, n_blocks=4, classes=3, anchors = 5):
        """ Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images)
        ngf (int) -- the number of filters first conv layer
        n_blocks (int) -- teh number of ResNet blocks
        """
        self.classes = classes
        self.anchors = anchors

        assert (n_blocks >= 0) 
        super(HW5Net, self).__init__() 
        # The first conv layer
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True)
        ]
        # Add downsampling layers
        n_downsampling_1 = 3
        mult = 1
        for _ in range(n_downsampling_1):
            model += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(ngf * mult * 2),
                nn.ReLU(True)
            ] 
            mult*=2

        # Add your own ResNet blocks
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, ngf * mult)] 

        # Add downsampling layers with ResNet
        n_downsampling_2 = 2
        for _ in range(n_downsampling_2):
            model += [
                # nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                # nn.BatchNorm2d(ngf * mult * 2),
                # nn.ReLU(True),
                ResnetBlock(ngf * mult, ngf * mult * 2, downsample=True)
            ] 
            mult*=2

        # Generate final model
        self.model = nn.Sequential(*model) 
        # Head for generating the output of the network 
        head = [
            nn.Conv2d(ngf * mult, (classes + 5) * anchors, kernel_size=1, stride=1, padding=0)
        ]
        self.head = nn.Sequential(*head) # The bounding box regression head

    def forward(self, input):
        ft = self.model(input) 
        # print("pre ", ft[0, 0, 0,...])
        out = self.head(ft).view(-1, self.anchors, 8, 8, 5 + self.classes)
        return out
    

if __name__ == "__main__":
    # Verify NN structure 
    model = HW5Net(3).to(torch.float32)
    test = torch.rand(16, 3, 256, 256, dtype=torch.float32)
    # summary(model, input_size=(16, 3, 256, 256))
    outputs = model(test)

    print(len(list(model.parameters())))
    print(outputs[0, 0, 0,...])

    print(model(test).size(), model(test).size(), model(test).dtype)