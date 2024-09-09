import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
def ID1():
    '''
        Personal ID of the first student.
    '''
    # Insert your ID here
    return 315333997

def ID2():
    '''
        Personal ID of the second student. Fill this only if you were allowed to submit in pairs, Otherwise leave it zeros.
    '''
    # Insert your ID here
    return 000000000

class CNN(nn.Module):
    def __init__(self): # Do NOT change the signature of this function
        super(CNN, self).__init__()
        self.n = 6
        kernel_size = 5
        padding = int((kernel_size - 1) / 2)
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=self.n,kernel_size=kernel_size,padding=padding)
        self.conv2 = nn.Conv2d(in_channels=self.n,out_channels=2*self.n,kernel_size=kernel_size,padding=padding)
        self.conv4 = nn.Conv2d(in_channels=2*self.n,out_channels=4*self.n,kernel_size=kernel_size,padding=padding)
        self.conv8 = nn.Conv2d(in_channels=4*self.n,out_channels=8*self.n,kernel_size=kernel_size,padding=padding)
        self.fc1 = nn.Linear(28 * 14 * 8 * self.n,100)
        self.fc2 = nn.Linear(100,2)

    def forward(self,inp):# Do NOT change the signature of this function
        '''
          prerequests:
          parameter inp: the input image, pytorch tensor.
          inp.shape == (N,3,448,224):
            N   := batch size
            3   := RGB channels
            448 := Height
            224 := Width
          
          return output, pytorch tensor
          output.shape == (N,2):
            N := batch size
            2 := same/different pair
        '''
        out = self.conv1(inp)
        out = F.relu(out)
        out = F.max_pool2d(out, kernel_size=2)

        out = self.conv2(out)
        out = F.relu(out)
        out = F.max_pool2d(out, kernel_size=2)

        out = self.conv4(out)
        out = F.relu(out)
        out = F.max_pool2d(out, kernel_size=2)

        out = self.conv8(out)
        out = F.relu(out)
        out = F.max_pool2d(out, kernel_size=2)

        out = out.reshape(-1, 8 * self.n * 28 * 14)
        out = self.fc1(out)
        out = F.relu(out)

        out = self.fc2(out)
        out = F.softmax(out, dim=1)

        return out

class CNNChannel(nn.Module):
    def __init__(self):# Do NOT change the signature of this function
        super(CNNChannel, self).__init__()
        self.n = 6
        kernel_size = 5
        padding = int((kernel_size - 1) / 2)
        self.conv1 = nn.Conv2d(in_channels=6,out_channels=self.n,kernel_size=kernel_size,stride=1,padding=padding)
        self.conv2 = nn.Conv2d(in_channels=self.n,out_channels=2*self.n,kernel_size=kernel_size,stride=1,padding=padding)
        self.conv4 = nn.Conv2d(in_channels=2*self.n,out_channels=4*self.n,kernel_size=kernel_size,stride=1,padding=padding)
        self.conv8 = nn.Conv2d(in_channels=4*self.n,out_channels=8*self.n,kernel_size=kernel_size,stride=1,padding=padding)
        self.fc1 = nn.Linear(14 * 14 * 8 * self.n,100)
        self.fc2 = nn.Linear(100,2)

    def forward(self,inp):# Do NOT change the signature of this function
        '''
          prerequests:
          parameter inp: the input image, pytorch tensor
          inp.shape == (N,3,448,224):
            N   := batch size
            3   := RGB channels
            448 := Height
            224 := Width
          
          return output, pytorch tensor
          output.shape == (N,2):
            N := batch size
            2 := same/different pair
        '''
        # TODO start by changing the shape of the input to (N,6,224,224)
        inp = torch.cat([inp[:, :, :224, :], inp[:, :, 224:, :]], dim=1)

        # TODO: complete this function
        out = self.conv1(inp)
        out = F.relu(out)
        out = F.max_pool2d(out, kernel_size=2)

        out = self.conv2(out)
        out = F.relu(out)
        out = F.max_pool2d(out, kernel_size=2)

        out = self.conv4(out)
        out = F.relu(out)
        out = F.max_pool2d(out, kernel_size=2)

        out = self.conv8(out)
        out = F.relu(out)
        out = F.max_pool2d(out, kernel_size=2)

        out = out.reshape(-1, 8 * self.n * 14 * 14)
        out = self.fc1(out)
        out = F.relu(out)

        out = self.fc2(out)
        out = F.softmax(out, dim=1)

        return out
