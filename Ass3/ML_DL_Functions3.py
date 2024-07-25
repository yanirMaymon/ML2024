import numpy as np
import torch
import torch.nn as nn
def ID1():
    '''
        Personal ID of the first student.
    '''
    # Insert your ID here
    return 000000000

def ID2():
    '''
        Personal ID of the second student. Fill this only if you were allowed to submit in pairs, Otherwise leave it zeros.
    '''
    # Insert your ID here
    return 000000000

class CNN(nn.Module):
    def __init__(self): # Do NOT change the signature of this function
        super(CNN, self).__init__()
        n = ...
        kernel_size = ...
        padding = ...
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=n,kernel_size=kernel_size,padding=padding),
        # TODO: complete this method
    
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
        # TODO: complete this function
        return ...

class CNNChannel(nn.Module):
    def __init__(self):# Do NOT change the signature of this function
        super(CNNChannel, self).__init__()
        # TODO: complete this method

    # TODO: complete this class
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
        # TODO: complete this function
        return ...