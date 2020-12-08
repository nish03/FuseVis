import os
import natsort
import glob
import numpy as np
import imageio
from FuseVis._init_ import _init_


image_length, image_width, gray_channels, device = _init_()
def test_input1(image_width,image_length,device):
    filenames = os.listdir('/home/......./Testing/Input1/')
    dataset = os.path.join(os.getcwd(), '/home/......./Testing/Input1/')
    data = glob.glob(os.path.join(dataset, "*.gif"))
    data = natsort.natsorted(data,reverse=False)
    input1 = np.zeros((len(data), image_width,image_length))
    for i in range(len(data)):
        input1[i,:,:] =(imageio.imread(data[i]))
        input1[i,:,:] =(input1[i,:,:] - np.min(input1[i,:,:])) / (np.max(input1[i,:,:]) - np.min(input1[i,:,:]))
        input1[i,:,:] = np.float32(input1[i,:,:])
    
    input1 = np.expand_dims(input1,axis=1)
    #convert the input image into a pytorch tensor
    input1_tensor = torch.from_numpy(input1).float()
    input1_tensor = input1_tensor.to(device)
    input1_tensor.requires_grad =True
    return input1, input1_tensor


def test_input2(image_width,image_length,device):
    filenames = os.listdir('/home/......./Testing/Input2/')
    dataset = os.path.join(os.getcwd(), '/home/......./Testing/Input2/')
    data = glob.glob(os.path.join(dataset, "*.gif"))
    data = natsort.natsorted(data,reverse=False)
    input2 = np.zeros((len(data),image_width,image_length))
    for i in range(len(data)):
        input2[i,:,:] =(imageio.imread(data[i]))
        input2[i,:,:] =(input2[i,:,:] - np.min(input2[i,:,:])) / (np.max(input2[i,:,:]) - np.min(input2[i,:,:]))
        input2[i,:,:] = np.float32(input2[i,:,:])
    
    input2 = np.expand_dims(input2,axis=1)
    #convert the input image into a pytorch tensor
    input2_tensor = torch.from_numpy(input2).float()
    input2_tensor = input2_tensor.to(device)
    input2_tensor.requires_grad =True
    return input2, input2_tensor