#import generic packages
import time
import torch.nn as nn
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
from skimage import img_as_ubyte
import torch.utils.data as Data
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import scipy
import argparse
import glob
import imageio
from skimage import color
import natsort
import functools
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
import pprint
from scipy.ndimage import correlate
from scipy.ndimage.filters import gaussian_gradient_magnitude
import torchvision.datasets as dset
import torch.utils.data as data
import os
import os.path
from tkinter import *
import tkinter as tk
import tkinter.font as tkFont
from PIL import ImageTk, Image, ImageDraw
import pylab
import cv2
import h5py
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

#import project files
from FuseVis._init_ import _init_
from FuseVis.preprocess import test_input1, test_input2
from FuseVis.networks import Weighted_Averaging, FunFuseAn, MaskNet, DeepFuse, DeepPedestrian

#import the image dimensions and GPU 
image_length, image_width, gray_channels, device = _init_()

#import test images and tensors 
input1, input1_tensor = test_input1()
input2, input2_tensor = test_input2()


#define weights for the Weighted Averaging Model
w1 = np.zeros((image_width,image_length),dtype = float)
w2 = np.zeros((image_width,image_length),dtype = float)

for i in range(0,image_width):
    for j in range(0,image_length):
        if input1[0,0,i,j] == input2[0,0,i,j]:
            w1[i,j] = 0.5
            w2[i,j] = 0.5
        else:
            w1[i,j] = input1[0,0,i,j] / (input1[0,0,i,j] +input2[0,0,i,j])
            w2[i,j] = input2[0,0,i,j] / (input1[0,0,i,j] +input2[0,0,i,j])
                    
#convert the weights of weighted averaging method as a pytorch tensor
w1_tensor = torch.from_numpy(w1).float().to(device)
w2_tensor = torch.from_numpy(w2).float().to(device)
model0 = Weighted_Averaging().to(device)
model0 = model0.float()
print(model0)

#import other neural network models
model1 = FunFuseAn().to(device)
model1 = model1.float()
print(model1)


model4 = MaskNet().to(device)
model4 = model4.float()
print(model4)


model5 = DeepFuse().to(device)
model5 = model5.float()
print(model5)


model6 = DeepPedestrian().to(device)
model6 = model6.float()
print(model6)



#define the user interface window
root = Tk()  
root.title('FuseVis')
root.configure(background='white')
default_font = tkFont.nametofont("TkDefaultFont")
default_font.configure(size=12)

#define the frame
canvasframe = Frame(root)  # define Input and output frame
buttonframe = Frame(root)  # define button frame

canvasframe.pack()  # pack the Input and Output frame
buttonframe.pack()  # pack the button frame

#define the canvas
canvas = Canvas(canvasframe, width=1805, height=940, bg = 'white')
canvas.grid(row=0, column=0)


def start_mouseover(fused_tensor, model, min_, max_, guide_fused_mri, guide_fused_pet,fused_tensor_, fused_RGB_, id_,gamma,fused_numpy_norm,gamma1): 
    # function called when user clicks the button 
    # link the function to the left-mouse-click event
    obj = ButtonObject(fused_tensor, model, min_, max_, guide_fused_mri, guide_fused_pet,fused_tensor_, fused_RGB_, id_,gamma,fused_numpy_norm,gamma1)
    canvas.bind("<B1-Motion>", obj.mouseover_Callback)
    var1 = DoubleVar(root)
    var1.set(1)
    var = DoubleVar(root)
    var.set(1)
    slide1 = Scale(buttonframe, variable = var1, from_=0.1, to=2,resolution=0.1,length=200,repeatdelay=50,orient=HORIZONTAL, command = obj.slider_Callback1, label = '                     Gamma1')
    slide1.grid(row=1, column=0, pady=0)
    slide = Scale(buttonframe, variable = var, from_=0.1, to=2,resolution=0.1,length=200,repeatdelay=50,orient=HORIZONTAL, command = obj.slider_Callback, label = '                     Gamma2')
    slide.grid(row=1, column=1, pady=0)


   
class ButtonObject:
    def __init__(self, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9,arg10,arg13,arg14):
        self.arg1 =  arg1
        self.arg2 =  arg2
        self.arg3 =  arg3
        self.arg4 =  arg4
        self.arg5 =  arg5
        self.arg6 =  arg6
        self.arg7 =  arg7
        self.arg8 =  arg8
        self.arg9 =  arg9
        self.arg10 = arg10
        self.arg11 = 0
        self.arg12 = 0
        self.arg13 = arg13
        self.arg14 = arg14
    def slider_Callback1(self,slider_event):
        self.arg14 = float(slider_event)
        print(self.arg14)
        if self.arg11 >= 0 and self.arg11 <= 255 and self.arg12 >= 0 and self.arg12 <= 255:

            #time1 = time.time()
            #print('mouse position is at' + '(' + str(self.arg12) + ',' + str(self.arg11) + ')', end='\r')
            #display the output MRI Jacobian image
            jacobian_fuse_mri = torch.autograd.grad(self.arg1[0,0,self.arg12,self.arg11], input1_tensor, retain_graph=True, create_graph=True)[0]
            jacobian_fuse_pet = torch.autograd.grad(self.arg1[0,0,self.arg12,self.arg11], input2_tensor, retain_graph=True, create_graph=True)[0]

            jacob_ = torch.autograd.grad(self.arg7[0,0,self.arg12,self.arg11], input1_tensor, retain_graph=True, create_graph=True)[0]

            jacob_val_mri = np.squeeze(jacobian_fuse_mri.data.cpu().numpy())    
            jacob_val_pet = np.squeeze(jacobian_fuse_pet.data.cpu().numpy())
            jacob_val_numpy = np.squeeze(jacob_.data.cpu().numpy())
            

            x_mri = np.asarray(np.where(np.any(jacob_val_numpy, axis = 0)))
            y_mri = np.asarray(np.where(np.any(jacob_val_numpy, axis = 1)))
            minx_mri, maxx_mri, miny_mri, maxy_mri = np.min(x_mri), np.max(x_mri), np.min(y_mri), np.max(y_mri)  #return min and max coordinates
               
            radius = 7
            i = canvas.create_rectangle(self.arg11-radius, self.arg12-radius, self.arg11+radius, self.arg12+radius, outline = 'red')
           
            j = canvas.create_rectangle(self.arg11-radius, self.arg12+650-radius, self.arg11+radius, self.arg12+650+radius, outline = 'red')
        
            m = canvas.create_rectangle(self.arg11+600-radius, self.arg12+320-radius, self.arg11+600+radius, self.arg12+320+radius, outline = 'red') 
        
            n = canvas.create_rectangle(self.arg11-radius, self.arg12+320-radius, self.arg11+radius, self.arg12+320+radius, outline = 'red')
            
            if i > self.arg9+1:
                canvas.delete(i-24)
                canvas.delete(j-24)
                canvas.delete(m-24)
                canvas.delete(n-24)

            #color map details for the zoom images
            cmap = plt.get_cmap('viridis')
            #cNorm = mpl.colors.Normalize(vmin=self.arg3, vmax=self.arg4) #re-wrapping normalization
            cNorm_jacob = mpl.colors.PowerNorm(gamma=self.arg14, vmin=self.arg3, vmax=self.arg4)
            cNorm_guide = mpl.colors.PowerNorm(gamma=self.arg10, vmin=self.arg3, vmax=self.arg4)
            scalarMap_guide = mpl.cm.ScalarMappable(norm=cNorm_guide, cmap=cmap)
            scalarMap_jacob = mpl.cm.ScalarMappable(norm=cNorm_jacob, cmap=cmap)
        
            path = 'C:/Users/........./FuseVis/Guidance images/'

            im_fused = Image.fromarray(np.uint8(self.arg13[miny_mri:maxy_mri,minx_mri:maxx_mri]*255)).resize((256,256))
            im_fused = im_fused.convert("RGB")
            draw = ImageDraw.Draw(im_fused)
            draw.rectangle(((128, 128), (141, 141)), outline="Red")        
            im_out7 = ImageTk.PhotoImage(image=im_fused)
            canvas.create_image(300,320,image=im_out7,anchor=NW)
            canvas.image7 = im_out7
            #plt.tight_layout()
    
            im_mri = Image.fromarray(np.uint8(input1[0,0,miny_mri:maxy_mri,minx_mri:maxx_mri]*255)).resize((256,256))
            im_mri = im_mri.convert("RGB")
            draw = ImageDraw.Draw(im_mri)
            draw.rectangle(((128, 128), (141, 141)), outline="Red")
            im_out8 = ImageTk.PhotoImage(image=im_mri)
            canvas.create_image(300,0,image=im_out8,anchor=NW)
            canvas.image8 = im_out8
    
            im_pet = Image.fromarray(np.uint8(input2[0,0,miny_mri:maxy_mri,minx_mri:maxx_mri]*255)).resize((256,256)) 
            im_pet = im_pet.convert("RGB")
            draw = ImageDraw.Draw(im_pet)
            draw.rectangle(((128, 128), (141, 141)), outline="Red")
            im_out9 = ImageTk.PhotoImage(image=im_pet)
            canvas.create_image(300,650,image=im_out9,anchor=NW)
            canvas.image9 = im_out9  

            im_out10 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(255*(scalarMap_jacob.to_rgba(jacob_val_mri)))).resize((256,256)))
            canvas.create_image(1200,0,image=im_out10,anchor=NW)
            canvas.image10 = im_out10
        
            o = canvas.create_rectangle(self.arg11+1200-radius, self.arg12-radius, self.arg11+1200+radius, self.arg12+radius, outline = 'red')
            if i > self.arg9+1:
                canvas.delete(o-24)

            im_out11 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(255*(scalarMap_jacob.to_rgba(jacob_val_mri[miny_mri:maxy_mri,minx_mri:maxx_mri])))).resize((256,256)))
            canvas.create_image(1500,0,image=im_out11,anchor=NW)
            canvas.image11 = im_out11
        
            im_out13 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(255*scalarMap_jacob.to_rgba(jacob_val_pet))).resize((256,256)))
            canvas.create_image(1200,650,image=im_out13,anchor=NW)       
            canvas.image13 = im_out13
        
                
            p = canvas.create_rectangle(self.arg11+1200-radius, self.arg12+650-radius, self.arg11+1200+radius, self.arg12+650+radius, outline = 'red')
            if i > self.arg9+1:
                canvas.delete(p-24)
               
    
            im_out14 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(255*scalarMap_jacob.to_rgba(jacob_val_pet[miny_mri:maxy_mri,minx_mri:maxx_mri]))).resize((256,256)))
            canvas.create_image(1500,650,image=im_out14,anchor=NW)
            canvas.image14 = im_out14
    
            
            #display the zoom guidance MRI images 
            im_guide_mri = Image.fromarray(np.uint8(255*scalarMap_guide.to_rgba(self.arg5[miny_mri:maxy_mri,minx_mri:maxx_mri]))).resize((256,256))
            draw = ImageDraw.Draw(im_guide_mri)
            draw.rectangle(((128, 128), (141, 141)), outline="Red")
            im_out15 = ImageTk.PhotoImage(image=im_guide_mri)
            canvas.create_image(900,0,image=im_out15,anchor=NW)
            canvas.image15 = im_out15
        
        
            #display the zoom guidance PET images 
            im_guide_pet = Image.fromarray(np.uint8(255*scalarMap_guide.to_rgba(self.arg6[miny_mri:maxy_mri,minx_mri:maxx_mri]))).resize((256,256))
            draw = ImageDraw.Draw(im_guide_pet)
            draw.rectangle(((128, 128), (141, 141)), outline="Red")
            im_out16 = ImageTk.PhotoImage(image=im_guide_pet)
            canvas.create_image(900,650,image=im_out16,anchor=NW)
            canvas.image16 = im_out16
        
        
            #display the zoom fused RGB images
            im_fused_RGB = Image.fromarray(np.uint8(self.arg8[miny_mri:maxy_mri,minx_mri:maxx_mri]*255)).resize((256,256))
            #im_guide_pet = im_guide_pet.convert("RGB")
            draw = ImageDraw.Draw(im_fused_RGB)
            draw.rectangle(((128, 128), (141, 141)), outline="Red")
            im_out17 = ImageTk.PhotoImage(image=im_fused_RGB)
            canvas.create_image(900,320,image=im_out17,anchor=NW)
            canvas.image17 = im_out17 
            
            im_out2 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(255*scalarMap_guide.to_rgba(self.arg5))).resize((256,256)))
            canvas.create_image(600,0,image=im_out2,anchor=NW)
            canvas.image2 = im_out2
            
            k = canvas.create_rectangle(self.arg11+600-radius, self.arg12-radius, self.arg11+600+radius, self.arg12+radius, outline = 'red')
            if i>self.arg9+1:
                canvas.delete(k-24)        
        
            im_out3 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(255*scalarMap_guide.to_rgba(self.arg6))).resize((256,256)))
            canvas.create_image(600,650,image=im_out3,anchor=NW)
            canvas.image3 = im_out3   
            
            l = canvas.create_rectangle(self.arg11+600-radius, self.arg12+650-radius, self.arg11+600+radius, self.arg12+650+radius, outline = 'red')
            if i>self.arg9+1:
                canvas.delete(l-24)
          
            fig = plt.figure(figsize=(0.2, 23))
            ax1 = fig.add_axes([0.05, 0.80, 0.9, 0.15])
            cb1 = mpl.colorbar.ColorbarBase(ax1, cmap=cmap,norm=cNorm_jacob,orientation='vertical')
            plt.plot()
            plt.savefig('C:/Users/........./FuseVis/Guidance images/colorbar.png', bbox_inches = 'tight',pad_inches = 0)
            plt.close()
            im_out50 = ImageTk.PhotoImage(file ='C:/Users/........./FuseVis/Guidance images/colorbar.png')
            
            fig2 = plt.figure(figsize=(0.2, 23))
            ax2 = fig2.add_axes([0.05, 0.80, 0.9, 0.15])
            cb2 = mpl.colorbar.ColorbarBase(ax2, cmap=cmap,norm=cNorm_guide,orientation='vertical')
            plt.plot()
            plt.savefig('C:/Users/........./FuseVis/Guidance images/colorbar.png', bbox_inches = 'tight',pad_inches = 0)
            plt.close()
            im_out51 = ImageTk.PhotoImage(file ='C:/Users/........./FuseVis/Guidance images/colorbar.png')
            
            canvas.create_image(1152,650,image=im_out51,anchor=NW)
            canvas.image50 = im_out51
            
            canvas.create_image(1752,650,image=im_out50,anchor=NW)
            canvas.image51 = im_out50
        
            canvas.create_image(1752,4,image=im_out50,anchor=NW)
            canvas.image52 = im_out50  
        
            canvas.create_image(1152,4,image=im_out51,anchor=NW)
            canvas.image53 = im_out51  
            
            #time2=time.time()
            #print(time2-time1)
            
        elif self.arg11 >= 0 and self.arg11 <= 255 and self.arg12 >= 320 and self.arg12 <= 575:
            #time1 = time.time()
            print('mouse position is at' + '(' + str(self.arg12-320) + ',' + str(self.arg11) + ')', end='\r')
            #display the output MRI Jacobian image
            jacobian_fuse_mri = torch.autograd.grad(self.arg1[0,0,self.arg12-320,self.arg11], input1_tensor, retain_graph=True, create_graph=True)[0]
            jacobian_fuse_pet = torch.autograd.grad(self.arg1[0,0,self.arg12-320,self.arg11], input2_tensor, retain_graph=True, create_graph=True)[0]

            jacob_ = torch.autograd.grad(self.arg7[0,0,self.arg12-320,self.arg11], input1_tensor, retain_graph=True, create_graph=True)[0]

            jacob_val_mri = np.squeeze(jacobian_fuse_mri.data.cpu().numpy())    
            jacob_val_pet = np.squeeze(jacobian_fuse_pet.data.cpu().numpy())
            jacob_val_numpy = np.squeeze(jacob_.data.cpu().numpy())

            x_mri = np.asarray(np.where(np.any(jacob_val_numpy, axis = 0)))
            y_mri = np.asarray(np.where(np.any(jacob_val_numpy, axis = 1)))
            minx_mri, maxx_mri, miny_mri, maxy_mri = np.min(x_mri), np.max(x_mri), np.min(y_mri), np.max(y_mri)  #return min and max coordinates
       
        
            radius = 7
            i = canvas.create_rectangle(self.arg11-radius, self.arg12-320-radius, self.arg11+radius, self.arg12-320+radius, outline = 'red')
            
            j = canvas.create_rectangle(self.arg11-radius, self.arg12-320+650-radius, self.arg11+radius, self.arg12-320+650+radius, outline = 'red')
        
            m = canvas.create_rectangle(self.arg11+600-radius, self.arg12-320+320-radius, self.arg11+600+radius, self.arg12-320+320+radius, outline = 'red') 
        
            n = canvas.create_rectangle(self.arg11-radius, self.arg12-320+320-radius, self.arg11+radius, self.arg12-320+320+radius, outline = 'red')
            if i > self.arg9+1:
                canvas.delete(i-24)
                canvas.delete(j-24)
                canvas.delete(m-24)
                canvas.delete(n-24)
                
            #color map details for the zoom images
            cmap = plt.get_cmap('viridis')
            #cNorm = mpl.colors.Normalize(vmin=self.arg3, vmax=self.arg4) #re-wrapping normalization
            cNorm_jacob = mpl.colors.PowerNorm(gamma=self.arg14, vmin=self.arg3, vmax=self.arg4)
            scalarMap_jacob = mpl.cm.ScalarMappable(norm=cNorm_jacob, cmap=cmap)
            cNorm_guide = mpl.colors.PowerNorm(gamma=self.arg10, vmin=self.arg3, vmax=self.arg4)
            scalarMap_guide = mpl.cm.ScalarMappable(norm=cNorm_guide, cmap=cmap)
        
            path = 'C:/Users/........./FuseVis/Guidance images/'

            im_fused = Image.fromarray(np.uint8(self.arg13[miny_mri:maxy_mri,minx_mri:maxx_mri]*255)).resize((256,256))
            im_fused = im_fused.convert("RGB")
            draw = ImageDraw.Draw(im_fused)
            draw.rectangle(((128, 128), (141, 141)), outline="Red")        
            im_out7 = ImageTk.PhotoImage(image=im_fused)
            canvas.create_image(300,320,image=im_out7,anchor=NW)
            canvas.image7 = im_out7
    
            im_mri = Image.fromarray(np.uint8(input1[0,0,miny_mri:maxy_mri,minx_mri:maxx_mri]*255)).resize((256,256))
            im_mri = im_mri.convert("RGB")
            draw = ImageDraw.Draw(im_mri)
            draw.rectangle(((128, 128), (141, 141)), outline="Red")
            im_out8 = ImageTk.PhotoImage(image=im_mri)
            canvas.create_image(300,0,image=im_out8,anchor=NW)
            canvas.image8 = im_out8
    
            im_pet = Image.fromarray(np.uint8(input2[0,0,miny_mri:maxy_mri,minx_mri:maxx_mri]*255)).resize((256,256)) 
            im_pet = im_pet.convert("RGB")
            draw = ImageDraw.Draw(im_pet)
            draw.rectangle(((128, 128), (141, 141)), outline="Red")
            im_out9 = ImageTk.PhotoImage(image=im_pet)
            canvas.create_image(300,650,image=im_out9,anchor=NW)
            canvas.image9 = im_out9  

            im_out10 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(255*scalarMap_jacob.to_rgba(jacob_val_mri))).resize((256,256)))
            canvas.create_image(1200,0,image=im_out10,anchor=NW)
            canvas.image10 = im_out10
        
            o = canvas.create_rectangle(self.arg11+1200-radius, self.arg12-320-radius, self.arg11+1200+radius, self.arg12-320+radius, outline = 'red')
            if i > self.arg9+1:
                canvas.delete(o-24)

            im_out11 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(255*scalarMap_jacob.to_rgba(jacob_val_mri[miny_mri:maxy_mri,minx_mri:maxx_mri]))).resize((256,256)))
            canvas.create_image(1500,0,image=im_out11,anchor=NW)
            canvas.image11 = im_out11
        
            im_out13 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(255*scalarMap_jacob.to_rgba(jacob_val_pet))).resize((256,256)))
            canvas.create_image(1200,650,image=im_out13,anchor=NW)       
            canvas.image13 = im_out13
                
            p = canvas.create_rectangle(self.arg11+1200-radius, self.arg12-320+650-radius, self.arg11+1200+radius, self.arg12-320+650+radius, outline = 'red')
            if i > self.arg9+1:
                canvas.delete(p-24)               
    
            im_out14 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(255*scalarMap_jacob.to_rgba(jacob_val_pet[miny_mri:maxy_mri,minx_mri:maxx_mri]))).resize((256,256)))
            canvas.create_image(1500,650,image=im_out14,anchor=NW)
            canvas.i4age14 = im_out14
        
            #display the zoom guidance MRI images 
            im_guide_mri = Image.fromarray(np.uint8(255*scalarMap_guide.to_rgba(self.arg5[miny_mri:maxy_mri,minx_mri:maxx_mri]))).resize((256,256))
            im_guide_mri = im_guide_mri.convert("RGB")
            draw = ImageDraw.Draw(im_guide_mri)
            draw.rectangle(((128, 128), (141, 141)), outline="Red")
            im_out15 = ImageTk.PhotoImage(image=im_guide_mri)
            canvas.create_image(900,0,image=im_out15,anchor=NW)
            canvas.image15 = im_out15
        
            #display the zoom guidance PET images 
            im_guide_pet = Image.fromarray(np.uint8(255*scalarMap_guide.to_rgba(self.arg6[miny_mri:maxy_mri,minx_mri:maxx_mri]))).resize((256,256))
            im_guide_pet = im_guide_pet.convert("RGB")
            draw = ImageDraw.Draw(im_guide_pet)
            draw.rectangle(((128, 128), (141, 141)), outline="Red")
            im_out16 = ImageTk.PhotoImage(image=im_guide_pet)
            canvas.create_image(900,650,image=im_out16,anchor=NW)
            canvas.image16 = im_out16
        
        
            #display the zoom fused RGB images
            im_fused_RGB = Image.fromarray(np.uint8(self.arg8[miny_mri:maxy_mri,minx_mri:maxx_mri]*255)).resize((256,256))
            #im_guide_pet = im_guide_pet.convert("RGB")
            draw = ImageDraw.Draw(im_fused_RGB)
            draw.rectangle(((128, 128), (141, 141)), outline="Red")
            im_out17 = ImageTk.PhotoImage(image=im_fused_RGB)
            canvas.create_image(900,320,image=im_out17,anchor=NW)
            canvas.image17 = im_out17  
            
            im_out2 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(255*scalarMap_guide.to_rgba(self.arg5))).resize((256,256)))
            canvas.create_image(600,0,image=im_out2,anchor=NW)
            canvas.image2 = im_out2
          
            k = canvas.create_rectangle(self.arg11+600-radius, self.arg12-320-radius, self.arg11+600+radius, self.arg12-320+radius, outline = 'red')
            if i > self.arg9+1:
                canvas.delete(k-24)

            im_out3 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(255*scalarMap_guide.to_rgba(self.arg6))).resize((256,256)))
            canvas.create_image(600,650,image=im_out3,anchor=NW)
            canvas.image3 = im_out3
            
            l = canvas.create_rectangle(self.arg11+600-radius, self.arg12-320+650-radius, self.arg11+600+radius, self.arg12-320+650+radius, outline = 'red')
            if i > self.arg9+1:
                canvas.delete(l-24) 
                
            fig = plt.figure(figsize=(0.2, 23))
            ax1 = fig.add_axes([0.05, 0.80, 0.9, 0.15])
            cb1 = mpl.colorbar.ColorbarBase(ax1, cmap=cmap,norm=cNorm_jacob,orientation='vertical')
            plt.plot()
            plt.savefig('C:/Users/........./FuseVis/Guidance images/colorbar.png', bbox_inches = 'tight',pad_inches = 0)
            plt.close()
            im_out50 = ImageTk.PhotoImage(file ='C:/Users/........./FuseVis/Guidance images/colorbar.png')
            
            fig1 = plt.figure(figsize=(0.2, 23))
            ax2 = fig1.add_axes([0.05, 0.80, 0.9, 0.15])
            cb2 = mpl.colorbar.ColorbarBase(ax2, cmap=cmap,norm=cNorm_guide,orientation='vertical')
            plt.plot()
            plt.savefig('C:/Users/........./FuseVis/Guidance images/colorbar.png', bbox_inches = 'tight',pad_inches = 0)
            plt.close()
            im_out51 = ImageTk.PhotoImage(file ='C:/Users/........./FuseVis/Guidance images/colorbar.png')
            
            canvas.create_image(1152,650,image=im_out51,anchor=NW)
            canvas.image50 = im_out51
            
            canvas.create_image(1752,650,image=im_out50,anchor=NW)
            canvas.image51 = im_out50
        
            canvas.create_image(1752,4,image=im_out50,anchor=NW)
            canvas.image52 = im_out50  
        
            canvas.create_image(1152,4,image=im_out51,anchor=NW)
            canvas.image53 = im_out51    
                
            #time2=time.time()
            #print(time2-time1) 
            
        elif self.arg11 >= 0 and self.arg11 <= 255 and self.arg12 >= 650 and self.arg12 <= 905:
            #time1 = time.time()
            print('mouse position is at' + '(' + str(self.arg12-320) + ',' + str(self.arg11) + ')', end='\r')
            #display the output MRI Jacobian image
            jacobian_fuse_mri = torch.autograd.grad(self.arg1[0,0,self.arg12-650,self.arg11], input1_tensor, retain_graph=True, create_graph=True)[0]
            jacobian_fuse_pet = torch.autograd.grad(self.arg1[0,0,self.arg12-650,self.arg11], input2_tensor, retain_graph=True, create_graph=True)[0]

            jacob_ = torch.autograd.grad(self.arg7[0,0,self.arg12-650,self.arg11], input1_tensor, retain_graph=True, create_graph=True)[0]

            jacob_val_mri = np.squeeze(jacobian_fuse_mri.data.cpu().numpy())    
            jacob_val_pet = np.squeeze(jacobian_fuse_pet.data.cpu().numpy())
            jacob_val_numpy = np.squeeze(jacob_.data.cpu().numpy())

            x_mri = np.asarray(np.where(np.any(jacob_val_numpy, axis = 0)))
            y_mri = np.asarray(np.where(np.any(jacob_val_numpy, axis = 1)))
            minx_mri, maxx_mri, miny_mri, maxy_mri = np.min(x_mri), np.max(x_mri), np.min(y_mri), np.max(y_mri)  #return min and max coordinates
       
        
            radius = 7
            i = canvas.create_rectangle(self.arg11-radius, self.arg12-650-radius, self.arg11+radius, self.arg12-650+radius, outline = 'red')
           
            j = canvas.create_rectangle(self.arg11-radius, self.arg12-650+650-radius, self.arg11+radius, self.arg12-650+650+radius, outline = 'red')
        
            m = canvas.create_rectangle(self.arg11+600-radius, self.arg12-650+320-radius, self.arg11+600+radius, self.arg12-650+320+radius, outline = 'red') 
        
            n = canvas.create_rectangle(self.arg11-radius, self.arg12-650+320-radius, self.arg11+radius, self.arg12-650+320+radius, outline = 'red')
            if i > self.arg9+1:
                canvas.delete(i-24)
                canvas.delete(j-24)
                canvas.delete(m-24)
                canvas.delete(n-24)

            #color map details for the zoom images
            cmap = plt.get_cmap('viridis')
            #cNorm = mpl.colors.Normalize(vmin=self.arg3, vmax=self.arg4) #re-wrapping normalization
            cNorm_jacob = mpl.colors.PowerNorm(gamma=self.arg14, vmin=self.arg3, vmax=self.arg4)
            scalarMap_jacob = mpl.cm.ScalarMappable(norm=cNorm_jacob, cmap=cmap)
            cNorm_guide = mpl.colors.PowerNorm(gamma=self.arg10, vmin=self.arg3, vmax=self.arg4)
            scalarMap_guide = mpl.cm.ScalarMappable(norm=cNorm_guide, cmap=cmap)
        
            path = 'C:/Users/........./FuseVis/Guidance images/'

            im_fused = Image.fromarray(np.uint8(self.arg13[miny_mri:maxy_mri,minx_mri:maxx_mri]*255)).resize((256,256))
            im_fused = im_fused.convert("RGB")
            draw = ImageDraw.Draw(im_fused)
            draw.rectangle(((128, 128), (141, 141)), outline="Red")        
            im_out7 = ImageTk.PhotoImage(image=im_fused)
            canvas.create_image(300,320,image=im_out7,anchor=NW)
            canvas.image7 = im_out7
            #plt.tight_layout()
    
            im_mri = Image.fromarray(np.uint8(input1[0,0,miny_mri:maxy_mri,minx_mri:maxx_mri]*255)).resize((256,256))
            im_mri = im_mri.convert("RGB")
            draw = ImageDraw.Draw(im_mri)
            draw.rectangle(((128, 128), (141, 141)), outline="Red")
            im_out8 = ImageTk.PhotoImage(image=im_mri)
            canvas.create_image(300,0,image=im_out8,anchor=NW)
            canvas.image8 = im_out8
    
            im_pet = Image.fromarray(np.uint8(input2[0,0,miny_mri:maxy_mri,minx_mri:maxx_mri]*255)).resize((256,256)) 
            im_pet = im_pet.convert("RGB")
            draw = ImageDraw.Draw(im_pet)
            draw.rectangle(((128, 128), (141, 141)), outline="Red")
            im_out9 = ImageTk.PhotoImage(image=im_pet)
            canvas.create_image(300,650,image=im_out9,anchor=NW)
            canvas.image9 = im_out9  

            im_out10 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(255*scalarMap_jacob.to_rgba(jacob_val_mri))).resize((256,256)))
            canvas.create_image(1200,0,image=im_out10,anchor=NW)
            canvas.image10 = im_out10
        
            o = canvas.create_rectangle(self.arg11+1200-radius, self.arg12-650-radius, self.arg11+1200+radius, self.arg12-650+radius, outline = 'red')
            if i > self.arg9+1:
                canvas.delete(o-24)

            im_out11 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(255*scalarMap_jacob.to_rgba(jacob_val_mri[miny_mri:maxy_mri,minx_mri:maxx_mri]))).resize((256,256)))
            canvas.create_image(1500,0,image=im_out11,anchor=NW)
            canvas.image11 = im_out11
        
            im_out13 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(255*scalarMap_jacob.to_rgba(jacob_val_pet))).resize((256,256)))
            canvas.create_image(1200,650,image=im_out13,anchor=NW)       
            canvas.image13 = im_out13
        
                
            p = canvas.create_rectangle(self.arg11+1200-radius, self.arg12-650+650-radius, self.arg11+1200+radius, self.arg12-650+650+radius, outline = 'red')
            if i > self.arg9+1:
                canvas.delete(p-24)
               
    
            im_out14 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(255*scalarMap_jacob.to_rgba(jacob_val_pet[miny_mri:maxy_mri,minx_mri:maxx_mri]))).resize((256,256)))
            canvas.create_image(1500,650,image=im_out14,anchor=NW)
            canvas.i4age14 = im_out14
        
                
            #display the zoom guidance MRI images 
            im_guide_mri = Image.fromarray(np.uint8(255*scalarMap_guide.to_rgba(self.arg5[miny_mri:maxy_mri,minx_mri:maxx_mri]))).resize((256,256))
            im_guide_mri = im_guide_mri.convert("RGB")
            draw = ImageDraw.Draw(im_guide_mri)
            draw.rectangle(((128, 128), (141, 141)), outline="Red")
            im_out15 = ImageTk.PhotoImage(image=im_guide_mri)
            canvas.create_image(900,0,image=im_out15,anchor=NW)
            canvas.image15 = im_out15
        
        
            #display the zoom guidance PET images 
            im_guide_pet = Image.fromarray(np.uint8(255*scalarMap_guide.to_rgba(self.arg6[miny_mri:maxy_mri,minx_mri:maxx_mri]))).resize((256,256))
            im_guide_pet = im_guide_pet.convert("RGB")
            draw = ImageDraw.Draw(im_guide_pet)
            draw.rectangle(((128, 128), (141, 141)), outline="Red")
            im_out16 = ImageTk.PhotoImage(image=im_guide_pet)
            canvas.create_image(900,650,image=im_out16,anchor=NW)
            canvas.image16 = im_out16
        
        
            #display the zoom fused RGB images
            im_fused_RGB = Image.fromarray(np.uint8(self.arg8[miny_mri:maxy_mri,minx_mri:maxx_mri]*255)).resize((256,256))
            #im_guide_pet = im_guide_pet.convert("RGB")
            draw = ImageDraw.Draw(im_fused_RGB)
            draw.rectangle(((128, 128), (141, 141)), outline="Red")
            im_out17 = ImageTk.PhotoImage(image=im_fused_RGB)
            canvas.create_image(900,320,image=im_out17,anchor=NW)
            canvas.image17 = im_out17  
            
            im_out2 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(255*scalarMap_guide.to_rgba(self.arg5))).resize((256,256)))
            canvas.create_image(600,0,image=im_out2,anchor=NW)
            canvas.image2 = im_out2
            
            k = canvas.create_rectangle(self.arg11+600-radius, self.arg12-650-radius, self.arg11+600+radius, self.arg12-650+radius, outline = 'red')
            if i > self.arg9+1:
                canvas.delete(k-24)
                
            im_out3 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(255*scalarMap_guide.to_rgba(self.arg6))).resize((256,256)))
            canvas.create_image(600,650,image=im_out3,anchor=NW)
            canvas.image3 = im_out3
            
            l = canvas.create_rectangle(self.arg11+600-radius, self.arg12-650+650-radius, self.arg11+600+radius, self.arg12-650+650+radius, outline = 'red')
            if i > self.arg9+1:
                canvas.delete(l-24)   
                
            fig = plt.figure(figsize=(0.2, 23))
            ax1 = fig.add_axes([0.05, 0.80, 0.9, 0.15])
            cb1 = mpl.colorbar.ColorbarBase(ax1, cmap=cmap,norm=cNorm_jacob,orientation='vertical')
            plt.plot()
            plt.savefig('C:/Users/........./FuseVis/Guidance images/colorbar.png', bbox_inches = 'tight',pad_inches = 0)
            plt.close()
            im_out50 = ImageTk.PhotoImage(file ='C:/Users/........./FuseVis/Guidance images/colorbar.png')
            
            fig = plt.figure(figsize=(0.2, 23))
            ax2 = fig.add_axes([0.05, 0.80, 0.9, 0.15])
            cb2 = mpl.colorbar.ColorbarBase(ax2, cmap=cmap,norm=cNorm_guide,orientation='vertical')
            plt.plot()
            plt.savefig('C:/Users/........./FuseVis/Guidance images/colorbar.png', bbox_inches = 'tight',pad_inches = 0)
            plt.close()
            im_out51 = ImageTk.PhotoImage(file ='C:/Users/........./FuseVis/Guidance images/colorbar.png')
            
            canvas.create_image(1152,650,image=im_out51,anchor=NW)
            canvas.image50 = im_out51
            
            canvas.create_image(1752,650,image=im_out50,anchor=NW)
            canvas.image51 = im_out50
        
            canvas.create_image(1752,4,image=im_out50,anchor=NW)
            canvas.image52 = im_out50  
        
            canvas.create_image(1152,4,image=im_out51,anchor=NW)
            canvas.image53 = im_out51   
            
            
        elif self.arg11 >= 600 and self.arg11 <= 855 and self.arg12 >= 0 and self.arg12 <= 255:
            #time1 = time.time()
            print('mouse position is at' + '(' + str(self.arg12) + ',' + str(self.arg11) + ')', end='\r')
            #display the output MRI Jacobian image
            jacobian_fuse_mri = torch.autograd.grad(self.arg1[0,0,self.arg12,self.arg11-600], input1_tensor, retain_graph=True, create_graph=True)[0]
            jacobian_fuse_pet = torch.autograd.grad(self.arg1[0,0,self.arg12,self.arg11-600], input2_tensor, retain_graph=True, create_graph=True)[0]

            jacob_ = torch.autograd.grad(self.arg7[0,0,self.arg12,self.arg11-600], input1_tensor, retain_graph=True, create_graph=True)[0]

            jacob_val_mri = np.squeeze(jacobian_fuse_mri.data.cpu().numpy())    
            jacob_val_pet = np.squeeze(jacobian_fuse_pet.data.cpu().numpy())
            jacob_val_numpy = np.squeeze(jacob_.data.cpu().numpy())

            x_mri = np.asarray(np.where(np.any(jacob_val_numpy, axis = 0)))
            y_mri = np.asarray(np.where(np.any(jacob_val_numpy, axis = 1)))
            minx_mri, maxx_mri, miny_mri, maxy_mri = np.min(x_mri), np.max(x_mri), np.min(y_mri), np.max(y_mri)  #return min and max coordinates
       
        
            radius = 7
            i = canvas.create_rectangle(self.arg11-600-radius, self.arg12-radius, self.arg11-600+radius, self.arg12+radius, outline = 'red')
     
            j = canvas.create_rectangle(self.arg11-600-radius, self.arg12+650-radius, self.arg11-600+radius, self.arg12+650+radius, outline = 'red')
        
            m = canvas.create_rectangle(self.arg11-600+600-radius, self.arg12+320-radius, self.arg11-600+600+radius, self.arg12+320+radius, outline = 'red') 
        
            n = canvas.create_rectangle(self.arg11-600-radius, self.arg12+320-radius, self.arg11-600+radius, self.arg12+320+radius, outline = 'red')
            if i > self.arg9+1:
                canvas.delete(i-24)
                canvas.delete(j-24)
                canvas.delete(m-24)
                canvas.delete(n-24)
                
     
            #color map detprintails for the zoom images
            cmap = plt.get_cmap('viridis')
            #cNorm = mpl.colors.Normalize(vmin=self.arg3, vmax=self.arg4) #re-wrapping normalization
            cNorm_jacob = mpl.colors.PowerNorm(gamma=self.arg14, vmin=self.arg3, vmax=self.arg4)
            scalarMap_jacob = mpl.cm.ScalarMappable(norm=cNorm_jacob, cmap=cmap)
            cNorm_guide = mpl.colors.PowerNorm(gamma=self.arg10, vmin=self.arg3, vmax=self.arg4)
            scalarMap_guide = mpl.cm.ScalarMappable(norm=cNorm_guide, cmap=cmap)
        
            path = 'C:/Users/........./FuseVis/Guidance images/'

            im_fused = Image.fromarray(np.uint8(self.arg13[miny_mri:maxy_mri,minx_mri:maxx_mri]*255)).resize((256,256))
            im_fused = im_fused.convert("RGB")
            draw = ImageDraw.Draw(im_fused)
            draw.rectangle(((128, 128), (141, 141)), outline="Red")        
            im_out7 = ImageTk.PhotoImage(image=im_fused)
            canvas.create_image(300,320,image=im_out7,anchor=NW)
            canvas.image7 = im_out7
            #plt.tight_layout()
    
            im_mri = Image.fromarray(np.uint8(input1[0,0,miny_mri:maxy_mri,minx_mri:maxx_mri]*255)).resize((256,256))
            im_mri = im_mri.convert("RGB")
            draw = ImageDraw.Draw(im_mri)
            draw.rectangle(((128, 128), (141, 141)), outline="Red")
            im_out8 = ImageTk.PhotoImage(image=im_mri)
            canvas.create_image(300,0,image=im_out8,anchor=NW)
            canvas.image8 = im_out8
    
            im_pet = Image.fromarray(np.uint8(input2[0,0,miny_mri:maxy_mri,minx_mri:maxx_mri]*255)).resize((256,256)) 
            im_pet = im_pet.convert("RGB")
            draw = ImageDraw.Draw(im_pet)
            draw.rectangle(((128, 128), (141, 141)), outline="Red")
            im_out9 = ImageTk.PhotoImage(image=im_pet)
            canvas.create_image(300,650,image=im_out9,anchor=NW)
            canvas.image9 = im_out9  

            im_out10 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(255*scalarMap_jacob.to_rgba(jacob_val_mri))).resize((256,256)))
            canvas.create_image(1200,0,image=im_out10,anchor=NW)
            canvas.image10 = im_out10
        
            o = canvas.create_rectangle(self.arg11-600+1200-radius, self.arg12-radius, self.arg11-600+1200+radius, self.arg12+radius, outline = 'red')
            if i > self.arg9+1:
                canvas.delete(o-24)

            im_out11 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(255*scalarMap_jacob.to_rgba(jacob_val_mri[miny_mri:maxy_mri,minx_mri:maxx_mri]))).resize((256,256)))
            canvas.create_image(1500,0,image=im_out11,anchor=NW)
            canvas.image11 = im_out11
        
            im_out13 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(255*scalarMap_jacob.to_rgba(jacob_val_pet))).resize((256,256)))
            canvas.create_image(1200,650,image=im_out13,anchor=NW)       
            canvas.image13 = im_out13
        
                
            p = canvas.create_rectangle(self.arg11-600+1200-radius, self.arg12+650-radius, self.arg11-600+1200+radius, self.arg12+650+radius, outline = 'red')
            if i > self.arg9+1:
                canvas.delete(p-24)
               
    
            im_out14 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(255*scalarMap_jacob.to_rgba(jacob_val_pet[miny_mri:maxy_mri,minx_mri:maxx_mri]))).resize((256,256)))
            canvas.create_image(1500,650,image=im_out14,anchor=NW)
            canvas.i4age14 = im_out14
        
                
            #display the zoom guidance MRI images 
            im_guide_mri = Image.fromarray(np.uint8(255*scalarMap_guide.to_rgba(self.arg5[miny_mri:maxy_mri,minx_mri:maxx_mri]))).resize((256,256))
            im_guide_mri = im_guide_mri.convert("RGB")
            draw = ImageDraw.Draw(im_guide_mri)
            draw.rectangle(((128, 128), (141, 141)), outline="Red")
            im_out15 = ImageTk.PhotoImage(image=im_guide_mri)
            canvas.create_image(900,0,image=im_out15,anchor=NW)
            canvas.image15 = im_out15
        
        
            #display the zoom guidance PET images 
            im_guide_pet = Image.fromarray(np.uint8(255*scalarMap_guide.to_rgba(self.arg6[miny_mri:maxy_mri,minx_mri:maxx_mri]))).resize((256,256))
            im_guide_pet = im_guide_pet.convert("RGB")
            draw = ImageDraw.Draw(im_guide_pet)
            draw.rectangle(((128, 128), (141, 141)), outline="Red")
            im_out16 = ImageTk.PhotoImage(image=im_guide_pet)
            canvas.create_image(900,650,image=im_out16,anchor=NW)
            canvas.image16 = im_out16
        
        
            #display the zoom fused RGB images
            im_fused_RGB = Image.fromarray(np.uint8(self.arg8[miny_mri:maxy_mri,minx_mri:maxx_mri]*255)).resize((256,256))
            #im_guide_pet = im_guide_pet.convert("RGB")
            draw = ImageDraw.Draw(im_fused_RGB)
            draw.rectangle(((128, 128), (141, 141)), outline="Red")
            im_out17 = ImageTk.PhotoImage(image=im_fused_RGB)
            canvas.create_image(900,320,image=im_out17,anchor=NW)
            canvas.image17 = im_out17     
            
            im_out2 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(255*scalarMap_guide.to_rgba(self.arg5))).resize((256,256)))
            canvas.create_image(600,0,image=im_out2,anchor=NW)
            canvas.image2 = im_out2
            
            k = canvas.create_rectangle(self.arg11-600+600-radius, self.arg12-radius, self.arg11-600+600+radius, self.arg12+radius, outline = 'red')
            if i > self.arg9+1:
                canvas.delete(k-24)
          
            im_out3 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(255*scalarMap_guide.to_rgba(self.arg6))).resize((256,256)))
            canvas.create_image(600,650,image=im_out3,anchor=NW)
            canvas.image3 = im_out3
            
            l = canvas.create_rectangle(self.arg11-600+600-radius, self.arg12+650-radius, self.arg11-600+600+radius, self.arg12+650+radius, outline = 'red')
            if i > self.arg9+1:
                canvas.delete(l-24)
                
            fig = plt.figure(figsize=(0.2, 23))
            ax1 = fig.add_axes([0.05, 0.80, 0.9, 0.15])
            cb1 = mpl.colorbar.ColorbarBase(ax1, cmap=cmap,norm=cNorm_jacob,orientation='vertical')
            plt.plot()
            plt.savefig('C:/Users/........./FuseVis/Guidance images/colorbar.png', bbox_inches = 'tight',pad_inches = 0)
            plt.close()
            im_out50 = ImageTk.PhotoImage(file ='C:/Users/........./FuseVis/Guidance images/colorbar.png')
            
            fig1 = plt.figure(figsize=(0.2, 23))
            ax2 = fig1.add_axes([0.05, 0.80, 0.9, 0.15])
            cb2 = mpl.colorbar.ColorbarBase(ax2, cmap=cmap,norm=cNorm_guide,orientation='vertical')
            plt.plot()
            plt.savefig('C:/Users/........./FuseVis/Guidance images/colorbar.png', bbox_inches = 'tight',pad_inches = 0)
            plt.close()
            im_out51 = ImageTk.PhotoImage(file ='C:/Users/........./FuseVis/Guidance images/colorbar.png')
            
            canvas.create_image(1152,650,image=im_out51,anchor=NW)
            canvas.image50 = im_out51
            
            canvas.create_image(1752,650,image=im_out50,anchor=NW)
            canvas.image51 = im_out50
        
            canvas.create_image(1752,4,image=im_out50,anchor=NW)
            canvas.image52 = im_out50  
        
            canvas.create_image(1152,4,image=im_out51,anchor=NW)
            canvas.image53 = im_out51   
            
            #time2=time.time()
            #print(time2-time1)
            
        elif self.arg11 >= 600 and self.arg11 <= 855 and self.arg12 >= 320 and self.arg12 <= 575:
            #time1 = time.time()
            print('mouse position is at' + '(' + str(self.arg12) + ',' + str(self.arg11) + ')', end='\r')
            #display the output MRI Jacobian image
            jacobian_fuse_mri = torch.autograd.grad(self.arg1[0,0,self.arg12-320,self.arg11-600], input1_tensor, retain_graph=True, create_graph=True)[0]
            jacobian_fuse_pet = torch.autograd.grad(self.arg1[0,0,self.arg12-320,self.arg11-600], input2_tensor, retain_graph=True, create_graph=True)[0]

            jacob_ = torch.autograd.grad(self.arg7[0,0,self.arg12-320,self.arg11-600], input1_tensor, retain_graph=True, create_graph=True)[0]

            jacob_val_mri = np.squeeze(jacobian_fuse_mri.data.cpu().numpy())    
            jacob_val_pet = np.squeeze(jacobian_fuse_pet.data.cpu().numpy())
            jacob_val_numpy = np.squeeze(jacob_.data.cpu().numpy())

            x_mri = np.asarray(np.where(np.any(jacob_val_numpy, axis = 0)))
            y_mri = np.asarray(np.where(np.any(jacob_val_numpy, axis = 1)))
            minx_mri, maxx_mri, miny_mri, maxy_mri = np.min(x_mri), np.max(x_mri), np.min(y_mri), np.max(y_mri)  #return min and max coordinates
       
        
            radius = 7
            i = canvas.create_rectangle(self.arg11-600-radius, self.arg12-320-radius, self.arg11-600+radius, self.arg12-320+radius, outline = 'red')
          
            j = canvas.create_rectangle(self.arg11-600-radius, self.arg12-320+650-radius, self.arg11-600+radius, self.arg12-320+650+radius, outline = 'red')
        
            m = canvas.create_rectangle(self.arg11-600+600-radius, self.arg12-320+320-radius, self.arg11-600+600+radius, self.arg12-320+320+radius, outline = 'red') 
        
            n = canvas.create_rectangle(self.arg11-600-radius, self.arg12-320+320-radius, self.arg11-600+radius, self.arg12-320+320+radius, outline = 'red')
            if i > self.arg9+1:
                canvas.delete(i-24)
                canvas.delete(j-24)
                canvas.delete(m-24)
                canvas.delete(n-24)
                
           
            #color map details for the zoom images
            cmap = plt.get_cmap('viridis')
            #cNorm = mpl.colors.Normalize(vmin=self.arg3, vmax=self.arg4) #re-wrapping normalization
            cNorm_jacob = mpl.colors.PowerNorm(gamma=self.arg14, vmin=self.arg3, vmax=self.arg4)
            scalarMap_jacob = mpl.cm.ScalarMappable(norm=cNorm_jacob, cmap=cmap)
            cNorm_guide = mpl.colors.PowerNorm(gamma=self.arg10, vmin=self.arg3, vmax=self.arg4)
            scalarMap_guide = mpl.cm.ScalarMappable(norm=cNorm_guide, cmap=cmap)
        
            path = 'C:/Users/........./FuseVis/Guidance images/'

            im_fused = Image.fromarray(np.uint8(self.arg13[miny_mri:maxy_mri,minx_mri:maxx_mri]*255)).resize((256,256))
            im_fused = im_fused.convert("RGB")
            draw = ImageDraw.Draw(im_fused)
            draw.rectangle(((128, 128), (141, 141)), outline="Red")        
            im_out7 = ImageTk.PhotoImage(image=im_fused)
            canvas.create_image(300,320,image=im_out7,anchor=NW)
            canvas.image7 = im_out7
            #plt.tight_layout()
    
            im_mri = Image.fromarray(np.uint8(input1[0,0,miny_mri:maxy_mri,minx_mri:maxx_mri]*255)).resize((256,256))
            im_mri = im_mri.convert("RGB")
            draw = ImageDraw.Draw(im_mri)
            draw.rectangle(((128, 128), (141, 141)), outline="Red")
            im_out8 = ImageTk.PhotoImage(image=im_mri)
            canvas.create_image(300,0,image=im_out8,anchor=NW)
            canvas.image8 = im_out8
    
            im_pet = Image.fromarray(np.uint8(input2[0,0,miny_mri:maxy_mri,minx_mri:maxx_mri]*255)).resize((256,256)) 
            im_pet = im_pet.convert("RGB")
            draw = ImageDraw.Draw(im_pet)
            draw.rectangle(((128, 128), (141, 141)), outline="Red")
            im_out9 = ImageTk.PhotoImage(image=im_pet)
            canvas.create_image(300,650,image=im_out9,anchor=NW)
            canvas.image9 = im_out9  

            im_out10 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(255*scalarMap_jacob.to_rgba(jacob_val_mri))).resize((256,256)))
            canvas.create_image(1200,0,image=im_out10,anchor=NW)
            canvas.image10 = im_out10
        
            o = canvas.create_rectangle(self.arg11-600+1200-radius, self.arg12-320-radius, self.arg11-600+1200+radius, self.arg12-320+radius, outline = 'red')
            if i > self.arg9+1:
                canvas.delete(o-24)

            im_out11 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(255*scalarMap_jacob.to_rgba(jacob_val_mri[miny_mri:maxy_mri,minx_mri:maxx_mri]))).resize((256,256)))
            canvas.create_image(1500,0,image=im_out11,anchor=NW)
            canvas.image11 = im_out11
        
            im_out13 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(255*scalarMap_jacob.to_rgba(jacob_val_pet))).resize((256,256)))
            canvas.create_image(1200,650,image=im_out13,anchor=NW)       
            canvas.image13 = im_out13
        
                
            p = canvas.create_rectangle(self.arg11-600+1200-radius, self.arg12-320+650-radius, self.arg11-600+1200+radius, self.arg12-320+650+radius, outline = 'red')
            if i > self.arg9+1:
                canvas.delete(p-24)
               
    
            im_out14 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(255*scalarMap_jacob.to_rgba(jacob_val_pet[miny_mri:maxy_mri,minx_mri:maxx_mri]))).resize((256,256)))
            canvas.create_image(1500,650,image=im_out14,anchor=NW)
            canvas.i4age14 = im_out14
        
                
            #display the zoom guidance MRI images 
            im_guide_mri = Image.fromarray(np.uint8(255*scalarMap_guide.to_rgba(self.arg5[miny_mri:maxy_mri,minx_mri:maxx_mri]))).resize((256,256))
            im_guide_mri = im_guide_mri.convert("RGB")
            draw = ImageDraw.Draw(im_guide_mri)
            draw.rectangle(((128, 128), (141, 141)), outline="Red")
            im_out15 = ImageTk.PhotoImage(image=im_guide_mri)
            canvas.create_image(900,0,image=im_out15,anchor=NW)
            canvas.image15 = im_out15
        
        
            #display the zoom guidance PET images 
            im_guide_pet = Image.fromarray(np.uint8(255*scalarMap_guide.to_rgba(self.arg6[miny_mri:maxy_mri,minx_mri:maxx_mri]))).resize((256,256))
            im_guide_pet = im_guide_pet.convert("RGB")
            draw = ImageDraw.Draw(im_guide_pet)
            draw.rectangle(((128, 128), (141, 141)), outline="Red")
            im_out16 = ImageTk.PhotoImage(image=im_guide_pet)
            canvas.create_image(900,650,image=im_out16,anchor=NW)
            canvas.image16 = im_out16
        
        
            #display the zoom fused RGB images
            im_fused_RGB = Image.fromarray(np.uint8(self.arg8[miny_mri:maxy_mri,minx_mri:maxx_mri]*255)).resize((256,256))
            #im_guide_pet = im_guide_pet.convert("RGB")
            draw = ImageDraw.Draw(im_fused_RGB)
            draw.rectangle(((128, 128), (141, 141)), outline="Red")
            im_out17 = ImageTk.PhotoImage(image=im_fused_RGB)
            canvas.create_image(900,320,image=im_out17,anchor=NW)
            canvas.image17 = im_out17   
            
            im_out2 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(255*scalarMap_guide.to_rgba(self.arg5))).resize((256,256)))
            canvas.create_image(600,0,image=im_out2,anchor=NW)
            canvas.image2 = im_out2
            
            k = canvas.create_rectangle(self.arg11-600+600-radius, self.arg12-320-radius, self.arg11-600+600+radius, self.arg12-320+radius, outline = 'red')
            if i > self.arg9+1:
                canvas.delete(k-24)
                
            im_out3 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(255*scalarMap_guide.to_rgba(self.arg6))).resize((256,256)))
            canvas.create_image(600,650,image=im_out3,anchor=NW)
            canvas.image3 = im_out3
            
            l = canvas.create_rectangle(self.arg11-600+600-radius, self.arg12-320+650-radius, self.arg11-600+600+radius, self.arg12-320+650+radius, outline = 'red')
            if i > self.arg9+1:
                canvas.delete(l-24)   
                
            fig = plt.figure(figsize=(0.2, 23))
            ax1 = fig.add_axes([0.05, 0.80, 0.9, 0.15])
            cb1 = mpl.colorbar.ColorbarBase(ax1, cmap=cmap,norm=cNorm_jacob,orientation='vertical')
            plt.plot()
            plt.savefig('C:/Users/........./FuseVis/Guidance images/colorbar.png', bbox_inches = 'tight',pad_inches = 0)
            plt.close()
            im_out50 = ImageTk.PhotoImage(file ='C:/Users/........./FuseVis/Guidance images/colorbar.png')
            
            fig1 = plt.figure(figsize=(0.2, 23))
            ax2 = fig1.add_axes([0.05, 0.80, 0.9, 0.15])
            cb2 = mpl.colorbar.ColorbarBase(ax2, cmap=cmap,norm=cNorm_guide,orientation='vertical')
            plt.plot()
            plt.savefig('C:/Users/........./FuseVis/Guidance images/colorbar.png', bbox_inches = 'tight',pad_inches = 0)
            plt.close()
            im_out51 = ImageTk.PhotoImage(file ='C:/Users/........./FuseVis/Guidance images/colorbar.png')
            
            canvas.create_image(1152,650,image=im_out51,anchor=NW)
            canvas.image50 = im_out51
            
            canvas.create_image(1752,650,image=im_out50,anchor=NW)
            canvas.image51 = im_out50
        
            canvas.create_image(1752,4,image=im_out50,anchor=NW)
            canvas.image52 = im_out50  
        
            canvas.create_image(1152,4,image=im_out51,anchor=NW)
            canvas.image53 = im_out51  
            
            
        elif self.arg11 >= 600 and self.arg11 <= 855 and self.arg12 >= 650 and self.arg12 <= 905:
            #time1 = time.time()
            print('mouse position is at' + '(' + str(self.arg12) + ',' + str(self.arg11) + ')', end='\r')
            #display the output MRI Jacobian image
            jacobian_fuse_mri = torch.autograd.grad(self.arg1[0,0,self.arg12-650,self.arg11-600], input1_tensor, retain_graph=True, create_graph=True)[0]
            jacobian_fuse_pet = torch.autograd.grad(self.arg1[0,0,self.arg12-650,self.arg11-600], input2_tensor, retain_graph=True, create_graph=True)[0]

            jacob_ = torch.autograd.grad(self.arg7[0,0,self.arg12-650,self.arg11-600], input1_tensor, retain_graph=True, create_graph=True)[0]

            jacob_val_mri = np.squeeze(jacobian_fuse_mri.data.cpu().numpy())    
            jacob_val_pet = np.squeeze(jacobian_fuse_pet.data.cpu().numpy())
            jacob_val_numpy = np.squeeze(jacob_.data.cpu().numpy())

            x_mri = np.asarray(np.where(np.any(jacob_val_numpy, axis = 0)))
            y_mri = np.asarray(np.where(np.any(jacob_val_numpy, axis = 1)))
            minx_mri, maxx_mri, miny_mri, maxy_mri = np.min(x_mri), np.max(x_mri), np.min(y_mri), np.max(y_mri)  #return min and max coordinates
       
        
            radius = 7
            i = canvas.create_rectangle(self.arg11-600-radius, self.arg12-650-radius, self.arg11-600+radius, self.arg12-650+radius, outline = 'red')
           
            j = canvas.create_rectangle(self.arg11-600-radius, self.arg12-650+650-radius, self.arg11-600+radius, self.arg12-650+650+radius, outline = 'red')
        
            m = canvas.create_rectangle(self.arg11-600+600-radius, self.arg12-650+320-radius, self.arg11-600+600+radius, self.arg12-650+320+radius, outline = 'red') 
        
            n = canvas.create_rectangle(self.arg11-600-radius, self.arg12-650+320-radius, self.arg11-600+radius, self.arg12-650+320+radius, outline = 'red')
            if i > self.arg9+1:
                canvas.delete(i-24)
                canvas.delete(j-24)
                canvas.delete(m-24)
                canvas.delete(n-24)
                
        
            #color map details for the zoom images
            cmap = plt.get_cmap('viridis')
            #cNorm = mpl.colors.Normalize(vmin=self.arg3, vmax=self.arg4) #re-wrapping normalization
            cNorm_jacob = mpl.colors.PowerNorm(gamma=self.arg14, vmin=self.arg3, vmax=self.arg4)
            scalarMap_jacob = mpl.cm.ScalarMappable(norm=cNorm_jacob, cmap=cmap)
            cNorm_guide = mpl.colors.PowerNorm(gamma=self.arg10, vmin=self.arg3, vmax=self.arg4)
            scalarMap_guide = mpl.cm.ScalarMappable(norm=cNorm_guide, cmap=cmap)
        
            path = 'C:/Users/........./FuseVis/Guidance images/'

            im_fused = Image.fromarray(np.uint8(self.arg13[miny_mri:maxy_mri,minx_mri:maxx_mri]*255)).resize((256,256))
            im_fused = im_fused.convert("RGB")
            draw = ImageDraw.Draw(im_fused)
            draw.rectangle(((128, 128), (141, 141)), outline="Red")        
            im_out7 = ImageTk.PhotoImage(image=im_fused)
            canvas.create_image(300,320,image=im_out7,anchor=NW)
            canvas.image7 = im_out7
            #plt.tight_layout()
    
            im_mri = Image.fromarray(np.uint8(input1[0,0,miny_mri:maxy_mri,minx_mri:maxx_mri]*255)).resize((256,256))
            im_mri = im_mri.convert("RGB")
            draw = ImageDraw.Draw(im_mri)
            draw.rectangle(((128, 128), (141, 141)), outline="Red")
            im_out8 = ImageTk.PhotoImage(image=im_mri)
            canvas.create_image(300,0,image=im_out8,anchor=NW)
            canvas.image8 = im_out8
    
            im_pet = Image.fromarray(np.uint8(input2[0,0,miny_mri:maxy_mri,minx_mri:maxx_mri]*255)).resize((256,256)) 
            im_pet = im_pet.convert("RGB")
            draw = ImageDraw.Draw(im_pet)
            draw.rectangle(((128, 128), (141, 141)), outline="Red")
            im_out9 = ImageTk.PhotoImage(image=im_pet)
            canvas.create_image(300,650,image=im_out9,anchor=NW)
            canvas.image9 = im_out9  

            im_out10 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(255*scalarMap_jacob.to_rgba(jacob_val_mri))).resize((256,256)))
            canvas.create_image(1200,0,image=im_out10,anchor=NW)
            canvas.image10 = im_out10
        
            o = canvas.create_rectangle(self.arg11-600+1200-radius, self.arg12-650-radius, self.arg11-600+1200+radius, self.arg12-650+radius, outline = 'red')
            if i > self.arg9+1:
                canvas.delete(o-24)

            im_out11 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(255*scalarMap_jacob.to_rgba(jacob_val_mri[miny_mri:maxy_mri,minx_mri:maxx_mri]))).resize((256,256)))
            canvas.create_image(1500,0,image=im_out11,anchor=NW)
            canvas.image11 = im_out11
        
            im_out13 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(255*scalarMap_jacob.to_rgba(jacob_val_pet))).resize((256,256)))
            canvas.create_image(1200,650,image=im_out13,anchor=NW)       
            canvas.image13 = im_out13
        
                
            p = canvas.create_rectangle(self.arg11-600+1200-radius, self.arg12-650+650-radius, self.arg11-600+1200+radius, self.arg12-650+650+radius, outline = 'red')
            if i > self.arg9+1:
                canvas.delete(p-24)
               
    
            im_out14 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(255*scalarMap_jacob.to_rgba(jacob_val_pet[miny_mri:maxy_mri,minx_mri:maxx_mri]))).resize((256,256)))
            canvas.create_image(1500,650,image=im_out14,anchor=NW)
            canvas.image14 = im_out14
        
                
            #display the zoom guidance MRI images 
            im_guide_mri = Image.fromarray(np.uint8(255*scalarMap_guide.to_rgba(self.arg5[miny_mri:maxy_mri,minx_mri:maxx_mri]))).resize((256,256))
            im_guide_mri = im_guide_mri.convert("RGB")
            draw = ImageDraw.Draw(im_guide_mri)
            draw.rectangle(((128, 128), (141, 141)), outline="Red")
            im_out15 = ImageTk.PhotoImage(image=im_guide_mri)
            canvas.create_image(900,0,image=im_out15,anchor=NW)
            canvas.image15 = im_out15
        
        
            #display the zoom guidance PET images 
            im_guide_pet = Image.fromarray(np.uint8(255*scalarMap_guide.to_rgba(self.arg6[miny_mri:maxy_mri,minx_mri:maxx_mri]))).resize((256,256))
            im_guide_pet = im_guide_pet.convert("RGB")
            draw = ImageDraw.Draw(im_guide_pet)
            draw.rectangle(((128, 128), (141, 141)), outline="Red")
            im_out16 = ImageTk.PhotoImage(image=im_guide_pet)
            canvas.create_image(900,650,image=im_out16,anchor=NW)
            canvas.image16 = im_out16
        
        
            #display the zoom fused RGB images
            im_fused_RGB = Image.fromarray(np.uint8(self.arg8[miny_mri:maxy_mri,minx_mri:maxx_mri]*255)).resize((256,256))
            #im_guide_pet = im_guide_pet.convert("RGB")
            draw = ImageDraw.Draw(im_fused_RGB)
            draw.rectangle(((128, 128), (141, 141)), outline="Red")
            im_out17 = ImageTk.PhotoImage(image=im_fused_RGB)
            canvas.create_image(900,320,image=im_out17,anchor=NW)
            canvas.image17 = im_out17        
            
            im_out2 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(255*scalarMap_guide.to_rgba(self.arg5))).resize((256,256)))
            canvas.create_image(600,0,image=im_out2,anchor=NW)
            canvas.image2 = im_out2
            
            k = canvas.create_rectangle(self.arg11-600+600-radius, self.arg12-650-radius, self.arg11-600+600+radius, self.arg12-650+radius, outline = 'red')
            if i > self.arg9+1:
                canvas.delete(k-24)
                
            im_out3 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(255*scalarMap_guide.to_rgba(self.arg6))).resize((256,256)))
            canvas.create_image(600,650,image=im_out3,anchor=NW)
            canvas.image3 = im_out3
            
            l = canvas.create_rectangle(self.arg11-600+600-radius, self.arg12-650+650-radius, self.arg11-600+600+radius, self.arg12-650+650+radius, outline = 'red')
            if i > self.arg9+1:
                canvas.delete(l-24)
                
            fig = plt.figure(figsize=(0.2, 23))
            ax1 = fig.add_axes([0.05, 0.80, 0.9, 0.15])
            cb1 = mpl.colorbar.ColorbarBase(ax1, cmap=cmap,norm=cNorm_jacob,orientation='vertical')
            plt.plot()
            plt.savefig('C:/Users/........./FuseVis/Guidance images/colorbar.png', bbox_inches = 'tight',pad_inches = 0)
            plt.close()
            im_out50 = ImageTk.PhotoImage(file ='C:/Users/........./FuseVis/Guidance images/colorbar.png')
            
            fig1 = plt.figure(figsize=(0.2, 23))
            ax2 = fig1.add_axes([0.05, 0.80, 0.9, 0.15])
            cb2 = mpl.colorbar.ColorbarBase(ax2, cmap=cmap,norm=cNorm_guide,orientation='vertical')
            plt.plot()
            plt.savefig('C:/Users/........./FuseVis/Guidance images/colorbar.png', bbox_inches = 'tight',pad_inches = 0)
            plt.close()
            im_out51 = ImageTk.PhotoImage(file ='C:/Users/........./FuseVis/Guidance images/colorbar.png')
            
            canvas.create_image(1152,650,image=im_out51,anchor=NW)
            canvas.image50 = im_out51
            
            canvas.create_image(1752,650,image=im_out50,anchor=NW)
            canvas.image51 = im_out50
        
            canvas.create_image(1752,4,image=im_out50,anchor=NW)
            canvas.image52 = im_out50  
        
            canvas.create_image(1152,4,image=im_out51,anchor=NW)
            canvas.image53 = im_out51   
            
            #time2=time.time()
        return self.arg14

        
    def slider_Callback(self,slider_event):
        self.arg10 = float(slider_event)
        if self.arg11 >= 0 and self.arg11 <= 255 and self.arg12 >= 0 and self.arg12 <= 255:

            #time1 = time.time()
            #print('mouse position is at' + '(' + str(self.arg12) + ',' + str(self.arg11) + ')', end='\r')
            #display the output MRI Jacobian image
            jacobian_fuse_mri = torch.autograd.grad(self.arg1[0,0,self.arg12,self.arg11], input1_tensor, retain_graph=True, create_graph=True)[0]
            jacobian_fuse_pet = torch.autograd.grad(self.arg1[0,0,self.arg12,self.arg11], input2_tensor, retain_graph=True, create_graph=True)[0]

            jacob_ = torch.autograd.grad(self.arg7[0,0,self.arg12,self.arg11], input1_tensor, retain_graph=True, create_graph=True)[0]

            jacob_val_mri = np.squeeze(jacobian_fuse_mri.data.cpu().numpy())    
            jacob_val_pet = np.squeeze(jacobian_fuse_pet.data.cpu().numpy())
            jacob_val_numpy = np.squeeze(jacob_.data.cpu().numpy())
            

            x_mri = np.asarray(np.where(np.any(jacob_val_numpy, axis = 0)))
            y_mri = np.asarray(np.where(np.any(jacob_val_numpy, axis = 1)))
            minx_mri, maxx_mri, miny_mri, maxy_mri = np.min(x_mri), np.max(x_mri), np.min(y_mri), np.max(y_mri)  #return min and max coordinates
               
            radius = 7
            i = canvas.create_rectangle(self.arg11-radius, self.arg12-radius, self.arg11+radius, self.arg12+radius, outline = 'red')
           
            j = canvas.create_rectangle(self.arg11-radius, self.arg12+650-radius, self.arg11+radius, self.arg12+650+radius, outline = 'red')
        
            m = canvas.create_rectangle(self.arg11+600-radius, self.arg12+320-radius, self.arg11+600+radius, self.arg12+320+radius, outline = 'red') 
        
            n = canvas.create_rectangle(self.arg11-radius, self.arg12+320-radius, self.arg11+radius, self.arg12+320+radius, outline = 'red')
            
            if i > self.arg9+1:
                canvas.delete(i-24)
                canvas.delete(j-24)
                canvas.delete(m-24)
                canvas.delete(n-24)

            #color map details for the zoom images
            cmap = plt.get_cmap('viridis')
            #cNorm = mpl.colors.Normalize(vmin=self.arg3, vmax=self.arg4) #re-wrapping normalization
            cNorm_guide = mpl.colors.PowerNorm(gamma=self.arg10, vmin=self.arg3, vmax=self.arg4)
            scalarMap_guide = mpl.cm.ScalarMappable(norm=cNorm_guide, cmap=cmap)
            cNorm_jacob = mpl.colors.PowerNorm(gamma=self.arg14, vmin=self.arg3, vmax=self.arg4)
            scalarMap_jacob = mpl.cm.ScalarMappable(norm=cNorm_jacob, cmap=cmap)
        
            path = 'C:/Users/........./FuseVis/Guidance images/'

            im_fused = Image.fromarray(np.uint8(self.arg13[miny_mri:maxy_mri,minx_mri:maxx_mri]*255)).resize((256,256))
            im_fused = im_fused.convert("RGB")
            draw = ImageDraw.Draw(im_fused)
            draw.rectangle(((128, 128), (141, 141)), outline="Red")        
            im_out7 = ImageTk.PhotoImage(image=im_fused)
            canvas.create_image(300,320,image=im_out7,anchor=NW)
            canvas.image7 = im_out7
            #plt.tight_layout()
    
            im_mri = Image.fromarray(np.uint8(input1[0,0,miny_mri:maxy_mri,minx_mri:maxx_mri]*255)).resize((256,256))
            im_mri = im_mri.convert("RGB")
            draw = ImageDraw.Draw(im_mri)
            draw.rectangle(((128, 128), (141, 141)), outline="Red")
            im_out8 = ImageTk.PhotoImage(image=im_mri)
            canvas.create_image(300,0,image=im_out8,anchor=NW)
            canvas.image8 = im_out8
    
            im_pet = Image.fromarray(np.uint8(input2[0,0,miny_mri:maxy_mri,minx_mri:maxx_mri]*255)).resize((256,256)) 
            im_pet = im_pet.convert("RGB")
            draw = ImageDraw.Draw(im_pet)
            draw.rectangle(((128, 128), (141, 141)), outline="Red")
            im_out9 = ImageTk.PhotoImage(image=im_pet)
            canvas.create_image(300,650,image=im_out9,anchor=NW)
            canvas.image9 = im_out9  

            im_out10 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(255*(scalarMap_jacob.to_rgba(jacob_val_mri)))).resize((256,256)))
            canvas.create_image(1200,0,image=im_out10,anchor=NW)
            canvas.image10 = im_out10
        
            o = canvas.create_rectangle(self.arg11+1200-radius, self.arg12-radius, self.arg11+1200+radius, self.arg12+radius, outline = 'red')
            if i > self.arg9+1:
                canvas.delete(o-24)

            im_out11 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(255*(scalarMap_jacob.to_rgba(jacob_val_mri[miny_mri:maxy_mri,minx_mri:maxx_mri])))).resize((256,256)))
            canvas.create_image(1500,0,image=im_out11,anchor=NW)
            canvas.image11 = im_out11
        
            im_out13 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(255*scalarMap_jacob.to_rgba(jacob_val_pet))).resize((256,256)))
            canvas.create_image(1200,650,image=im_out13,anchor=NW)       
            canvas.image13 = im_out13
        
                
            p = canvas.create_rectangle(self.arg11+1200-radius, self.arg12+650-radius, self.arg11+1200+radius, self.arg12+650+radius, outline = 'red')
            if i > self.arg9+1:
                canvas.delete(p-24)
               
    
            im_out14 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(255*scalarMap_jacob.to_rgba(jacob_val_pet[miny_mri:maxy_mri,minx_mri:maxx_mri]))).resize((256,256)))
            canvas.create_image(1500,650,image=im_out14,anchor=NW)
            canvas.image14 = im_out14
    
            
            #display the zoom guidance MRI images 
            im_guide_mri = Image.fromarray(np.uint8(255*scalarMap_guide.to_rgba(self.arg5[miny_mri:maxy_mri,minx_mri:maxx_mri]))).resize((256,256))
            draw = ImageDraw.Draw(im_guide_mri)
            draw.rectangle(((128, 128), (141, 141)), outline="Red")
            im_out15 = ImageTk.PhotoImage(image=im_guide_mri)
            canvas.create_image(900,0,image=im_out15,anchor=NW)
            canvas.image15 = im_out15
        
        
            #display the zoom guidance PET images 
            im_guide_pet = Image.fromarray(np.uint8(255*scalarMap_guide.to_rgba(self.arg6[miny_mri:maxy_mri,minx_mri:maxx_mri]))).resize((256,256))
            draw = ImageDraw.Draw(im_guide_pet)
            draw.rectangle(((128, 128), (141, 141)), outline="Red")
            im_out16 = ImageTk.PhotoImage(image=im_guide_pet)
            canvas.create_image(900,650,image=im_out16,anchor=NW)
            canvas.image16 = im_out16
        
        
            #display the zoom fused RGB images
            im_fused_RGB = Image.fromarray(np.uint8(self.arg8[miny_mri:maxy_mri,minx_mri:maxx_mri]*255)).resize((256,256))
            #im_guide_pet = im_guide_pet.convert("RGB")
            draw = ImageDraw.Draw(im_fused_RGB)
            draw.rectangle(((128, 128), (141, 141)), outline="Red")
            im_out17 = ImageTk.PhotoImage(image=im_fused_RGB)
            canvas.create_image(900,320,image=im_out17,anchor=NW)
            canvas.image17 = im_out17 
            
            im_out2 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(255*scalarMap_guide.to_rgba(self.arg5))).resize((256,256)))
            canvas.create_image(600,0,image=im_out2,anchor=NW)
            canvas.image2 = im_out2
            
            k = canvas.create_rectangle(self.arg11+600-radius, self.arg12-radius, self.arg11+600+radius, self.arg12+radius, outline = 'red')
            if i>self.arg9+1:
                canvas.delete(k-24)        
        
            im_out3 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(255*scalarMap_guide.to_rgba(self.arg6))).resize((256,256)))
            canvas.create_image(600,650,image=im_out3,anchor=NW)
            canvas.image3 = im_out3   
            
            l = canvas.create_rectangle(self.arg11+600-radius, self.arg12+650-radius, self.arg11+600+radius, self.arg12+650+radius, outline = 'red')
            if i>self.arg9+1:
                canvas.delete(l-24)
          
            fig = plt.figure(figsize=(0.2, 23))
            ax1 = fig.add_axes([0.05, 0.80, 0.9, 0.15])
            cb1 = mpl.colorbar.ColorbarBase(ax1, cmap=cmap,norm=cNorm_jacob,orientation='vertical')
            plt.plot()
            plt.savefig('C:/Users/........./FuseVis/Guidance images/colorbar.png', bbox_inches = 'tight',pad_inches = 0)
            plt.close()
            im_out50 = ImageTk.PhotoImage(file ='C:/Users/........./FuseVis/Guidance images/colorbar.png')
            
            fig1 = plt.figure(figsize=(0.2, 23))
            ax2 = fig1.add_axes([0.05, 0.80, 0.9, 0.15])
            cb2 = mpl.colorbar.ColorbarBase(ax2, cmap=cmap,norm=cNorm_guide,orientation='vertical')
            plt.plot()
            plt.savefig('C:/Users/........./FuseVis/Guidance images/colorbar.png', bbox_inches = 'tight',pad_inches = 0)
            plt.close()
            im_out51 = ImageTk.PhotoImage(file ='C:/Users/........./FuseVis/Guidance images/colorbar.png')
            
            canvas.create_image(1152,650,image=im_out51,anchor=NW)
            canvas.image50 = im_out51
            
            canvas.create_image(1752,650,image=im_out50,anchor=NW)
            canvas.image51 = im_out50
        
            canvas.create_image(1752,4,image=im_out50,anchor=NW)
            canvas.image52 = im_out50  
        
            canvas.create_image(1152,4,image=im_out51,anchor=NW)
            canvas.image53 = im_out51    
            
            
        elif self.arg11 >= 0 and self.arg11 <= 255 and self.arg12 >= 320 and self.arg12 <= 575:
            #time1 = time.time()
            print('mouse position is at' + '(' + str(self.arg12-320) + ',' + str(self.arg11) + ')', end='\r')
            #display the output MRI Jacobian image
            jacobian_fuse_mri = torch.autograd.grad(self.arg1[0,0,self.arg12-320,self.arg11], input1_tensor, retain_graph=True, create_graph=True)[0]
            jacobian_fuse_pet = torch.autograd.grad(self.arg1[0,0,self.arg12-320,self.arg11], input2_tensor, retain_graph=True, create_graph=True)[0]

            jacob_ = torch.autograd.grad(self.arg7[0,0,self.arg12-320,self.arg11], input1_tensor, retain_graph=True, create_graph=True)[0]

            jacob_val_mri = np.squeeze(jacobian_fuse_mri.data.cpu().numpy())    
            jacob_val_pet = np.squeeze(jacobian_fuse_pet.data.cpu().numpy())
            jacob_val_numpy = np.squeeze(jacob_.data.cpu().numpy())

            x_mri = np.asarray(np.where(np.any(jacob_val_numpy, axis = 0)))
            y_mri = np.asarray(np.where(np.any(jacob_val_numpy, axis = 1)))
            minx_mri, maxx_mri, miny_mri, maxy_mri = np.min(x_mri), np.max(x_mri), np.min(y_mri), np.max(y_mri)  #return min and max coordinates
       
        
            radius = 7
            i = canvas.create_rectangle(self.arg11-radius, self.arg12-320-radius, self.arg11+radius, self.arg12-320+radius, outline = 'red')
            
            j = canvas.create_rectangle(self.arg11-radius, self.arg12-320+650-radius, self.arg11+radius, self.arg12-320+650+radius, outline = 'red')
        
            m = canvas.create_rectangle(self.arg11+600-radius, self.arg12-320+320-radius, self.arg11+600+radius, self.arg12-320+320+radius, outline = 'red') 
        
            n = canvas.create_rectangle(self.arg11-radius, self.arg12-320+320-radius, self.arg11+radius, self.arg12-320+320+radius, outline = 'red')
            if i > self.arg9+1:
                canvas.delete(i-24)
                canvas.delete(j-24)
                canvas.delete(m-24)
                canvas.delete(n-24)
                
            #color map details for the zoom images
            cmap = plt.get_cmap('viridis')
            #cNorm = mpl.colors.Normalize(vmin=self.arg3, vmax=self.arg4) #re-wrapping normalization
            cNorm_guide = mpl.colors.PowerNorm(gamma=self.arg10, vmin=self.arg3, vmax=self.arg4)
            scalarMap_guide = mpl.cm.ScalarMappable(norm=cNorm_guide, cmap=cmap)
            cNorm_jacob = mpl.colors.PowerNorm(gamma=self.arg14, vmin=self.arg3, vmax=self.arg4)
            scalarMap_jacob = mpl.cm.ScalarMappable(norm=cNorm_jacob, cmap=cmap)
        
            path = 'C:/Users/........./FuseVis/Guidance images/'

            im_fused = Image.fromarray(np.uint8(self.arg13[miny_mri:maxy_mri,minx_mri:maxx_mri]*255)).resize((256,256))
            im_fused = im_fused.convert("RGB")
            draw = ImageDraw.Draw(im_fused)
            draw.rectangle(((128, 128), (141, 141)), outline="Red")        
            im_out7 = ImageTk.PhotoImage(image=im_fused)
            canvas.create_image(300,320,image=im_out7,anchor=NW)
            canvas.image7 = im_out7
            #plt.tight_layout()
    
            im_mri = Image.fromarray(np.uint8(input1[0,0,miny_mri:maxy_mri,minx_mri:maxx_mri]*255)).resize((256,256))
            im_mri = im_mri.convert("RGB")
            draw = ImageDraw.Draw(im_mri)
            draw.rectangle(((128, 128), (141, 141)), outline="Red")
            im_out8 = ImageTk.PhotoImage(image=im_mri)
            canvas.create_image(300,0,image=im_out8,anchor=NW)
            canvas.image8 = im_out8
    
            im_pet = Image.fromarray(np.uint8(input2[0,0,miny_mri:maxy_mri,minx_mri:maxx_mri]*255)).resize((256,256)) 
            im_pet = im_pet.convert("RGB")
            draw = ImageDraw.Draw(im_pet)
            draw.rectangle(((128, 128), (141, 141)), outline="Red")
            im_out9 = ImageTk.PhotoImage(image=im_pet)
            canvas.create_image(300,650,image=im_out9,anchor=NW)
            canvas.image9 = im_out9  

            im_out10 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(255*scalarMap_jacob.to_rgba(jacob_val_mri))).resize((256,256)))
            canvas.create_image(1200,0,image=im_out10,anchor=NW)
            canvas.image10 = im_out10
        
            o = canvas.create_rectangle(self.arg11+1200-radius, self.arg12-320-radius, self.arg11+1200+radius, self.arg12-320+radius, outline = 'red')
            if i > self.arg9+1:
                canvas.delete(o-24)

            im_out11 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(255*scalarMap_jacob.to_rgba(jacob_val_mri[miny_mri:maxy_mri,minx_mri:maxx_mri]))).resize((256,256)))
            canvas.create_image(1500,0,image=im_out11,anchor=NW)
            canvas.image11 = im_out11
        
            im_out13 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(255*scalarMap_jacob.to_rgba(jacob_val_pet))).resize((256,256)))
            canvas.create_image(1200,650,image=im_out13,anchor=NW)       
            canvas.image13 = im_out13
                
            p = canvas.create_rectangle(self.arg11+1200-radius, self.arg12-320+650-radius, self.arg11+1200+radius, self.arg12-320+650+radius, outline = 'red')
            if i > self.arg9+1:
                canvas.delete(p-24)               
    
            im_out14 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(255*scalarMap_jacob.to_rgba(jacob_val_pet[miny_mri:maxy_mri,minx_mri:maxx_mri]))).resize((256,256)))
            canvas.create_image(1500,650,image=im_out14,anchor=NW)
            canvas.i4age14 = im_out14
        
            #display the zoom guidance MRI images 
            im_guide_mri = Image.fromarray(np.uint8(255*scalarMap_guide.to_rgba(self.arg5[miny_mri:maxy_mri,minx_mri:maxx_mri]))).resize((256,256))
            im_guide_mri = im_guide_mri.convert("RGB")
            draw = ImageDraw.Draw(im_guide_mri)
            draw.rectangle(((128, 128), (141, 141)), outline="Red")
            im_out15 = ImageTk.PhotoImage(image=im_guide_mri)
            canvas.create_image(900,0,image=im_out15,anchor=NW)
            canvas.image15 = im_out15
        
            #display the zoom guidance PET images 
            im_guide_pet = Image.fromarray(np.uint8(255*scalarMap_guide.to_rgba(self.arg6[miny_mri:maxy_mri,minx_mri:maxx_mri]))).resize((256,256))
            im_guide_pet = im_guide_pet.convert("RGB")
            draw = ImageDraw.Draw(im_guide_pet)
            draw.rectangle(((128, 128), (141, 141)), outline="Red")
            im_out16 = ImageTk.PhotoImage(image=im_guide_pet)
            canvas.create_image(900,650,image=im_out16,anchor=NW)
            canvas.image16 = im_out16
        
        
            #display the zoom fused RGB images
            im_fused_RGB = Image.fromarray(np.uint8(self.arg8[miny_mri:maxy_mri,minx_mri:maxx_mri]*255)).resize((256,256))
            #im_guide_pet = im_guide_pet.convert("RGB")
            draw = ImageDraw.Draw(im_fused_RGB)
            draw.rectangle(((128, 128), (141, 141)), outline="Red")
            im_out17 = ImageTk.PhotoImage(image=im_fused_RGB)
            canvas.create_image(900,320,image=im_out17,anchor=NW)
            canvas.image17 = im_out17  
            
            im_out2 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(255*scalarMap_guide.to_rgba(self.arg5))).resize((256,256)))
            canvas.create_image(600,0,image=im_out2,anchor=NW)
            canvas.image2 = im_out2
          
            k = canvas.create_rectangle(self.arg11+600-radius, self.arg12-320-radius, self.arg11+600+radius, self.arg12-320+radius, outline = 'red')
            if i > self.arg9+1:
                canvas.delete(k-24)

            im_out3 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(255*scalarMap_guide.to_rgba(self.arg6))).resize((256,256)))
            canvas.create_image(600,650,image=im_out3,anchor=NW)
            canvas.image3 = im_out3
            
            l = canvas.create_rectangle(self.arg11+600-radius, self.arg12-320+650-radius, self.arg11+600+radius, self.arg12-320+650+radius, outline = 'red')
            if i > self.arg9+1:
                canvas.delete(l-24) 
                
            fig = plt.figure(figsize=(0.2, 23))
            ax1 = fig.add_axes([0.05, 0.80, 0.9, 0.15])
            cb1 = mpl.colorbar.ColorbarBase(ax1, cmap=cmap,norm=cNorm_jacob,orientation='vertical')
            plt.plot()
            plt.savefig('C:/Users/........./FuseVis/Guidance images/colorbar.png', bbox_inches = 'tight',pad_inches = 0)
            plt.close()
            im_out50 = ImageTk.PhotoImage(file ='C:/Users/........./FuseVis/Guidance images/colorbar.png')
            
            fig1 = plt.figure(figsize=(0.2, 23))
            ax2 = fig1.add_axes([0.05, 0.80, 0.9, 0.15])
            cb2 = mpl.colorbar.ColorbarBase(ax2, cmap=cmap,norm=cNorm_guide,orientation='vertical')
            plt.plot()
            plt.savefig('C:/Users/........./FuseVis/Guidance images/colorbar.png', bbox_inches = 'tight',pad_inches = 0)
            plt.close()
            im_out51 = ImageTk.PhotoImage(file ='C:/Users/........./FuseVis/Guidance images/colorbar.png')
            
            canvas.create_image(1152,650,image=im_out51,anchor=NW)
            canvas.image50 = im_out51
            
            canvas.create_image(1752,650,image=im_out50,anchor=NW)
            canvas.image51 = im_out50
        
            canvas.create_image(1752,4,image=im_out50,anchor=NW)
            canvas.image52 = im_out50  
        
            canvas.create_image(1152,4,image=im_out51,anchor=NW)
            canvas.image53 = im_out51    

            
        elif self.arg11 >= 0 and self.arg11 <= 255 and self.arg12 >= 650 and self.arg12 <= 905:
            #time1 = time.time()
            print('mouse position is at' + '(' + str(self.arg12-320) + ',' + str(self.arg11) + ')', end='\r')
            #display the output MRI Jacobian image
            jacobian_fuse_mri = torch.autograd.grad(self.arg1[0,0,self.arg12-650,self.arg11], input1_tensor, retain_graph=True, create_graph=True)[0]
            jacobian_fuse_pet = torch.autograd.grad(self.arg1[0,0,self.arg12-650,self.arg11], input2_tensor, retain_graph=True, create_graph=True)[0]

            jacob_ = torch.autograd.grad(self.arg7[0,0,self.arg12-650,self.arg11], input1_tensor, retain_graph=True, create_graph=True)[0]

            jacob_val_mri = np.squeeze(jacobian_fuse_mri.data.cpu().numpy())    
            jacob_val_pet = np.squeeze(jacobian_fuse_pet.data.cpu().numpy())
            jacob_val_numpy = np.squeeze(jacob_.data.cpu().numpy())

            x_mri = np.asarray(np.where(np.any(jacob_val_numpy, axis = 0)))
            y_mri = np.asarray(np.where(np.any(jacob_val_numpy, axis = 1)))
            minx_mri, maxx_mri, miny_mri, maxy_mri = np.min(x_mri), np.max(x_mri), np.min(y_mri), np.max(y_mri)  #return min and max coordinates
       
        
            radius = 7
            i = canvas.create_rectangle(self.arg11-radius, self.arg12-650-radius, self.arg11+radius, self.arg12-650+radius, outline = 'red')
           
            j = canvas.create_rectangle(self.arg11-radius, self.arg12-650+650-radius, self.arg11+radius, self.arg12-650+650+radius, outline = 'red')
        
            m = canvas.create_rectangle(self.arg11+600-radius, self.arg12-650+320-radius, self.arg11+600+radius, self.arg12-650+320+radius, outline = 'red') 
        
            n = canvas.create_rectangle(self.arg11-radius, self.arg12-650+320-radius, self.arg11+radius, self.arg12-650+320+radius, outline = 'red')
            if i > self.arg9+1:
                canvas.delete(i-24)
                canvas.delete(j-24)
                canvas.delete(m-24)
                canvas.delete(n-24)

            #color map details for the zoom images
            cmap = plt.get_cmap('viridis')
            #cNorm = mpl.colors.Normalize(vmin=self.arg3, vmax=self.arg4) #re-wrapping normalization
            cNorm_guide = mpl.colors.PowerNorm(gamma=self.arg10, vmin=self.arg3, vmax=self.arg4)
            scalarMap_guide = mpl.cm.ScalarMappable(norm=cNorm_guide, cmap=cmap)
            cNorm_jacob = mpl.colors.PowerNorm(gamma=self.arg14, vmin=self.arg3, vmax=self.arg4)
            scalarMap_jacob = mpl.cm.ScalarMappable(norm=cNorm_jacob, cmap=cmap)
        
            path = 'C:/Users/........./FuseVis/Guidance images/'

            im_fused = Image.fromarray(np.uint8(self.arg13[miny_mri:maxy_mri,minx_mri:maxx_mri]*255)).resize((256,256))
            im_fused = im_fused.convert("RGB")
            draw = ImageDraw.Draw(im_fused)
            draw.rectangle(((128, 128), (141, 141)), outline="Red")        
            im_out7 = ImageTk.PhotoImage(image=im_fused)
            canvas.create_image(300,320,image=im_out7,anchor=NW)
            canvas.image7 = im_out7
            #plt.tight_layout()
    
            im_mri = Image.fromarray(np.uint8(input1[0,0,miny_mri:maxy_mri,minx_mri:maxx_mri]*255)).resize((256,256))
            im_mri = im_mri.convert("RGB")
            draw = ImageDraw.Draw(im_mri)
            draw.rectangle(((128, 128), (141, 141)), outline="Red")
            im_out8 = ImageTk.PhotoImage(image=im_mri)
            canvas.create_image(300,0,image=im_out8,anchor=NW)
            canvas.image8 = im_out8
    
            im_pet = Image.fromarray(np.uint8(input2[0,0,miny_mri:maxy_mri,minx_mri:maxx_mri]*255)).resize((256,256)) 
            im_pet = im_pet.convert("RGB")
            draw = ImageDraw.Draw(im_pet)
            draw.rectangle(((128, 128), (141, 141)), outline="Red")
            im_out9 = ImageTk.PhotoImage(image=im_pet)
            canvas.create_image(300,650,image=im_out9,anchor=NW)
            canvas.image9 = im_out9  

            im_out10 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(255*scalarMap_jacob.to_rgba(jacob_val_mri))).resize((256,256)))
            canvas.create_image(1200,0,image=im_out10,anchor=NW)
            canvas.image10 = im_out10
        
            o = canvas.create_rectangle(self.arg11+1200-radius, self.arg12-650-radius, self.arg11+1200+radius, self.arg12-650+radius, outline = 'red')
            if i > self.arg9+1:
                canvas.delete(o-24)

            im_out11 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(255*scalarMap_jacob.to_rgba(jacob_val_mri[miny_mri:maxy_mri,minx_mri:maxx_mri]))).resize((256,256)))
            canvas.create_image(1500,0,image=im_out11,anchor=NW)
            canvas.image11 = im_out11
        
            im_out13 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(255*scalarMap_jacob.to_rgba(jacob_val_pet))).resize((256,256)))
            canvas.create_image(1200,650,image=im_out13,anchor=NW)       
            canvas.image13 = im_out13
        
                
            p = canvas.create_rectangle(self.arg11+1200-radius, self.arg12-650+650-radius, self.arg11+1200+radius, self.arg12-650+650+radius, outline = 'red')
            if i > self.arg9+1:
                canvas.delete(p-24)
               
    
            im_out14 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(255*scalarMap_jacob.to_rgba(jacob_val_pet[miny_mri:maxy_mri,minx_mri:maxx_mri]))).resize((256,256)))
            canvas.create_image(1500,650,image=im_out14,anchor=NW)
            canvas.i4age14 = im_out14
        
                
            #display the zoom guidance MRI images 
            im_guide_mri = Image.fromarray(np.uint8(255*scalarMap_guide.to_rgba(self.arg5[miny_mri:maxy_mri,minx_mri:maxx_mri]))).resize((256,256))
            im_guide_mri = im_guide_mri.convert("RGB")
            draw = ImageDraw.Draw(im_guide_mri)
            draw.rectangle(((128, 128), (141, 141)), outline="Red")
            im_out15 = ImageTk.PhotoImage(image=im_guide_mri)
            canvas.create_image(900,0,image=im_out15,anchor=NW)
            canvas.image15 = im_out15
        
        
            #display the zoom guidance PET images 
            im_guide_pet = Image.fromarray(np.uint8(255*scalarMap_guide.to_rgba(self.arg6[miny_mri:maxy_mri,minx_mri:maxx_mri]))).resize((256,256))
            im_guide_pet = im_guide_pet.convert("RGB")
            draw = ImageDraw.Draw(im_guide_pet)
            draw.rectangle(((128, 128), (141, 141)), outline="Red")
            im_out16 = ImageTk.PhotoImage(image=im_guide_pet)
            canvas.create_image(900,650,image=im_out16,anchor=NW)
            canvas.image16 = im_out16
        
        
            #display the zoom fused RGB images
            im_fused_RGB = Image.fromarray(np.uint8(self.arg8[miny_mri:maxy_mri,minx_mri:maxx_mri]*255)).resize((256,256))
            #im_guide_pet = im_guide_pet.convert("RGB")
            draw = ImageDraw.Draw(im_fused_RGB)
            draw.rectangle(((128, 128), (141, 141)), outline="Red")
            im_out17 = ImageTk.PhotoImage(image=im_fused_RGB)
            canvas.create_image(900,320,image=im_out17,anchor=NW)
            canvas.image17 = im_out17  
            
            im_out2 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(255*scalarMap_guide.to_rgba(self.arg5))).resize((256,256)))
            canvas.create_image(600,0,image=im_out2,anchor=NW)
            canvas.image2 = im_out2
            
            k = canvas.create_rectangle(self.arg11+600-radius, self.arg12-650-radius, self.arg11+600+radius, self.arg12-650+radius, outline = 'red')
            if i > self.arg9+1:
                canvas.delete(k-24)
                
            im_out3 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(255*scalarMap_guide.to_rgba(self.arg6))).resize((256,256)))
            canvas.create_image(600,650,image=im_out3,anchor=NW)
            canvas.image3 = im_out3
            
            l = canvas.create_rectangle(self.arg11+600-radius, self.arg12-650+650-radius, self.arg11+600+radius, self.arg12-650+650+radius, outline = 'red')
            if i > self.arg9+1:
                canvas.delete(l-24)   
                
            fig = plt.figure(figsize=(0.2, 23))
            ax1 = fig.add_axes([0.05, 0.80, 0.9, 0.15])
            cb1 = mpl.colorbar.ColorbarBase(ax1, cmap=cmap,norm=cNorm_jacob,orientation='vertical')
            plt.plot()
            plt.savefig('C:/Users/........./FuseVis/Guidance images/colorbar.png', bbox_inches = 'tight',pad_inches = 0)
            plt.close()
            im_out50 = ImageTk.PhotoImage(file ='C:/Users/........./FuseVis/Guidance images/colorbar.png')
            
            fig1 = plt.figure(figsize=(0.2, 23))
            ax2 = fig1.add_axes([0.05, 0.80, 0.9, 0.15])
            cb2 = mpl.colorbar.ColorbarBase(ax2, cmap=cmap,norm=cNorm_guide,orientation='vertical')
            plt.plot()
            plt.savefig('C:/Users/........./FuseVis/Guidance images/colorbar.png', bbox_inches = 'tight',pad_inches = 0)
            plt.close()
            im_out51 = ImageTk.PhotoImage(file ='C:/Users/........./FuseVis/Guidance images/colorbar.png')
            
            canvas.create_image(1152,650,image=im_out51,anchor=NW)
            canvas.image50 = im_out51
            
            canvas.create_image(1752,650,image=im_out50,anchor=NW)
            canvas.image51 = im_out50
        
            canvas.create_image(1752,4,image=im_out50,anchor=NW)
            canvas.image52 = im_out50  
        
            canvas.create_image(1152,4,image=im_out51,anchor=NW)
            canvas.image53 = im_out51      
            
            
        elif self.arg11 >= 600 and self.arg11 <= 855 and self.arg12 >= 0 and self.arg12 <= 255:
            #time1 = time.time()
            print('mouse position is at' + '(' + str(self.arg12) + ',' + str(self.arg11) + ')', end='\r')
            #display the output MRI Jacobian image
            jacobian_fuse_mri = torch.autograd.grad(self.arg1[0,0,self.arg12,self.arg11-600], input1_tensor, retain_graph=True, create_graph=True)[0]
            jacobian_fuse_pet = torch.autograd.grad(self.arg1[0,0,self.arg12,self.arg11-600], input2_tensor, retain_graph=True, create_graph=True)[0]

            jacob_ = torch.autograd.grad(self.arg7[0,0,self.arg12,self.arg11-600], input1_tensor, retain_graph=True, create_graph=True)[0]

            jacob_val_mri = np.squeeze(jacobian_fuse_mri.data.cpu().numpy())    
            jacob_val_pet = np.squeeze(jacobian_fuse_pet.data.cpu().numpy())
            jacob_val_numpy = np.squeeze(jacob_.data.cpu().numpy())

            x_mri = np.asarray(np.where(np.any(jacob_val_numpy, axis = 0)))
            y_mri = np.asarray(np.where(np.any(jacob_val_numpy, axis = 1)))
            minx_mri, maxx_mri, miny_mri, maxy_mri = np.min(x_mri), np.max(x_mri), np.min(y_mri), np.max(y_mri)  #return min and max coordinates
       
        
            radius = 7
            i = canvas.create_rectangle(self.arg11-600-radius, self.arg12-radius, self.arg11-600+radius, self.arg12+radius, outline = 'red')
     
            j = canvas.create_rectangle(self.arg11-600-radius, self.arg12+650-radius, self.arg11-600+radius, self.arg12+650+radius, outline = 'red')
        
            m = canvas.create_rectangle(self.arg11-600+600-radius, self.arg12+320-radius, self.arg11-600+600+radius, self.arg12+320+radius, outline = 'red') 
        
            n = canvas.create_rectangle(self.arg11-600-radius, self.arg12+320-radius, self.arg11-600+radius, self.arg12+320+radius, outline = 'red')
            if i > self.arg9+1:
                canvas.delete(i-24)
                canvas.delete(j-24)
                canvas.delete(m-24)
                canvas.delete(n-24)
                
     
            #color map detprintails for the zoom images
            cmap = plt.get_cmap('viridis')
            #cNorm = mpl.colors.Normalize(vmin=self.arg3, vmax=self.arg4) #re-wrapping normalization
            cNorm_guide = mpl.colors.PowerNorm(gamma=self.arg10, vmin=self.arg3, vmax=self.arg4)
            scalarMap_guide = mpl.cm.ScalarMappable(norm=cNorm_guide, cmap=cmap)
            cNorm_jacob = mpl.colors.PowerNorm(gamma=self.arg14, vmin=self.arg3, vmax=self.arg4)
            scalarMap_jacob = mpl.cm.ScalarMappable(norm=cNorm_jacob, cmap=cmap)
        
            path = 'C:/Users/........./FuseVis/Guidance images/'

            im_fused = Image.fromarray(np.uint8(self.arg13[miny_mri:maxy_mri,minx_mri:maxx_mri]*255)).resize((256,256))
            im_fused = im_fused.convert("RGB")
            draw = ImageDraw.Draw(im_fused)
            draw.rectangle(((128, 128), (141, 141)), outline="Red")        
            im_out7 = ImageTk.PhotoImage(image=im_fused)
            canvas.create_image(300,320,image=im_out7,anchor=NW)
            canvas.image7 = im_out7
            #plt.tight_layout()
    
            im_mri = Image.fromarray(np.uint8(input1[0,0,miny_mri:maxy_mri,minx_mri:maxx_mri]*255)).resize((256,256))
            im_mri = im_mri.convert("RGB")
            draw = ImageDraw.Draw(im_mri)
            draw.rectangle(((128, 128), (141, 141)), outline="Red")
            im_out8 = ImageTk.PhotoImage(image=im_mri)
            canvas.create_image(300,0,image=im_out8,anchor=NW)
            canvas.image8 = im_out8
    
            im_pet = Image.fromarray(np.uint8(input2[0,0,miny_mri:maxy_mri,minx_mri:maxx_mri]*255)).resize((256,256)) 
            im_pet = im_pet.convert("RGB")
            draw = ImageDraw.Draw(im_pet)
            draw.rectangle(((128, 128), (141, 141)), outline="Red")
            im_out9 = ImageTk.PhotoImage(image=im_pet)
            canvas.create_image(300,650,image=im_out9,anchor=NW)
            canvas.image9 = im_out9  

            im_out10 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(255*scalarMap_jacob.to_rgba(jacob_val_mri))).resize((256,256)))
            canvas.create_image(1200,0,image=im_out10,anchor=NW)
            canvas.image10 = im_out10
        
            o = canvas.create_rectangle(self.arg11-600+1200-radius, self.arg12-radius, self.arg11-600+1200+radius, self.arg12+radius, outline = 'red')
            if i > self.arg9+1:
                canvas.delete(o-24)

            im_out11 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(255*scalarMap_jacob.to_rgba(jacob_val_mri[miny_mri:maxy_mri,minx_mri:maxx_mri]))).resize((256,256)))
            canvas.create_image(1500,0,image=im_out11,anchor=NW)
            canvas.image11 = im_out11
        
            im_out13 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(255*scalarMap_jacob.to_rgba(jacob_val_pet))).resize((256,256)))
            canvas.create_image(1200,650,image=im_out13,anchor=NW)       
            canvas.image13 = im_out13
        
                
            p = canvas.create_rectangle(self.arg11-600+1200-radius, self.arg12+650-radius, self.arg11-600+1200+radius, self.arg12+650+radius, outline = 'red')
            if i > self.arg9+1:
                canvas.delete(p-24)
               
    
            im_out14 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(255*scalarMap_jacob.to_rgba(jacob_val_pet[miny_mri:maxy_mri,minx_mri:maxx_mri]))).resize((256,256)))
            canvas.create_image(1500,650,image=im_out14,anchor=NW)
            canvas.i4age14 = im_out14
        
                
            #display the zoom guidance MRI images 
            im_guide_mri = Image.fromarray(np.uint8(255*scalarMap_guide.to_rgba(self.arg5[miny_mri:maxy_mri,minx_mri:maxx_mri]))).resize((256,256))
            im_guide_mri = im_guide_mri.convert("RGB")
            draw = ImageDraw.Draw(im_guide_mri)
            draw.rectangle(((128, 128), (141, 141)), outline="Red")
            im_out15 = ImageTk.PhotoImage(image=im_guide_mri)
            canvas.create_image(900,0,image=im_out15,anchor=NW)
            canvas.image15 = im_out15
        
        
            #display the zoom guidance PET images 
            im_guide_pet = Image.fromarray(np.uint8(255*scalarMap_guide.to_rgba(self.arg6[miny_mri:maxy_mri,minx_mri:maxx_mri]))).resize((256,256))
            im_guide_pet = im_guide_pet.convert("RGB")
            draw = ImageDraw.Draw(im_guide_pet)
            draw.rectangle(((128, 128), (141, 141)), outline="Red")
            im_out16 = ImageTk.PhotoImage(image=im_guide_pet)
            canvas.create_image(900,650,image=im_out16,anchor=NW)
            canvas.image16 = im_out16
        
        
            #display the zoom fused RGB images
            im_fused_RGB = Image.fromarray(np.uint8(self.arg8[miny_mri:maxy_mri,minx_mri:maxx_mri]*255)).resize((256,256))
            #im_guide_pet = im_guide_pet.convert("RGB")
            draw = ImageDraw.Draw(im_fused_RGB)
            draw.rectangle(((128, 128), (141, 141)), outline="Red")
            im_out17 = ImageTk.PhotoImage(image=im_fused_RGB)
            canvas.create_image(900,320,image=im_out17,anchor=NW)
            canvas.image17 = im_out17     
            
            im_out2 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(255*scalarMap_guide.to_rgba(self.arg5))).resize((256,256)))
            canvas.create_image(600,0,image=im_out2,anchor=NW)
            canvas.image2 = im_out2
            
            k = canvas.create_rectangle(self.arg11-600+600-radius, self.arg12-radius, self.arg11-600+600+radius, self.arg12+radius, outline = 'red')
            if i > self.arg9+1:
                canvas.delete(k-24)
          
            im_out3 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(255*scalarMap_guide.to_rgba(self.arg6))).resize((256,256)))
            canvas.create_image(600,650,image=im_out3,anchor=NW)
            canvas.image3 = im_out3
            
            l = canvas.create_rectangle(self.arg11-600+600-radius, self.arg12+650-radius, self.arg11-600+600+radius, self.arg12+650+radius, outline = 'red')
            if i > self.arg9+1:
                canvas.delete(l-24)
                
            fig = plt.figure(figsize=(0.2, 23))
            ax1 = fig.add_axes([0.05, 0.80, 0.9, 0.15])
            cb1 = mpl.colorbar.ColorbarBase(ax1, cmap=cmap,norm=cNorm_jacob,orientation='vertical')
            plt.plot()
            plt.savefig('C:/Users/........./FuseVis/Guidance images/colorbar.png', bbox_inches = 'tight',pad_inches = 0)
            plt.close()
            im_out50 = ImageTk.PhotoImage(file ='C:/Users/........./FuseVis/Guidance images/colorbar.png')
            
            fig1 = plt.figure(figsize=(0.2, 23))
            ax2 = fig1.add_axes([0.05, 0.80, 0.9, 0.15])
            cb2 = mpl.colorbar.ColorbarBase(ax2, cmap=cmap,norm=cNorm_guide,orientation='vertical')
            plt.plot()
            plt.savefig('C:/Users/........./FuseVis/Guidance images/colorbar.png', bbox_inches = 'tight',pad_inches = 0)
            plt.close()
            im_out51 = ImageTk.PhotoImage(file ='C:/Users/........./FuseVis/Guidance images/colorbar.png')
            
            canvas.create_image(1152,650,image=im_out51,anchor=NW)
            canvas.image50 = im_out51
            
            canvas.create_image(1752,650,image=im_out50,anchor=NW)
            canvas.image51 = im_out50
        
            canvas.create_image(1752,4,image=im_out50,anchor=NW)
            canvas.image52 = im_out50  
        
            canvas.create_image(1152,4,image=im_out51,anchor=NW)
            canvas.image53 = im_out51 
            
            #time2=time.time()
            #print(time2-time1)
            
        elif self.arg11 >= 600 and self.arg11 <= 855 and self.arg12 >= 320 and self.arg12 <= 575:
            #time1 = time.time()
            print('mouse position is at' + '(' + str(self.arg12) + ',' + str(self.arg11) + ')', end='\r')
            #display the output MRI Jacobian image
            jacobian_fuse_mri = torch.autograd.grad(self.arg1[0,0,self.arg12-320,self.arg11-600], input1_tensor, retain_graph=True, create_graph=True)[0]
            jacobian_fuse_pet = torch.autograd.grad(self.arg1[0,0,self.arg12-320,self.arg11-600], input2_tensor, retain_graph=True, create_graph=True)[0]

            jacob_ = torch.autograd.grad(self.arg7[0,0,self.arg12-320,self.arg11-600], input1_tensor, retain_graph=True, create_graph=True)[0]

            jacob_val_mri = np.squeeze(jacobian_fuse_mri.data.cpu().numpy())    
            jacob_val_pet = np.squeeze(jacobian_fuse_pet.data.cpu().numpy())
            jacob_val_numpy = np.squeeze(jacob_.data.cpu().numpy())

            x_mri = np.asarray(np.where(np.any(jacob_val_numpy, axis = 0)))
            y_mri = np.asarray(np.where(np.any(jacob_val_numpy, axis = 1)))
            minx_mri, maxx_mri, miny_mri, maxy_mri = np.min(x_mri), np.max(x_mri), np.min(y_mri), np.max(y_mri)  #return min and max coordinates
       
        
            radius = 7
            i = canvas.create_rectangle(self.arg11-600-radius, self.arg12-320-radius, self.arg11-600+radius, self.arg12-320+radius, outline = 'red')
          
            j = canvas.create_rectangle(self.arg11-600-radius, self.arg12-320+650-radius, self.arg11-600+radius, self.arg12-320+650+radius, outline = 'red')
        
            m = canvas.create_rectangle(self.arg11-600+600-radius, self.arg12-320+320-radius, self.arg11-600+600+radius, self.arg12-320+320+radius, outline = 'red') 
        
            n = canvas.create_rectangle(self.arg11-600-radius, self.arg12-320+320-radius, self.arg11-600+radius, self.arg12-320+320+radius, outline = 'red')
            if i > self.arg9+1:
                canvas.delete(i-24)
                canvas.delete(j-24)
                canvas.delete(m-24)
                canvas.delete(n-24)
                
           
            #color map details for the zoom images
            cmap = plt.get_cmap('viridis')
            #cNorm = mpl.colors.Normalize(vmin=self.arg3, vmax=self.arg4) #re-wrapping normalization
            cNorm_guide = mpl.colors.PowerNorm(gamma=self.arg10, vmin=self.arg3, vmax=self.arg4)
            scalarMap_guide = mpl.cm.ScalarMappable(norm=cNorm_guide, cmap=cmap)
            cNorm_jacob = mpl.colors.PowerNorm(gamma=self.arg14, vmin=self.arg3, vmax=self.arg4)
            scalarMap_jacob = mpl.cm.ScalarMappable(norm=cNorm_jacob, cmap=cmap)
        
            path = 'C:/Users/........./FuseVis/Guidance images/'

            im_fused = Image.fromarray(np.uint8(self.arg13[miny_mri:maxy_mri,minx_mri:maxx_mri]*255)).resize((256,256))
            im_fused = im_fused.convert("RGB")
            draw = ImageDraw.Draw(im_fused)
            draw.rectangle(((128, 128), (141, 141)), outline="Red")        
            im_out7 = ImageTk.PhotoImage(image=im_fused)
            canvas.create_image(300,320,image=im_out7,anchor=NW)
            canvas.image7 = im_out7
            #plt.tight_layout()
    
            im_mri = Image.fromarray(np.uint8(input1[0,0,miny_mri:maxy_mri,minx_mri:maxx_mri]*255)).resize((256,256))
            im_mri = im_mri.convert("RGB")
            draw = ImageDraw.Draw(im_mri)
            draw.rectangle(((128, 128), (141, 141)), outline="Red")
            im_out8 = ImageTk.PhotoImage(image=im_mri)
            canvas.create_image(300,0,image=im_out8,anchor=NW)
            canvas.image8 = im_out8
    
            im_pet = Image.fromarray(np.uint8(input2[0,0,miny_mri:maxy_mri,minx_mri:maxx_mri]*255)).resize((256,256)) 
            im_pet = im_pet.convert("RGB")
            draw = ImageDraw.Draw(im_pet)
            draw.rectangle(((128, 128), (141, 141)), outline="Red")
            im_out9 = ImageTk.PhotoImage(image=im_pet)
            canvas.create_image(300,650,image=im_out9,anchor=NW)
            canvas.image9 = im_out9  

            im_out10 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(255*scalarMap_jacob.to_rgba(jacob_val_mri))).resize((256,256)))
            canvas.create_image(1200,0,image=im_out10,anchor=NW)
            canvas.image10 = im_out10
        
            o = canvas.create_rectangle(self.arg11-600+1200-radius, self.arg12-320-radius, self.arg11-600+1200+radius, self.arg12-320+radius, outline = 'red')
            if i > self.arg9+1:
                canvas.delete(o-24)

            im_out11 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(255*scalarMap_jacob.to_rgba(jacob_val_mri[miny_mri:maxy_mri,minx_mri:maxx_mri]))).resize((256,256)))
            canvas.create_image(1500,0,image=im_out11,anchor=NW)
            canvas.image11 = im_out11
        
            im_out13 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(255*scalarMap_jacob.to_rgba(jacob_val_pet))).resize((256,256)))
            canvas.create_image(1200,650,image=im_out13,anchor=NW)       
            canvas.image13 = im_out13
        
                
            p = canvas.create_rectangle(self.arg11-600+1200-radius, self.arg12-320+650-radius, self.arg11-600+1200+radius, self.arg12-320+650+radius, outline = 'red')
            if i > self.arg9+1:
                canvas.delete(p-24)
               
    
            im_out14 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(255*scalarMap_jacob.to_rgba(jacob_val_pet[miny_mri:maxy_mri,minx_mri:maxx_mri]))).resize((256,256)))
            canvas.create_image(1500,650,image=im_out14,anchor=NW)
            canvas.i4age14 = im_out14
        
                
            #display the zoom guidance MRI images 
            im_guide_mri = Image.fromarray(np.uint8(255*scalarMap_guide.to_rgba(self.arg5[miny_mri:maxy_mri,minx_mri:maxx_mri]))).resize((256,256))
            im_guide_mri = im_guide_mri.convert("RGB")
            draw = ImageDraw.Draw(im_guide_mri)
            draw.rectangle(((128, 128), (141, 141)), outline="Red")
            im_out15 = ImageTk.PhotoImage(image=im_guide_mri)
            canvas.create_image(900,0,image=im_out15,anchor=NW)
            canvas.image15 = im_out15
        
        
            #display the zoom guidance PET images 
            im_guide_pet = Image.fromarray(np.uint8(255*scalarMap_guide.to_rgba(self.arg6[miny_mri:maxy_mri,minx_mri:maxx_mri]))).resize((256,256))
            im_guide_pet = im_guide_pet.convert("RGB")
            draw = ImageDraw.Draw(im_guide_pet)
            draw.rectangle(((128, 128), (141, 141)), outline="Red")
            im_out16 = ImageTk.PhotoImage(image=im_guide_pet)
            canvas.create_image(900,650,image=im_out16,anchor=NW)
            canvas.image16 = im_out16
        
        
            #display the zoom fused RGB images
            im_fused_RGB = Image.fromarray(np.uint8(self.arg8[miny_mri:maxy_mri,minx_mri:maxx_mri]*255)).resize((256,256))
            #im_guide_pet = im_guide_pet.convert("RGB")
            draw = ImageDraw.Draw(im_fused_RGB)
            draw.rectangle(((128, 128), (141, 141)), outline="Red")
            im_out17 = ImageTk.PhotoImage(image=im_fused_RGB)
            canvas.create_image(900,320,image=im_out17,anchor=NW)
            canvas.image17 = im_out17   
            
            im_out2 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(255*scalarMap_guide.to_rgba(self.arg5))).resize((256,256)))
            canvas.create_image(600,0,image=im_out2,anchor=NW)
            canvas.image2 = im_out2
            
            k = canvas.create_rectangle(self.arg11-600+600-radius, self.arg12-320-radius, self.arg11-600+600+radius, self.arg12-320+radius, outline = 'red')
            if i > self.arg9+1:
                canvas.delete(k-24)
                
            im_out3 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(255*scalarMap_guide.to_rgba(self.arg6))).resize((256,256)))
            canvas.create_image(600,650,image=im_out3,anchor=NW)
            canvas.image3 = im_out3
            
            l = canvas.create_rectangle(self.arg11-600+600-radius, self.arg12-320+650-radius, self.arg11-600+600+radius, self.arg12-320+650+radius, outline = 'red')
            if i > self.arg9+1:
                canvas.delete(l-24)   
                
            fig = plt.figure(figsize=(0.2, 23))
            ax1 = fig.add_axes([0.05, 0.80, 0.9, 0.15])
            cb1 = mpl.colorbar.ColorbarBase(ax1, cmap=cmap,norm=cNorm_jacob,orientation='vertical')
            plt.plot()
            plt.savefig('C:/Users/........./FuseVis/Guidance images/colorbar.png', bbox_inches = 'tight',pad_inches = 0)
            plt.close()
            im_out50 = ImageTk.PhotoImage(file ='C:/Users/........./FuseVis/Guidance images/colorbar.png')
            
            fig1 = plt.figure(figsize=(0.2, 23))
            ax2 = fig1.add_axes([0.05, 0.80, 0.9, 0.15])
            cb2 = mpl.colorbar.ColorbarBase(ax2, cmap=cmap,norm=cNorm_guide,orientation='vertical')
            plt.plot()
            plt.savefig('C:/Users/........./FuseVis/Guidance images/colorbar.png', bbox_inches = 'tight',pad_inches = 0)
            plt.close()
            im_out51 = ImageTk.PhotoImage(file ='C:/Users/........./FuseVis/Guidance images/colorbar.png')
            
            canvas.create_image(1152,650,image=im_out51,anchor=NW)
            canvas.image50 = im_out51
            
            canvas.create_image(1752,650,image=im_out50,anchor=NW)
            canvas.image51 = im_out50
        
            canvas.create_image(1752,4,image=im_out50,anchor=NW)
            canvas.image52 = im_out50  
        
            canvas.create_image(1152,4,image=im_out51,anchor=NW)
            canvas.image53 = im_out51     
            
            
        elif self.arg11 >= 600 and self.arg11 <= 855 and self.arg12 >= 650 and self.arg12 <= 905:
            #time1 = time.time()
            print('mouse position is at' + '(' + str(self.arg12) + ',' + str(self.arg11) + ')', end='\r')
            #display the output MRI Jacobian image
            jacobian_fuse_mri = torch.autograd.grad(self.arg1[0,0,self.arg12-650,self.arg11-600], input1_tensor, retain_graph=True, create_graph=True)[0]
            jacobian_fuse_pet = torch.autograd.grad(self.arg1[0,0,self.arg12-650,self.arg11-600], input2_tensor, retain_graph=True, create_graph=True)[0]

            jacob_ = torch.autograd.grad(self.arg7[0,0,self.arg12-650,self.arg11-600], input1_tensor, retain_graph=True, create_graph=True)[0]

            jacob_val_mri = np.squeeze(jacobian_fuse_mri.data.cpu().numpy())    
            jacob_val_pet = np.squeeze(jacobian_fuse_pet.data.cpu().numpy())
            jacob_val_numpy = np.squeeze(jacob_.data.cpu().numpy())

            x_mri = np.asarray(np.where(np.any(jacob_val_numpy, axis = 0)))
            y_mri = np.asarray(np.where(np.any(jacob_val_numpy, axis = 1)))
            minx_mri, maxx_mri, miny_mri, maxy_mri = np.min(x_mri), np.max(x_mri), np.min(y_mri), np.max(y_mri)  #return min and max coordinates
       
        
            radius = 7
            i = canvas.create_rectangle(self.arg11-600-radius, self.arg12-650-radius, self.arg11-600+radius, self.arg12-650+radius, outline = 'red')
           
            j = canvas.create_rectangle(self.arg11-600-radius, self.arg12-650+650-radius, self.arg11-600+radius, self.arg12-650+650+radius, outline = 'red')
        
            m = canvas.create_rectangle(self.arg11-600+600-radius, self.arg12-650+320-radius, self.arg11-600+600+radius, self.arg12-650+320+radius, outline = 'red') 
        
            n = canvas.create_rectangle(self.arg11-600-radius, self.arg12-650+320-radius, self.arg11-600+radius, self.arg12-650+320+radius, outline = 'red')
            if i > self.arg9+1:
                canvas.delete(i-24)
                canvas.delete(j-24)
                canvas.delete(m-24)
                canvas.delete(n-24)
                
        
            #color map details for the zoom images
            cmap = plt.get_cmap('viridis')
            #cNorm = mpl.colors.Normalize(vmin=self.arg3, vmax=self.arg4) #re-wrapping normalization
            cNorm_guide = mpl.colors.PowerNorm(gamma=self.arg10, vmin=self.arg3, vmax=self.arg4)
            scalarMap_guide = mpl.cm.ScalarMappable(norm=cNorm_guide, cmap=cmap)
            cNorm_jacob = mpl.colors.PowerNorm(gamma=self.arg14, vmin=self.arg3, vmax=self.arg4)
            scalarMap_jacob = mpl.cm.ScalarMappable(norm=cNorm_jacob, cmap=cmap)
        
            path = 'C:/Users/........./FuseVis/Guidance images/'

            im_fused = Image.fromarray(np.uint8(self.arg13[miny_mri:maxy_mri,minx_mri:maxx_mri]*255)).resize((256,256))
            im_fused = im_fused.convert("RGB")
            draw = ImageDraw.Draw(im_fused)
            draw.rectangle(((128, 128), (141, 141)), outline="Red")        
            im_out7 = ImageTk.PhotoImage(image=im_fused)
            canvas.create_image(300,320,image=im_out7,anchor=NW)
            canvas.image7 = im_out7
            #plt.tight_layout()
    
            im_mri = Image.fromarray(np.uint8(input1[0,0,miny_mri:maxy_mri,minx_mri:maxx_mri]*255)).resize((256,256))
            im_mri = im_mri.convert("RGB")
            draw = ImageDraw.Draw(im_mri)
            draw.rectangle(((128, 128), (141, 141)), outline="Red")
            im_out8 = ImageTk.PhotoImage(image=im_mri)
            canvas.create_image(300,0,image=im_out8,anchor=NW)
            canvas.image8 = im_out8
    
            im_pet = Image.fromarray(np.uint8(input2[0,0,miny_mri:maxy_mri,minx_mri:maxx_mri]*255)).resize((256,256)) 
            im_pet = im_pet.convert("RGB")
            draw = ImageDraw.Draw(im_pet)
            draw.rectangle(((128, 128), (141, 141)), outline="Red")
            im_out9 = ImageTk.PhotoImage(image=im_pet)
            canvas.create_image(300,650,image=im_out9,anchor=NW)
            canvas.image9 = im_out9  

            im_out10 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(255*scalarMap_jacob.to_rgba(jacob_val_mri))).resize((256,256)))
            canvas.create_image(1200,0,image=im_out10,anchor=NW)
            canvas.image10 = im_out10
        
            o = canvas.create_rectangle(self.arg11-600+1200-radius, self.arg12-650-radius, self.arg11-600+1200+radius, self.arg12-650+radius, outline = 'red')
            if i > self.arg9+1:
                canvas.delete(o-24)

            im_out11 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(255*scalarMap_jacob.to_rgba(jacob_val_mri[miny_mri:maxy_mri,minx_mri:maxx_mri]))).resize((256,256)))
            canvas.create_image(1500,0,image=im_out11,anchor=NW)
            canvas.image11 = im_out11
        
            im_out13 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(255*scalarMap_jacob.to_rgba(jacob_val_pet))).resize((256,256)))
            canvas.create_image(1200,650,image=im_out13,anchor=NW)       
            canvas.image13 = im_out13
        
                
            p = canvas.create_rectangle(self.arg11-600+1200-radius, self.arg12-650+650-radius, self.arg11-600+1200+radius, self.arg12-650+650+radius, outline = 'red')
            if i > self.arg9+1:
                canvas.delete(p-24)
               
    
            im_out14 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(255*scalarMap_jacob.to_rgba(jacob_val_pet[miny_mri:maxy_mri,minx_mri:maxx_mri]))).resize((256,256)))
            canvas.create_image(1500,650,image=im_out14,anchor=NW)
            canvas.image14 = im_out14
        
                
            #display the zoom guidance MRI images 
            im_guide_mri = Image.fromarray(np.uint8(255*scalarMap_guide.to_rgba(self.arg5[miny_mri:maxy_mri,minx_mri:maxx_mri]))).resize((256,256))
            im_guide_mri = im_guide_mri.convert("RGB")
            draw = ImageDraw.Draw(im_guide_mri)
            draw.rectangle(((128, 128), (141, 141)), outline="Red")
            im_out15 = ImageTk.PhotoImage(image=im_guide_mri)
            canvas.create_image(900,0,image=im_out15,anchor=NW)
            canvas.image15 = im_out15
        
        
            #display the zoom guidance PET images 
            im_guide_pet = Image.fromarray(np.uint8(255*scalarMap_guide.to_rgba(self.arg6[miny_mri:maxy_mri,minx_mri:maxx_mri]))).resize((256,256))
            im_guide_pet = im_guide_pet.convert("RGB")
            draw = ImageDraw.Draw(im_guide_pet)
            draw.rectangle(((128, 128), (141, 141)), outline="Red")
            im_out16 = ImageTk.PhotoImage(image=im_guide_pet)
            canvas.create_image(900,650,image=im_out16,anchor=NW)
            canvas.image16 = im_out16
        
        
            #display the zoom fused RGB images
            im_fused_RGB = Image.fromarray(np.uint8(self.arg8[miny_mri:maxy_mri,minx_mri:maxx_mri]*255)).resize((256,256))
            #im_guide_pet = im_guide_pet.convert("RGB")
            draw = ImageDraw.Draw(im_fused_RGB)
            draw.rectangle(((128, 128), (141, 141)), outline="Red")
            im_out17 = ImageTk.PhotoImage(image=im_fused_RGB)
            canvas.create_image(900,320,image=im_out17,anchor=NW)
            canvas.image17 = im_out17        
            
            im_out2 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(255*scalarMap_guide.to_rgba(self.arg5))).resize((256,256)))
            canvas.create_image(600,0,image=im_out2,anchor=NW)
            canvas.image2 = im_out2
            
            k = canvas.create_rectangle(self.arg11-600+600-radius, self.arg12-650-radius, self.arg11-600+600+radius, self.arg12-650+radius, outline = 'red')
            if i > self.arg9+1:
                canvas.delete(k-24)
                
            im_out3 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(255*scalarMap_guide.to_rgba(self.arg6))).resize((256,256)))
            canvas.create_image(600,650,image=im_out3,anchor=NW)
            canvas.image3 = im_out3
            
            l = canvas.create_rectangle(self.arg11-600+600-radius, self.arg12-650+650-radius, self.arg11-600+600+radius, self.arg12-650+650+radius, outline = 'red')
            if i > self.arg9+1:
                canvas.delete(l-24)
                
            fig = plt.figure(figsize=(0.2, 23))
            ax1 = fig.add_axes([0.05, 0.80, 0.9, 0.15])
            cb1 = mpl.colorbar.ColorbarBase(ax1, cmap=cmap,norm=cNorm_jacob,orientation='vertical')
            plt.plot()
            plt.savefig('C:/Users/........./FuseVis/Guidance images/colorbar.png', bbox_inches = 'tight',pad_inches = 0)
            plt.close()
            im_out50 = ImageTk.PhotoImage(file ='C:/Users/........./FuseVis/Guidance images/colorbar.png')
            
            fig1 = plt.figure(figsize=(0.2, 23))
            ax2 = fig1.add_axes([0.05, 0.80, 0.9, 0.15])
            cb2 = mpl.colorbar.ColorbarBase(ax2, cmap=cmap,norm=cNorm_guide,orientation='vertical')
            plt.plot()
            plt.savefig('C:/Users/........./FuseVis/Guidance images/colorbar.png', bbox_inches = 'tight',pad_inches = 0)
            plt.close()
            im_out51 = ImageTk.PhotoImage(file ='C:/Users/........./FuseVis/Guidance images/colorbar.png')
            
            canvas.create_image(1152,650,image=im_out51,anchor=NW)
            canvas.image50 = im_out51
            
            canvas.create_image(1752,650,image=im_out50,anchor=NW)
            canvas.image51 = im_out50
        
            canvas.create_image(1752,4,image=im_out50,anchor=NW)
            canvas.image52 = im_out50  
        
            canvas.create_image(1152,4,image=im_out51,anchor=NW)
            canvas.image53 = im_out51    
            
            #time2=time.time()
        return self.arg10

    def mouseover_Callback(self, mouseover_event):
        self.arg11 = mouseover_event.x #74  
        self.arg12 = mouseover_event.y #141 

        if self.arg11 >= 0 and self.arg11 <= 255 and self.arg12 >= 0 and self.arg12 <= 255:

            #time1 = time.time()
            print('mouse position is at' + '(' + str(self.arg12) + ',' + str(self.arg11) + ')', end='\r')
            #display the output MRI Jacobian image
            time1 = time.time()
            jacobian_fuse_mri = torch.autograd.grad(self.arg1[0,0,self.arg12,self.arg11], input1_tensor, retain_graph=True, create_graph=True)[0]

            jacobian_fuse_pet = torch.autograd.grad(self.arg1[0,0,self.arg12,self.arg11], input2_tensor, retain_graph=True, create_graph=True)[0]

            jacob_ = torch.autograd.grad(self.arg7[0,0,self.arg12,self.arg11], input1_tensor, retain_graph=True, create_graph=True)[0]

            jacob_val_mri = np.squeeze(jacobian_fuse_mri.data.cpu().numpy())    
            jacob_val_pet = np.squeeze(jacobian_fuse_pet.data.cpu().numpy())
            jacob_val_numpy = np.squeeze(jacob_.data.cpu().numpy())

            x_mri = np.asarray(np.where(np.any(jacob_val_numpy, axis = 0)))
            y_mri = np.asarray(np.where(np.any(jacob_val_numpy, axis = 1)))
            minx_mri, maxx_mri, miny_mri, maxy_mri = np.min(x_mri), np.max(x_mri), np.min(y_mri), np.max(y_mri)  #return min and max coordinates
               
            radius = 7
            i = canvas.create_rectangle(self.arg11-radius, self.arg12-radius, self.arg11+radius, self.arg12+radius, outline = 'red')
           
            j = canvas.create_rectangle(self.arg11-radius, self.arg12+650-radius, self.arg11+radius, self.arg12+650+radius, outline = 'red')
        
            m = canvas.create_rectangle(self.arg11+600-radius, self.arg12+320-radius, self.arg11+600+radius, self.arg12+320+radius, outline = 'red') 
        
            n = canvas.create_rectangle(self.arg11-radius, self.arg12+320-radius, self.arg11+radius, self.arg12+320+radius, outline = 'red')
            
            if i > self.arg9+1:
                canvas.delete(i-24)
                canvas.delete(j-24)
                canvas.delete(m-24)
                canvas.delete(n-24)

            #color map details for the zoom images
            cmap = plt.get_cmap('viridis')
            #cNorm = mpl.colors.Normalize(vmin=self.arg3, vmax=self.arg4) #re-wrapping normalization
            cNorm_guide = mpl.colors.PowerNorm(gamma=self.arg10, vmin=self.arg3, vmax=self.arg4)
            scalarMap_guide = mpl.cm.ScalarMappable(norm=cNorm_guide, cmap=cmap)
            cNorm_jacob = mpl.colors.PowerNorm(gamma=self.arg14, vmin=self.arg3, vmax=self.arg4)
            scalarMap_jacob = mpl.cm.ScalarMappable(norm=cNorm_jacob, cmap=cmap)
        
            path = 'C:/Users/........./FuseVis/Guidance images/'

            im_fused = Image.fromarray(np.uint8(self.arg13[miny_mri:maxy_mri,minx_mri:maxx_mri]*255)).resize((256,256))
            im_fused = im_fused.convert("RGB")
            draw = ImageDraw.Draw(im_fused)
            draw.rectangle(((128, 128), (141, 141)), outline="Red")        
            im_out7 = ImageTk.PhotoImage(image=im_fused)
            canvas.create_image(300,320,image=im_out7,anchor=NW)
            canvas.image7 = im_out7
            #plt.tight_layout()
    
            im_mri = Image.fromarray(np.uint8(input1[0,0,miny_mri:maxy_mri,minx_mri:maxx_mri]*255)).resize((256,256))
            im_mri = im_mri.convert("RGB")
            draw = ImageDraw.Draw(im_mri)
            draw.rectangle(((128, 128), (141, 141)), outline="Red")
            im_out8 = ImageTk.PhotoImage(image=im_mri)
            canvas.create_image(300,0,image=im_out8,anchor=NW)
            canvas.image8 = im_out8
    
            im_pet = Image.fromarray(np.uint8(input2[0,0,miny_mri:maxy_mri,minx_mri:maxx_mri]*255)).resize((256,256)) 
            im_pet = im_pet.convert("RGB")
            draw = ImageDraw.Draw(im_pet)
            draw.rectangle(((128, 128), (141, 141)), outline="Red")
            im_out9 = ImageTk.PhotoImage(image=im_pet)
            canvas.create_image(300,650,image=im_out9,anchor=NW)
            canvas.image9 = im_out9  

            im_out10 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(255*(scalarMap_jacob.to_rgba(jacob_val_mri)))).resize((256,256)))
            canvas.create_image(1200,0,image=im_out10,anchor=NW)
            canvas.image10 = im_out10
        
            o = canvas.create_rectangle(self.arg11+1200-radius, self.arg12-radius, self.arg11+1200+radius, self.arg12+radius, outline = 'red')
            if i > self.arg9+1:
                canvas.delete(o-24)

            im_out11 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(255*(scalarMap_jacob.to_rgba(jacob_val_mri[miny_mri:maxy_mri,minx_mri:maxx_mri])))).resize((256,256)))
            canvas.create_image(1500,0,image=im_out11,anchor=NW)
            canvas.image11 = im_out11
        
            im_out13 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(255*scalarMap_jacob.to_rgba(jacob_val_pet))).resize((256,256)))
            canvas.create_image(1200,650,image=im_out13,anchor=NW)       
            canvas.image13 = im_out13
        
                
            p = canvas.create_rectangle(self.arg11+1200-radius, self.arg12+650-radius, self.arg11+1200+radius, self.arg12+650+radius, outline = 'red')
            if i > self.arg9+1:
                canvas.delete(p-24)
               
    
            im_out14 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(255*scalarMap_jacob.to_rgba(jacob_val_pet[miny_mri:maxy_mri,minx_mri:maxx_mri]))).resize((256,256)))
            canvas.create_image(1500,650,image=im_out14,anchor=NW)
            canvas.image14 = im_out14
    
            
            #display the zoom guidance MRI images 
            im_guide_mri = Image.fromarray(np.uint8(255*scalarMap_guide.to_rgba(self.arg5[miny_mri:maxy_mri,minx_mri:maxx_mri]))).resize((256,256))
            draw = ImageDraw.Draw(im_guide_mri)
            draw.rectangle(((128, 128), (141, 141)), outline="Red")
            im_out15 = ImageTk.PhotoImage(image=im_guide_mri)
            canvas.create_image(900,0,image=im_out15,anchor=NW)
            canvas.image15 = im_out15
        
        
            #display the zoom guidance PET images 
            im_guide_pet = Image.fromarray(np.uint8(255*scalarMap_guide.to_rgba(self.arg6[miny_mri:maxy_mri,minx_mri:maxx_mri]))).resize((256,256))
            draw = ImageDraw.Draw(im_guide_pet)
            draw.rectangle(((128, 128), (141, 141)), outline="Red")
            im_out16 = ImageTk.PhotoImage(image=im_guide_pet)
            canvas.create_image(900,650,image=im_out16,anchor=NW)
            canvas.image16 = im_out16
        
        
            #display the zoom fused RGB images
            im_fused_RGB = Image.fromarray(np.uint8(self.arg8[miny_mri:maxy_mri,minx_mri:maxx_mri]*255)).resize((256,256))
            #im_guide_pet = im_guide_pet.convert("RGB")
            draw = ImageDraw.Draw(im_fused_RGB)
            draw.rectangle(((128, 128), (141, 141)), outline="Red")
            im_out17 = ImageTk.PhotoImage(image=im_fused_RGB)
            canvas.create_image(900,320,image=im_out17,anchor=NW)
            canvas.image17 = im_out17 
            
            im_out2 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(255*scalarMap_guide.to_rgba(self.arg5))).resize((256,256)))
            canvas.create_image(600,0,image=im_out2,anchor=NW)
            canvas.image2 = im_out2
            
            k = canvas.create_rectangle(self.arg11+600-radius, self.arg12-radius, self.arg11+600+radius, self.arg12+radius, outline = 'red')
            if i>self.arg9+1:
                canvas.delete(k-24)        
        
            im_out3 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(255*scalarMap_guide.to_rgba(self.arg6))).resize((256,256)))
            canvas.create_image(600,650,image=im_out3,anchor=NW)
            canvas.image3 = im_out3   
            
            l = canvas.create_rectangle(self.arg11+600-radius, self.arg12+650-radius, self.arg11+600+radius, self.arg12+650+radius, outline = 'red')
            if i>self.arg9+1:
                canvas.delete(l-24)
          
            fig = plt.figure(figsize=(0.2, 23))
            ax1 = fig.add_axes([0.05, 0.80, 0.9, 0.15])
            cb1 = mpl.colorbar.ColorbarBase(ax1, cmap=cmap,norm=cNorm_jacob,orientation='vertical')
            plt.plot()
            plt.savefig('C:/Users/........./FuseVis/Guidance images/colorbar.png', bbox_inches = 'tight',pad_inches = 0)
            plt.close()
            im_out50 = ImageTk.PhotoImage(file ='C:/Users/........./FuseVis/Guidance images/colorbar.png')
            
            fig1 = plt.figure(figsize=(0.2, 23))
            ax2 = fig1.add_axes([0.05, 0.80, 0.9, 0.15])
            cb2 = mpl.colorbar.ColorbarBase(ax2, cmap=cmap,norm=cNorm_guide,orientation='vertical')
            plt.plot()
            plt.savefig('C:/Users/........./FuseVis/Guidance images/colorbar.png', bbox_inches = 'tight',pad_inches = 0)
            plt.close()
            im_out51 = ImageTk.PhotoImage(file ='C:/Users/........./FuseVis/Guidance images/colorbar.png')
            
            canvas.create_image(1152,650,image=im_out51,anchor=NW)
            canvas.image50 = im_out51
            
            canvas.create_image(1752,650,image=im_out50,anchor=NW)
            canvas.image51 = im_out50
        
            canvas.create_image(1752,4,image=im_out50,anchor=NW)
            canvas.image52 = im_out50  
        
            canvas.create_image(1152,4,image=im_out51,anchor=NW)
            canvas.image53 = im_out51   
            
            time2=time.time()
            print(time2-time1)
            #
        elif self.arg11 >= 0 and self.arg11 <= 255 and self.arg12 >= 320 and self.arg12 <= 575:
            #time1 = time.time()
            print('mouse position is at' + '(' + str(self.arg12-320) + ',' + str(self.arg11) + ')', end='\r')
            #display the output MRI Jacobian image
            jacobian_fuse_mri = torch.autograd.grad(self.arg1[0,0,self.arg12-320,self.arg11], input1_tensor, retain_graph=True, create_graph=True)[0]
            jacobian_fuse_pet = torch.autograd.grad(self.arg1[0,0,self.arg12-320,self.arg11], input2_tensor, retain_graph=True, create_graph=True)[0]

            jacob_ = torch.autograd.grad(self.arg7[0,0,self.arg12-320,self.arg11], input1_tensor, retain_graph=True, create_graph=True)[0]

            jacob_val_mri = np.squeeze(jacobian_fuse_mri.data.cpu().numpy())    
            jacob_val_pet = np.squeeze(jacobian_fuse_pet.data.cpu().numpy())
            jacob_val_numpy = np.squeeze(jacob_.data.cpu().numpy())

            x_mri = np.asarray(np.where(np.any(jacob_val_numpy, axis = 0)))
            y_mri = np.asarray(np.where(np.any(jacob_val_numpy, axis = 1)))
            minx_mri, maxx_mri, miny_mri, maxy_mri = np.min(x_mri), np.max(x_mri), np.min(y_mri), np.max(y_mri)  #return min and max coordinates
       
        
            radius = 7
            i = canvas.create_rectangle(self.arg11-radius, self.arg12-320-radius, self.arg11+radius, self.arg12-320+radius, outline = 'red')
            
            j = canvas.create_rectangle(self.arg11-radius, self.arg12-320+650-radius, self.arg11+radius, self.arg12-320+650+radius, outline = 'red')
        
            m = canvas.create_rectangle(self.arg11+600-radius, self.arg12-320+320-radius, self.arg11+600+radius, self.arg12-320+320+radius, outline = 'red') 
        
            n = canvas.create_rectangle(self.arg11-radius, self.arg12-320+320-radius, self.arg11+radius, self.arg12-320+320+radius, outline = 'red')
            if i > self.arg9+1:
                canvas.delete(i-24)
                canvas.delete(j-24)
                canvas.delete(m-24)
                canvas.delete(n-24)
                
            #color map details for the zoom images
            cmap = plt.get_cmap('viridis')
            #cNorm = mpl.colors.Normalize(vmin=self.arg3, vmax=self.arg4) #re-wrapping normalization
            cNorm_guide = mpl.colors.PowerNorm(gamma=self.arg10, vmin=self.arg3, vmax=self.arg4)
            scalarMap_guide = mpl.cm.ScalarMappable(norm=cNorm_guide, cmap=cmap)
            cNorm_jacob = mpl.colors.PowerNorm(gamma=self.arg14, vmin=self.arg3, vmax=self.arg4)
            scalarMap_jacob = mpl.cm.ScalarMappable(norm=cNorm_jacob, cmap=cmap)
        
            path = 'C:/Users/........./FuseVis/Guidance images/'

            im_fused = Image.fromarray(np.uint8(self.arg13[miny_mri:maxy_mri,minx_mri:maxx_mri]*255)).resize((256,256))
            im_fused = im_fused.convert("RGB")
            draw = ImageDraw.Draw(im_fused)
            draw.rectangle(((128, 128), (141, 141)), outline="Red")        
            im_out7 = ImageTk.PhotoImage(image=im_fused)
            canvas.create_image(300,320,image=im_out7,anchor=NW)
            canvas.image7 = im_out7
            #plt.tight_layout()
    
            im_mri = Image.fromarray(np.uint8(input1[0,0,miny_mri:maxy_mri,minx_mri:maxx_mri]*255)).resize((256,256))
            im_mri = im_mri.convert("RGB")
            draw = ImageDraw.Draw(im_mri)
            draw.rectangle(((128, 128), (141, 141)), outline="Red")
            im_out8 = ImageTk.PhotoImage(image=im_mri)
            canvas.create_image(300,0,image=im_out8,anchor=NW)
            canvas.image8 = im_out8
    
            im_pet = Image.fromarray(np.uint8(input2[0,0,miny_mri:maxy_mri,minx_mri:maxx_mri]*255)).resize((256,256)) 
            im_pet = im_pet.convert("RGB")
            draw = ImageDraw.Draw(im_pet)
            draw.rectangle(((128, 128), (141, 141)), outline="Red")
            im_out9 = ImageTk.PhotoImage(image=im_pet)
            canvas.create_image(300,650,image=im_out9,anchor=NW)
            canvas.image9 = im_out9  

            im_out10 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(255*scalarMap_jacob.to_rgba(jacob_val_mri))).resize((256,256)))
            canvas.create_image(1200,0,image=im_out10,anchor=NW)
            canvas.image10 = im_out10
        
            o = canvas.create_rectangle(self.arg11+1200-radius, self.arg12-320-radius, self.arg11+1200+radius, self.arg12-320+radius, outline = 'red')
            if i > self.arg9+1:
                canvas.delete(o-24)

            im_out11 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(255*scalarMap_jacob.to_rgba(jacob_val_mri[miny_mri:maxy_mri,minx_mri:maxx_mri]))).resize((256,256)))
            canvas.create_image(1500,0,image=im_out11,anchor=NW)
            canvas.image11 = im_out11
        
            im_out13 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(255*scalarMap_jacob.to_rgba(jacob_val_pet))).resize((256,256)))
            canvas.create_image(1200,650,image=im_out13,anchor=NW)       
            canvas.image13 = im_out13
                
            p = canvas.create_rectangle(self.arg11+1200-radius, self.arg12-320+650-radius, self.arg11+1200+radius, self.arg12-320+650+radius, outline = 'red')
            if i > self.arg9+1:
                canvas.delete(p-24)               
    
            im_out14 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(255*scalarMap_jacob.to_rgba(jacob_val_pet[miny_mri:maxy_mri,minx_mri:maxx_mri]))).resize((256,256)))
            canvas.create_image(1500,650,image=im_out14,anchor=NW)
            canvas.i4age14 = im_out14
        
            #display the zoom guidance MRI images 
            im_guide_mri = Image.fromarray(np.uint8(255*scalarMap_guide.to_rgba(self.arg5[miny_mri:maxy_mri,minx_mri:maxx_mri]))).resize((256,256))
            im_guide_mri = im_guide_mri.convert("RGB")
            draw = ImageDraw.Draw(im_guide_mri)
            draw.rectangle(((128, 128), (141, 141)), outline="Red")
            im_out15 = ImageTk.PhotoImage(image=im_guide_mri)
            canvas.create_image(900,0,image=im_out15,anchor=NW)
            canvas.image15 = im_out15
        
            #display the zoom guidance PET images 
            im_guide_pet = Image.fromarray(np.uint8(255*scalarMap_guide.to_rgba(self.arg6[miny_mri:maxy_mri,minx_mri:maxx_mri]))).resize((256,256))
            im_guide_pet = im_guide_pet.convert("RGB")
            draw = ImageDraw.Draw(im_guide_pet)
            draw.rectangle(((128, 128), (141, 141)), outline="Red")
            im_out16 = ImageTk.PhotoImage(image=im_guide_pet)
            canvas.create_image(900,650,image=im_out16,anchor=NW)
            canvas.image16 = im_out16
        
        
            #display the zoom fused RGB images
            im_fused_RGB = Image.fromarray(np.uint8(self.arg8[miny_mri:maxy_mri,minx_mri:maxx_mri]*255)).resize((256,256))
            #im_guide_pet = im_guide_pet.convert("RGB")
            draw = ImageDraw.Draw(im_fused_RGB)
            draw.rectangle(((128, 128), (141, 141)), outline="Red")
            im_out17 = ImageTk.PhotoImage(image=im_fused_RGB)
            canvas.create_image(900,320,image=im_out17,anchor=NW)
            canvas.image17 = im_out17  
            
            im_out2 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(255*scalarMap_guide.to_rgba(self.arg5))).resize((256,256)))
            canvas.create_image(600,0,image=im_out2,anchor=NW)
            canvas.image2 = im_out2
          
            k = canvas.create_rectangle(self.arg11+600-radius, self.arg12-320-radius, self.arg11+600+radius, self.arg12-320+radius, outline = 'red')
            if i > self.arg9+1:
                canvas.delete(k-24)

            im_out3 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(255*scalarMap_guide.to_rgba(self.arg6))).resize((256,256)))
            canvas.create_image(600,650,image=im_out3,anchor=NW)
            canvas.image3 = im_out3
            
            l = canvas.create_rectangle(self.arg11+600-radius, self.arg12-320+650-radius, self.arg11+600+radius, self.arg12-320+650+radius, outline = 'red')
            if i > self.arg9+1:
                canvas.delete(l-24) 
                
            fig = plt.figure(figsize=(0.2, 23))
            ax1 = fig.add_axes([0.05, 0.80, 0.9, 0.15])
            cb1 = mpl.colorbar.ColorbarBase(ax1, cmap=cmap,norm=cNorm_jacob,orientation='vertical')
            plt.plot()
            plt.savefig('C:/Users/........./FuseVis/Guidance images/colorbar.png', bbox_inches = 'tight',pad_inches = 0)
            plt.close()
            im_out50 = ImageTk.PhotoImage(file ='C:/Users/........./FuseVis/Guidance images/colorbar.png')
            
            fig1 = plt.figure(figsize=(0.2, 23))
            ax2 = fig1.add_axes([0.05, 0.80, 0.9, 0.15])
            cb2 = mpl.colorbar.ColorbarBase(ax2, cmap=cmap,norm=cNorm_guide,orientation='vertical')
            plt.plot()
            plt.savefig('C:/Users/........./FuseVis/Guidance images/colorbar.png', bbox_inches = 'tight',pad_inches = 0)
            plt.close()
            im_out51 = ImageTk.PhotoImage(file ='C:/Users/........./FuseVis/Guidance images/colorbar.png')
            
            canvas.create_image(1152,650,image=im_out51,anchor=NW)
            canvas.image50 = im_out51
            
            canvas.create_image(1752,650,image=im_out50,anchor=NW)
            canvas.image51 = im_out50
        
            canvas.create_image(1752,4,image=im_out50,anchor=NW)
            canvas.image52 = im_out50  
        
            canvas.create_image(1152,4,image=im_out51,anchor=NW)
            canvas.image53 = im_out51    
                
            
        elif self.arg11 >= 0 and self.arg11 <= 255 and self.arg12 >= 650 and self.arg12 <= 905:
            #time1 = time.time()
            print('mouse position is at' + '(' + str(self.arg12-320) + ',' + str(self.arg11) + ')', end='\r')
            #display the output MRI Jacobian image
            jacobian_fuse_mri = torch.autograd.grad(self.arg1[0,0,self.arg12-650,self.arg11], input1_tensor, retain_graph=True, create_graph=True)[0]
            jacobian_fuse_pet = torch.autograd.grad(self.arg1[0,0,self.arg12-650,self.arg11], input2_tensor, retain_graph=True, create_graph=True)[0]

            jacob_ = torch.autograd.grad(self.arg7[0,0,self.arg12-650,self.arg11], input1_tensor, retain_graph=True, create_graph=True)[0]

            jacob_val_mri = np.squeeze(jacobian_fuse_mri.data.cpu().numpy())    
            jacob_val_pet = np.squeeze(jacobian_fuse_pet.data.cpu().numpy())
            jacob_val_numpy = np.squeeze(jacob_.data.cpu().numpy())

            x_mri = np.asarray(np.where(np.any(jacob_val_numpy, axis = 0)))
            y_mri = np.asarray(np.where(np.any(jacob_val_numpy, axis = 1)))
            minx_mri, maxx_mri, miny_mri, maxy_mri = np.min(x_mri), np.max(x_mri), np.min(y_mri), np.max(y_mri)  #return min and max coordinates
       
        
            radius = 7
            i = canvas.create_rectangle(self.arg11-radius, self.arg12-650-radius, self.arg11+radius, self.arg12-650+radius, outline = 'red')
           
            j = canvas.create_rectangle(self.arg11-radius, self.arg12-650+650-radius, self.arg11+radius, self.arg12-650+650+radius, outline = 'red')
        
            m = canvas.create_rectangle(self.arg11+600-radius, self.arg12-650+320-radius, self.arg11+600+radius, self.arg12-650+320+radius, outline = 'red') 
        
            n = canvas.create_rectangle(self.arg11-radius, self.arg12-650+320-radius, self.arg11+radius, self.arg12-650+320+radius, outline = 'red')
            if i > self.arg9+1:
                canvas.delete(i-24)
                canvas.delete(j-24)
                canvas.delete(m-24)
                canvas.delete(n-24)

            #color map details for the zoom images
            cmap = plt.get_cmap('viridis')
            #cNorm = mpl.colors.Normalize(vmin=self.arg3, vmax=self.arg4) #re-wrapping normalization
            cNorm_guide = mpl.colors.PowerNorm(gamma=self.arg10, vmin=self.arg3, vmax=self.arg4)
            scalarMap_guide = mpl.cm.ScalarMappable(norm=cNorm_guide, cmap=cmap)
            cNorm_jacob = mpl.colors.PowerNorm(gamma=self.arg14, vmin=self.arg3, vmax=self.arg4)
            scalarMap_jacob = mpl.cm.ScalarMappable(norm=cNorm_jacob, cmap=cmap)
        
            path = 'C:/Users/........./FuseVis/Guidance images/'

            im_fused = Image.fromarray(np.uint8(self.arg13[miny_mri:maxy_mri,minx_mri:maxx_mri]*255)).resize((256,256))
            im_fused = im_fused.convert("RGB")
            draw = ImageDraw.Draw(im_fused)
            draw.rectangle(((128, 128), (141, 141)), outline="Red")        
            im_out7 = ImageTk.PhotoImage(image=im_fused)
            canvas.create_image(300,320,image=im_out7,anchor=NW)
            canvas.image7 = im_out7
            #plt.tight_layout()
    
            im_mri = Image.fromarray(np.uint8(input1[0,0,miny_mri:maxy_mri,minx_mri:maxx_mri]*255)).resize((256,256))
            im_mri = im_mri.convert("RGB")
            draw = ImageDraw.Draw(im_mri)
            draw.rectangle(((128, 128), (141, 141)), outline="Red")
            im_out8 = ImageTk.PhotoImage(image=im_mri)
            canvas.create_image(300,0,image=im_out8,anchor=NW)
            canvas.image8 = im_out8
    
            im_pet = Image.fromarray(np.uint8(input2[0,0,miny_mri:maxy_mri,minx_mri:maxx_mri]*255)).resize((256,256)) 
            im_pet = im_pet.convert("RGB")
            draw = ImageDraw.Draw(im_pet)
            draw.rectangle(((128, 128), (141, 141)), outline="Red")
            im_out9 = ImageTk.PhotoImage(image=im_pet)
            canvas.create_image(300,650,image=im_out9,anchor=NW)
            canvas.image9 = im_out9  

            im_out10 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(255*scalarMap_jacob.to_rgba(jacob_val_mri))).resize((256,256)))
            canvas.create_image(1200,0,image=im_out10,anchor=NW)
            canvas.image10 = im_out10
        
            o = canvas.create_rectangle(self.arg11+1200-radius, self.arg12-650-radius, self.arg11+1200+radius, self.arg12-650+radius, outline = 'red')
            if i > self.arg9+1:
                canvas.delete(o-24)

            im_out11 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(255*scalarMap_jacob.to_rgba(jacob_val_mri[miny_mri:maxy_mri,minx_mri:maxx_mri]))).resize((256,256)))
            canvas.create_image(1500,0,image=im_out11,anchor=NW)
            canvas.image11 = im_out11
        
            im_out13 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(255*scalarMap_jacob.to_rgba(jacob_val_pet))).resize((256,256)))
            canvas.create_image(1200,650,image=im_out13,anchor=NW)       
            canvas.image13 = im_out13
        
                
            p = canvas.create_rectangle(self.arg11+1200-radius, self.arg12-650+650-radius, self.arg11+1200+radius, self.arg12-650+650+radius, outline = 'red')
            if i > self.arg9+1:
                canvas.delete(p-24)
               
    
            im_out14 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(255*scalarMap_jacob.to_rgba(jacob_val_pet[miny_mri:maxy_mri,minx_mri:maxx_mri]))).resize((256,256)))
            canvas.create_image(1500,650,image=im_out14,anchor=NW)
            canvas.i4age14 = im_out14
        
                
            #display the zoom guidance MRI images 
            im_guide_mri = Image.fromarray(np.uint8(255*scalarMap_guide.to_rgba(self.arg5[miny_mri:maxy_mri,minx_mri:maxx_mri]))).resize((256,256))
            im_guide_mri = im_guide_mri.convert("RGB")
            draw = ImageDraw.Draw(im_guide_mri)
            draw.rectangle(((128, 128), (141, 141)), outline="Red")
            im_out15 = ImageTk.PhotoImage(image=im_guide_mri)
            canvas.create_image(900,0,image=im_out15,anchor=NW)
            canvas.image15 = im_out15
        
        
            #display the zoom guidance PET images 
            im_guide_pet = Image.fromarray(np.uint8(255*scalarMap_guide.to_rgba(self.arg6[miny_mri:maxy_mri,minx_mri:maxx_mri]))).resize((256,256))
            im_guide_pet = im_guide_pet.convert("RGB")
            draw = ImageDraw.Draw(im_guide_pet)
            draw.rectangle(((128, 128), (141, 141)), outline="Red")
            im_out16 = ImageTk.PhotoImage(image=im_guide_pet)
            canvas.create_image(900,650,image=im_out16,anchor=NW)
            canvas.image16 = im_out16
        
        
            #display the zoom fused RGB images
            im_fused_RGB = Image.fromarray(np.uint8(self.arg8[miny_mri:maxy_mri,minx_mri:maxx_mri]*255)).resize((256,256))
            #im_guide_pet = im_guide_pet.convert("RGB")
            draw = ImageDraw.Draw(im_fused_RGB)
            draw.rectangle(((128, 128), (141, 141)), outline="Red")
            im_out17 = ImageTk.PhotoImage(image=im_fused_RGB)
            canvas.create_image(900,320,image=im_out17,anchor=NW)
            canvas.image17 = im_out17  
            
            im_out2 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(255*scalarMap_guide.to_rgba(self.arg5))).resize((256,256)))
            canvas.create_image(600,0,image=im_out2,anchor=NW)
            canvas.image2 = im_out2
            
            k = canvas.create_rectangle(self.arg11+600-radius, self.arg12-650-radius, self.arg11+600+radius, self.arg12-650+radius, outline = 'red')
            if i > self.arg9+1:
                canvas.delete(k-24)
                
            im_out3 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(255*scalarMap_guide.to_rgba(self.arg6))).resize((256,256)))
            canvas.create_image(600,650,image=im_out3,anchor=NW)
            canvas.image3 = im_out3
            
            l = canvas.create_rectangle(self.arg11+600-radius, self.arg12-650+650-radius, self.arg11+600+radius, self.arg12-650+650+radius, outline = 'red')
            if i > self.arg9+1:
                canvas.delete(l-24)   
                
            fig = plt.figure(figsize=(0.2, 23))
            ax1 = fig.add_axes([0.05, 0.80, 0.9, 0.15])
            cb1 = mpl.colorbar.ColorbarBase(ax1, cmap=cmap,norm=cNorm_jacob,orientation='vertical')
            plt.plot()
            plt.savefig('C:/Users/........./FuseVis/Guidance images/colorbar.png', bbox_inches = 'tight',pad_inches = 0)
            plt.close()
            im_out50 = ImageTk.PhotoImage(file ='C:/Users/........./FuseVis/Guidance images/colorbar.png')
            
            fig1 = plt.figure(figsize=(0.2, 23))
            ax2 = fig1.add_axes([0.05, 0.80, 0.9, 0.15])
            cb2 = mpl.colorbar.ColorbarBase(ax2, cmap=cmap,norm=cNorm_guide,orientation='vertical')
            plt.plot()
            plt.savefig('C:/Users/........./FuseVis/Guidance images/colorbar.png', bbox_inches = 'tight',pad_inches = 0)
            plt.close()
            im_out51 = ImageTk.PhotoImage(file ='C:/Users/........./FuseVis/Guidance images/colorbar.png')
            
            canvas.create_image(1152,650,image=im_out51,anchor=NW)
            canvas.image50 = im_out51
            
            canvas.create_image(1752,650,image=im_out50,anchor=NW)
            canvas.image51 = im_out50
        
            canvas.create_image(1752,4,image=im_out50,anchor=NW)
            canvas.image52 = im_out50  
        
            canvas.create_image(1152,4,image=im_out51,anchor=NW)
            canvas.image53 = im_out51  
            
            
        elif self.arg11 >= 600 and self.arg11 <= 855 and self.arg12 >= 0 and self.arg12 <= 255:
            #time1 = time.time()
            print('mouse position is at' + '(' + str(self.arg12) + ',' + str(self.arg11) + ')', end='\r')
            #display the output MRI Jacobian image
            jacobian_fuse_mri = torch.autograd.grad(self.arg1[0,0,self.arg12,self.arg11-600], input1_tensor, retain_graph=True, create_graph=True)[0]
            jacobian_fuse_pet = torch.autograd.grad(self.arg1[0,0,self.arg12,self.arg11-600], input2_tensor, retain_graph=True, create_graph=True)[0]

            jacob_ = torch.autograd.grad(self.arg7[0,0,self.arg12,self.arg11-600], input1_tensor, retain_graph=True, create_graph=True)[0]

            jacob_val_mri = np.squeeze(jacobian_fuse_mri.data.cpu().numpy())    
            jacob_val_pet = np.squeeze(jacobian_fuse_pet.data.cpu().numpy())
            jacob_val_numpy = np.squeeze(jacob_.data.cpu().numpy())

            x_mri = np.asarray(np.where(np.any(jacob_val_numpy, axis = 0)))
            y_mri = np.asarray(np.where(np.any(jacob_val_numpy, axis = 1)))
            minx_mri, maxx_mri, miny_mri, maxy_mri = np.min(x_mri), np.max(x_mri), np.min(y_mri), np.max(y_mri)  #return min and max coordinates
       
        
            radius = 7
            i = canvas.create_rectangle(self.arg11-600-radius, self.arg12-radius, self.arg11-600+radius, self.arg12+radius, outline = 'red')
     
            j = canvas.create_rectangle(self.arg11-600-radius, self.arg12+650-radius, self.arg11-600+radius, self.arg12+650+radius, outline = 'red')
        
            m = canvas.create_rectangle(self.arg11-600+600-radius, self.arg12+320-radius, self.arg11-600+600+radius, self.arg12+320+radius, outline = 'red') 
        
            n = canvas.create_rectangle(self.arg11-600-radius, self.arg12+320-radius, self.arg11-600+radius, self.arg12+320+radius, outline = 'red')
            if i > self.arg9+1:
                canvas.delete(i-24)
                canvas.delete(j-24)
                canvas.delete(m-24)
                canvas.delete(n-24)
                
     
            #color map detprintails for the zoom images
            cmap = plt.get_cmap('viridis')
            #cNorm = mpl.colors.Normalize(vmin=self.arg3, vmax=self.arg4) #re-wrapping normalization
            cNorm_guide = mpl.colors.PowerNorm(gamma=self.arg10, vmin=self.arg3, vmax=self.arg4)
            scalarMap_guide = mpl.cm.ScalarMappable(norm=cNorm_guide, cmap=cmap)
            cNorm_jacob = mpl.colors.PowerNorm(gamma=self.arg14, vmin=self.arg3, vmax=self.arg4)
            scalarMap_jacob = mpl.cm.ScalarMappable(norm=cNorm_jacob, cmap=cmap)
        
            path = 'C:/Users/........./FuseVis/Guidance images/'

            im_fused = Image.fromarray(np.uint8(self.arg13[miny_mri:maxy_mri,minx_mri:maxx_mri]*255)).resize((256,256))
            im_fused = im_fused.convert("RGB")
            draw = ImageDraw.Draw(im_fused)
            draw.rectangle(((128, 128), (141, 141)), outline="Red")        
            im_out7 = ImageTk.PhotoImage(image=im_fused)
            canvas.create_image(300,320,image=im_out7,anchor=NW)
            canvas.image7 = im_out7
            #plt.tight_layout()
    
            im_mri = Image.fromarray(np.uint8(input1[0,0,miny_mri:maxy_mri,minx_mri:maxx_mri]*255)).resize((256,256))
            im_mri = im_mri.convert("RGB")
            draw = ImageDraw.Draw(im_mri)
            draw.rectangle(((128, 128), (141, 141)), outline="Red")
            im_out8 = ImageTk.PhotoImage(image=im_mri)
            canvas.create_image(300,0,image=im_out8,anchor=NW)
            canvas.image8 = im_out8
    
            im_pet = Image.fromarray(np.uint8(input2[0,0,miny_mri:maxy_mri,minx_mri:maxx_mri]*255)).resize((256,256)) 
            im_pet = im_pet.convert("RGB")
            draw = ImageDraw.Draw(im_pet)
            draw.rectangle(((128, 128), (141, 141)), outline="Red")
            im_out9 = ImageTk.PhotoImage(image=im_pet)
            canvas.create_image(300,650,image=im_out9,anchor=NW)
            canvas.image9 = im_out9  

            im_out10 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(255*scalarMap_jacob.to_rgba(jacob_val_mri))).resize((256,256)))
            canvas.create_image(1200,0,image=im_out10,anchor=NW)
            canvas.image10 = im_out10
        
            o = canvas.create_rectangle(self.arg11-600+1200-radius, self.arg12-radius, self.arg11-600+1200+radius, self.arg12+radius, outline = 'red')
            if i > self.arg9+1:
                canvas.delete(o-24)

            im_out11 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(255*scalarMap_jacob.to_rgba(jacob_val_mri[miny_mri:maxy_mri,minx_mri:maxx_mri]))).resize((256,256)))
            canvas.create_image(1500,0,image=im_out11,anchor=NW)
            canvas.image11 = im_out11
        
            im_out13 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(255*scalarMap_jacob.to_rgba(jacob_val_pet))).resize((256,256)))
            canvas.create_image(1200,650,image=im_out13,anchor=NW)       
            canvas.image13 = im_out13
        
                
            p = canvas.create_rectangle(self.arg11-600+1200-radius, self.arg12+650-radius, self.arg11-600+1200+radius, self.arg12+650+radius, outline = 'red')
            if i > self.arg9+1:
                canvas.delete(p-24)
               
    
            im_out14 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(255*scalarMap_jacob.to_rgba(jacob_val_pet[miny_mri:maxy_mri,minx_mri:maxx_mri]))).resize((256,256)))
            canvas.create_image(1500,650,image=im_out14,anchor=NW)
            canvas.i4age14 = im_out14
        
                
            #display the zoom guidance MRI images 
            im_guide_mri = Image.fromarray(np.uint8(255*scalarMap_guide.to_rgba(self.arg5[miny_mri:maxy_mri,minx_mri:maxx_mri]))).resize((256,256))
            im_guide_mri = im_guide_mri.convert("RGB")
            draw = ImageDraw.Draw(im_guide_mri)
            draw.rectangle(((128, 128), (141, 141)), outline="Red")
            im_out15 = ImageTk.PhotoImage(image=im_guide_mri)
            canvas.create_image(900,0,image=im_out15,anchor=NW)
            canvas.image15 = im_out15
        
        
            #display the zoom guidance PET images 
            im_guide_pet = Image.fromarray(np.uint8(255*scalarMap_guide.to_rgba(self.arg6[miny_mri:maxy_mri,minx_mri:maxx_mri]))).resize((256,256))
            im_guide_pet = im_guide_pet.convert("RGB")
            draw = ImageDraw.Draw(im_guide_pet)
            draw.rectangle(((128, 128), (141, 141)), outline="Red")
            im_out16 = ImageTk.PhotoImage(image=im_guide_pet)
            canvas.create_image(900,650,image=im_out16,anchor=NW)
            canvas.image16 = im_out16
        
        
            #display the zoom fused RGB images
            im_fused_RGB = Image.fromarray(np.uint8(self.arg8[miny_mri:maxy_mri,minx_mri:maxx_mri]*255)).resize((256,256))
            #im_guide_pet = im_guide_pet.convert("RGB")
            draw = ImageDraw.Draw(im_fused_RGB)
            draw.rectangle(((128, 128), (141, 141)), outline="Red")
            im_out17 = ImageTk.PhotoImage(image=im_fused_RGB)
            canvas.create_image(900,320,image=im_out17,anchor=NW)
            canvas.image17 = im_out17     
            
            im_out2 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(255*scalarMap_guide.to_rgba(self.arg5))).resize((256,256)))
            canvas.create_image(600,0,image=im_out2,anchor=NW)
            canvas.image2 = im_out2
            
            k = canvas.create_rectangle(self.arg11-600+600-radius, self.arg12-radius, self.arg11-600+600+radius, self.arg12+radius, outline = 'red')
            if i > self.arg9+1:
                canvas.delete(k-24)
          
            im_out3 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(255*scalarMap_guide.to_rgba(self.arg6))).resize((256,256)))
            canvas.create_image(600,650,image=im_out3,anchor=NW)
            canvas.image3 = im_out3
            
            l = canvas.create_rectangle(self.arg11-600+600-radius, self.arg12+650-radius, self.arg11-600+600+radius, self.arg12+650+radius, outline = 'red')
            if i > self.arg9+1:
                canvas.delete(l-24)
                
            fig = plt.figure(figsize=(0.2, 23))
            ax1 = fig.add_axes([0.05, 0.80, 0.9, 0.15])
            cb1 = mpl.colorbar.ColorbarBase(ax1, cmap=cmap,norm=cNorm_jacob,orientation='vertical')
            plt.plot()
            plt.savefig('C:/Users/........./FuseVis/Guidance images/colorbar.png', bbox_inches = 'tight',pad_inches = 0)
            plt.close()
            im_out50 = ImageTk.PhotoImage(file ='C:/Users/........./FuseVis/Guidance images/colorbar.png')
            
            fig1 = plt.figure(figsize=(0.2, 23))
            ax2 = fig1.add_axes([0.05, 0.80, 0.9, 0.15])
            cb2 = mpl.colorbar.ColorbarBase(ax2, cmap=cmap,norm=cNorm_guide,orientation='vertical')
            plt.plot()
            plt.savefig('C:/Users/........./FuseVis/Guidance images/colorbar.png', bbox_inches = 'tight',pad_inches = 0)
            plt.close()
            im_out51 = ImageTk.PhotoImage(file ='C:/Users/........./FuseVis/Guidance images/colorbar.png')
            
            canvas.create_image(1152,650,image=im_out51,anchor=NW)
            canvas.image50 = im_out51
            
            canvas.create_image(1752,650,image=im_out50,anchor=NW)
            canvas.image51 = im_out50
        
            canvas.create_image(1752,4,image=im_out50,anchor=NW)
            canvas.image52 = im_out50  
        
            canvas.create_image(1152,4,image=im_out51,anchor=NW)
            canvas.image53 = im_out51   
            
            
        elif self.arg11 >= 600 and self.arg11 <= 855 and self.arg12 >= 320 and self.arg12 <= 575:
            #time1 = time.time()
            print('mouse position is at' + '(' + str(self.arg12) + ',' + str(self.arg11) + ')', end='\r')
            #display the output MRI Jacobian image
            jacobian_fuse_mri = torch.autograd.grad(self.arg1[0,0,self.arg12-320,self.arg11-600], input1_tensor, retain_graph=True, create_graph=True)[0]
            jacobian_fuse_pet = torch.autograd.grad(self.arg1[0,0,self.arg12-320,self.arg11-600], input2_tensor, retain_graph=True, create_graph=True)[0]

            jacob_ = torch.autograd.grad(self.arg7[0,0,self.arg12-320,self.arg11-600], input1_tensor, retain_graph=True, create_graph=True)[0]

            jacob_val_mri = np.squeeze(jacobian_fuse_mri.data.cpu().numpy())    
            jacob_val_pet = np.squeeze(jacobian_fuse_pet.data.cpu().numpy())
            jacob_val_numpy = np.squeeze(jacob_.data.cpu().numpy())

            x_mri = np.asarray(np.where(np.any(jacob_val_numpy, axis = 0)))
            y_mri = np.asarray(np.where(np.any(jacob_val_numpy, axis = 1)))
            minx_mri, maxx_mri, miny_mri, maxy_mri = np.min(x_mri), np.max(x_mri), np.min(y_mri), np.max(y_mri)  #return min and max coordinates
       
        
            radius = 7
            i = canvas.create_rectangle(self.arg11-600-radius, self.arg12-320-radius, self.arg11-600+radius, self.arg12-320+radius, outline = 'red')
          
            j = canvas.create_rectangle(self.arg11-600-radius, self.arg12-320+650-radius, self.arg11-600+radius, self.arg12-320+650+radius, outline = 'red')
        
            m = canvas.create_rectangle(self.arg11-600+600-radius, self.arg12-320+320-radius, self.arg11-600+600+radius, self.arg12-320+320+radius, outline = 'red') 
        
            n = canvas.create_rectangle(self.arg11-600-radius, self.arg12-320+320-radius, self.arg11-600+radius, self.arg12-320+320+radius, outline = 'red')
            if i > self.arg9+1:
                canvas.delete(i-24)
                canvas.delete(j-24)
                canvas.delete(m-24)
                canvas.delete(n-24)
                
           
            #color map details for the zoom images
            cmap = plt.get_cmap('viridis')
            #cNorm = mpl.colors.Normalize(vmin=self.arg3, vmax=self.arg4) #re-wrapping normalization
            cNorm_guide = mpl.colors.PowerNorm(gamma=self.arg10, vmin=self.arg3, vmax=self.arg4)
            scalarMap_guide = mpl.cm.ScalarMappable(norm=cNorm_guide, cmap=cmap)
            cNorm_jacob = mpl.colors.PowerNorm(gamma=self.arg14, vmin=self.arg3, vmax=self.arg4)
            scalarMap_jacob = mpl.cm.ScalarMappable(norm=cNorm_jacob, cmap=cmap)
        
            path = 'C:/Users/........./FuseVis/Guidance images/'

            im_fused = Image.fromarray(np.uint8(self.arg13[miny_mri:maxy_mri,minx_mri:maxx_mri]*255)).resize((256,256))
            im_fused = im_fused.convert("RGB")
            draw = ImageDraw.Draw(im_fused)
            draw.rectangle(((128, 128), (141, 141)), outline="Red")        
            im_out7 = ImageTk.PhotoImage(image=im_fused)
            canvas.create_image(300,320,image=im_out7,anchor=NW)
            canvas.image7 = im_out7
            #plt.tight_layout()
    
            im_mri = Image.fromarray(np.uint8(input1[0,0,miny_mri:maxy_mri,minx_mri:maxx_mri]*255)).resize((256,256))
            im_mri = im_mri.convert("RGB")
            draw = ImageDraw.Draw(im_mri)
            draw.rectangle(((128, 128), (141, 141)), outline="Red")
            im_out8 = ImageTk.PhotoImage(image=im_mri)
            canvas.create_image(300,0,image=im_out8,anchor=NW)
            canvas.image8 = im_out8
    
            im_pet = Image.fromarray(np.uint8(input2[0,0,miny_mri:maxy_mri,minx_mri:maxx_mri]*255)).resize((256,256)) 
            im_pet = im_pet.convert("RGB")
            draw = ImageDraw.Draw(im_pet)
            draw.rectangle(((128, 128), (141, 141)), outline="Red")
            im_out9 = ImageTk.PhotoImage(image=im_pet)
            canvas.create_image(300,650,image=im_out9,anchor=NW)
            canvas.image9 = im_out9  

            im_out10 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(255*scalarMap_jacob.to_rgba(jacob_val_mri))).resize((256,256)))
            canvas.create_image(1200,0,image=im_out10,anchor=NW)
            canvas.image10 = im_out10
        
            o = canvas.create_rectangle(self.arg11-600+1200-radius, self.arg12-320-radius, self.arg11-600+1200+radius, self.arg12-320+radius, outline = 'red')
            if i > self.arg9+1:
                canvas.delete(o-24)

            im_out11 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(255*scalarMap_jacob.to_rgba(jacob_val_mri[miny_mri:maxy_mri,minx_mri:maxx_mri]))).resize((256,256)))
            canvas.create_image(1500,0,image=im_out11,anchor=NW)
            canvas.image11 = im_out11
        
            im_out13 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(255*scalarMap_jacob.to_rgba(jacob_val_pet))).resize((256,256)))
            canvas.create_image(1200,650,image=im_out13,anchor=NW)       
            canvas.image13 = im_out13
        
                
            p = canvas.create_rectangle(self.arg11-600+1200-radius, self.arg12-320+650-radius, self.arg11-600+1200+radius, self.arg12-320+650+radius, outline = 'red')
            if i > self.arg9+1:
                canvas.delete(p-24)
               
    
            im_out14 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(255*scalarMap_jacob.to_rgba(jacob_val_pet[miny_mri:maxy_mri,minx_mri:maxx_mri]))).resize((256,256)))
            canvas.create_image(1500,650,image=im_out14,anchor=NW)
            canvas.image14 = im_out14
        
                
            #display the zoom guidance MRI images 
            im_guide_mri = Image.fromarray(np.uint8(255*scalarMap_guide.to_rgba(self.arg5[miny_mri:maxy_mri,minx_mri:maxx_mri]))).resize((256,256))
            im_guide_mri = im_guide_mri.convert("RGB")
            draw = ImageDraw.Draw(im_guide_mri)
            draw.rectangle(((128, 128), (141, 141)), outline="Red")
            im_out15 = ImageTk.PhotoImage(image=im_guide_mri)
            canvas.create_image(900,0,image=im_out15,anchor=NW)
            canvas.image15 = im_out15
        
        
            #display the zoom guidance PET images 
            im_guide_pet = Image.fromarray(np.uint8(255*scalarMap_guide.to_rgba(self.arg6[miny_mri:maxy_mri,minx_mri:maxx_mri]))).resize((256,256))
            im_guide_pet = im_guide_pet.convert("RGB")
            draw = ImageDraw.Draw(im_guide_pet)
            draw.rectangle(((128, 128), (141, 141)), outline="Red")
            im_out16 = ImageTk.PhotoImage(image=im_guide_pet)
            canvas.create_image(900,650,image=im_out16,anchor=NW)
            canvas.image16 = im_out16
        
        
            #display the zoom fused RGB images
            im_fused_RGB = Image.fromarray(np.uint8(self.arg8[miny_mri:maxy_mri,minx_mri:maxx_mri]*255)).resize((256,256))
            #im_guide_pet = im_guide_pet.convert("RGB")
            draw = ImageDraw.Draw(im_fused_RGB)
            draw.rectangle(((128, 128), (141, 141)), outline="Red")
            im_out17 = ImageTk.PhotoImage(image=im_fused_RGB)
            canvas.create_image(900,320,image=im_out17,anchor=NW)
            canvas.image17 = im_out17   
            
            im_out2 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(255*scalarMap_guide.to_rgba(self.arg5))).resize((256,256)))
            canvas.create_image(600,0,image=im_out2,anchor=NW)
            canvas.image2 = im_out2
            
            k = canvas.create_rectangle(self.arg11-600+600-radius, self.arg12-320-radius, self.arg11-600+600+radius, self.arg12-320+radius, outline = 'red')
            if i > self.arg9+1:
                canvas.delete(k-24)
                
            im_out3 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(255*scalarMap_guide.to_rgba(self.arg6))).resize((256,256)))
            canvas.create_image(600,650,image=im_out3,anchor=NW)
            canvas.image3 = im_out3
            
            l = canvas.create_rectangle(self.arg11-600+600-radius, self.arg12-320+650-radius, self.arg11-600+600+radius, self.arg12-320+650+radius, outline = 'red')
            if i > self.arg9+1:
                canvas.delete(l-24)   
                
            fig = plt.figure(figsize=(0.2, 23))
            ax1 = fig.add_axes([0.05, 0.80, 0.9, 0.15])
            cb1 = mpl.colorbar.ColorbarBase(ax1, cmap=cmap,norm=cNorm_jacob,orientation='vertical')
            plt.plot()
            plt.savefig('C:/Users/........./FuseVis/Guidance images/colorbar.png', bbox_inches = 'tight',pad_inches = 0)
            plt.close()
            im_out50 = ImageTk.PhotoImage(file ='C:/Users/........./FuseVis/Guidance images/colorbar.png')
            
            fig1 = plt.figure(figsize=(0.2, 23))
            ax2 = fig1.add_axes([0.05, 0.80, 0.9, 0.15])
            cb2 = mpl.colorbar.ColorbarBase(ax2, cmap=cmap,norm=cNorm_guide,orientation='vertical')
            plt.plot()
            plt.savefig('C:/Users/........./FuseVis/Guidance images/colorbar.png', bbox_inches = 'tight',pad_inches = 0)
            plt.close()
            im_out51 = ImageTk.PhotoImage(file ='C:/Users/........./FuseVis/Guidance images/colorbar.png')
            
            canvas.create_image(1152,650,image=im_out51,anchor=NW)
            canvas.image50 = im_out51
            
            canvas.create_image(1752,650,image=im_out50,anchor=NW)
            canvas.image51 = im_out50
        
            canvas.create_image(1752,4,image=im_out50,anchor=NW)
            canvas.image52 = im_out50  
        
            canvas.create_image(1152,4,image=im_out51,anchor=NW)
            canvas.image53 = im_out51   
            
            
        if self.arg11 >= 600 and self.arg11 <= 855 and self.arg12 >= 650 and self.arg12 <= 905:
            #time1 = time.time()
            print('mouse position is at' + '(' + str(self.arg12) + ',' + str(self.arg11) + ')', end='\r')
            #display the output MRI Jacobian image
            jacobian_fuse_mri = torch.autograd.grad(self.arg1[0,0,self.arg12-650,self.arg11-600], input1_tensor, retain_graph=True, create_graph=True)[0]
            jacobian_fuse_pet = torch.autograd.grad(self.arg1[0,0,self.arg12-650,self.arg11-600], input2_tensor, retain_graph=True, create_graph=True)[0]

            jacob_ = torch.autograd.grad(self.arg7[0,0,self.arg12-650,self.arg11-600], input1_tensor, retain_graph=True, create_graph=True)[0]

            jacob_val_mri = np.squeeze(jacobian_fuse_mri.data.cpu().numpy())    
            jacob_val_pet = np.squeeze(jacobian_fuse_pet.data.cpu().numpy())
            jacob_val_numpy = np.squeeze(jacob_.data.cpu().numpy())

            x_mri = np.asarray(np.where(np.any(jacob_val_numpy, axis = 0)))
            y_mri = np.asarray(np.where(np.any(jacob_val_numpy, axis = 1)))
            minx_mri, maxx_mri, miny_mri, maxy_mri = np.min(x_mri), np.max(x_mri), np.min(y_mri), np.max(y_mri)  #return min and max coordinates
       
        
            radius = 7
            i = canvas.create_rectangle(self.arg11-600-radius, self.arg12-650-radius, self.arg11-600+radius, self.arg12-650+radius, outline = 'red')
           
            j = canvas.create_rectangle(self.arg11-600-radius, self.arg12-650+650-radius, self.arg11-600+radius, self.arg12-650+650+radius, outline = 'red')
        
            m = canvas.create_rectangle(self.arg11-600+600-radius, self.arg12-650+320-radius, self.arg11-600+600+radius, self.arg12-650+320+radius, outline = 'red') 
        
            n = canvas.create_rectangle(self.arg11-600-radius, self.arg12-650+320-radius, self.arg11-600+radius, self.arg12-650+320+radius, outline = 'red')
            if i > self.arg9+1:
                canvas.delete(i-24)
                canvas.delete(j-24)
                canvas.delete(m-24)
                canvas.delete(n-24)
                
        
            #color map details for the zoom images
            cmap = plt.get_cmap('viridis')
            #cNorm = mpl.colors.Normalize(vmin=self.arg3, vmax=self.arg4) #re-wrapping normalization
            cNorm_guide = mpl.colors.PowerNorm(gamma=self.arg10, vmin=self.arg3, vmax=self.arg4)
            scalarMap_guide = mpl.cm.ScalarMappable(norm=cNorm_guide, cmap=cmap)
            cNorm_jacob = mpl.colors.PowerNorm(gamma=self.arg14, vmin=self.arg3, vmax=self.arg4)
            scalarMap_jacob = mpl.cm.ScalarMappable(norm=cNorm_jacob, cmap=cmap)
        
            path = 'C:/Users/........./FuseVis/Guidance images/'

            im_fused = Image.fromarray(np.uint8(self.arg13[miny_mri:maxy_mri,minx_mri:maxx_mri]*255)).resize((256,256))
            im_fused = im_fused.convert("RGB")
            draw = ImageDraw.Draw(im_fused)
            draw.rectangle(((128, 128), (141, 141)), outline="Red")        
            im_out7 = ImageTk.PhotoImage(image=im_fused)
            canvas.create_image(300,320,image=im_out7,anchor=NW)
            canvas.image7 = im_out7
            #plt.tight_layout()
    
            im_mri = Image.fromarray(np.uint8(input1[0,0,miny_mri:maxy_mri,minx_mri:maxx_mri]*255)).resize((256,256))
            im_mri = im_mri.convert("RGB")
            draw = ImageDraw.Draw(im_mri)
            draw.rectangle(((128, 128), (141, 141)), outline="Red")
            im_out8 = ImageTk.PhotoImage(image=im_mri)
            canvas.create_image(300,0,image=im_out8,anchor=NW)
            canvas.image8 = im_out8
    
            im_pet = Image.fromarray(np.uint8(input2[0,0,miny_mri:maxy_mri,minx_mri:maxx_mri]*255)).resize((256,256)) 
            im_pet = im_pet.convert("RGB")
            draw = ImageDraw.Draw(im_pet)
            draw.rectangle(((128, 128), (141, 141)), outline="Red")
            im_out9 = ImageTk.PhotoImage(image=im_pet)
            canvas.create_image(300,650,image=im_out9,anchor=NW)
            canvas.image9 = im_out9  

            im_out10 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(255*scalarMap_jacob.to_rgba(jacob_val_mri))).resize((256,256)))
            canvas.create_image(1200,0,image=im_out10,anchor=NW)
            canvas.image10 = im_out10
        
            o = canvas.create_rectangle(self.arg11-600+1200-radius, self.arg12-650-radius, self.arg11-600+1200+radius, self.arg12-650+radius, outline = 'red')
            if i > self.arg9+1:
                canvas.delete(o-24)

            im_out11 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(255*scalarMap_jacob.to_rgba(jacob_val_mri[miny_mri:maxy_mri,minx_mri:maxx_mri]))).resize((256,256)))
            canvas.create_image(1500,0,image=im_out11,anchor=NW)
            canvas.image11 = im_out11
        
            im_out13 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(255*scalarMap_jacob.to_rgba(jacob_val_pet))).resize((256,256)))
            canvas.create_image(1200,650,image=im_out13,anchor=NW)       
            canvas.image13 = im_out13
        
                
            p = canvas.create_rectangle(self.arg11-600+1200-radius, self.arg12-650+650-radius, self.arg11-600+1200+radius, self.arg12-650+650+radius, outline = 'red')
            if i > self.arg9+1:
                canvas.delete(p-24)
               
    
            im_out14 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(255*scalarMap_jacob.to_rgba(jacob_val_pet[miny_mri:maxy_mri,minx_mri:maxx_mri]))).resize((256,256)))
            canvas.create_image(1500,650,image=im_out14,anchor=NW)
            canvas.image14 = im_out14
        
                
            #display the zoom guidance MRI images 
            im_guide_mri = Image.fromarray(np.uint8(255*scalarMap_guide.to_rgba(self.arg5[miny_mri:maxy_mri,minx_mri:maxx_mri]))).resize((256,256))
            im_guide_mri = im_guide_mri.convert("RGB")
            draw = ImageDraw.Draw(im_guide_mri)
            draw.rectangle(((128, 128), (141, 141)), outline="Red")
            im_out15 = ImageTk.PhotoImage(image=im_guide_mri)
            canvas.create_image(900,0,image=im_out15,anchor=NW)
            canvas.image15 = im_out15
        
        
            #display the zoom guidance PET images 
            im_guide_pet = Image.fromarray(np.uint8(255*scalarMap_guide.to_rgba(self.arg6[miny_mri:maxy_mri,minx_mri:maxx_mri]))).resize((256,256))
            im_guide_pet = im_guide_pet.convert("RGB")
            draw = ImageDraw.Draw(im_guide_pet)
            draw.rectangle(((128, 128), (141, 141)), outline="Red")
            im_out16 = ImageTk.PhotoImage(image=im_guide_pet)
            canvas.create_image(900,650,image=im_out16,anchor=NW)
            canvas.image16 = im_out16
        
        
            #display the zoom fused RGB images
            im_fused_RGB = Image.fromarray(np.uint8(self.arg8[miny_mri:maxy_mri,minx_mri:maxx_mri]*255)).resize((256,256))
            #im_guide_pet = im_guide_pet.convert("RGB")
            draw = ImageDraw.Draw(im_fused_RGB)
            draw.rectangle(((128, 128), (141, 141)), outline="Red")
            im_out17 = ImageTk.PhotoImage(image=im_fused_RGB)
            canvas.create_image(900,320,image=im_out17,anchor=NW)
            canvas.image17 = im_out17        
            
            im_out2 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(255*scalarMap_guide.to_rgba(self.arg5))).resize((256,256)))
            canvas.create_image(600,0,image=im_out2,anchor=NW)
            canvas.image2 = im_out2
            
            k = canvas.create_rectangle(self.arg11-600+600-radius, self.arg12-650-radius, self.arg11-600+600+radius, self.arg12-650+radius, outline = 'red')
            if i > self.arg9+1:
                canvas.delete(k-24)
                
            im_out3 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(255*scalarMap_guide.to_rgba(self.arg6))).resize((256,256)))
            canvas.create_image(600,650,image=im_out3,anchor=NW)
            canvas.image3 = im_out3
            
            l = canvas.create_rectangle(self.arg11-600+600-radius, self.arg12-650+650-radius, self.arg11-600+600+radius, self.arg12-650+650+radius, outline = 'red')
            if i > self.arg9+1:
                canvas.delete(l-24)
                
            fig = plt.figure(figsize=(0.2, 23))
            ax1 = fig.add_axes([0.05, 0.80, 0.9, 0.15])
            cb1 = mpl.colorbar.ColorbarBase(ax1, cmap=cmap,norm=cNorm_jacob,orientation='vertical')
            plt.plot()
            plt.savefig('C:/Users/........./FuseVis/Guidance images/colorbar.png', bbox_inches = 'tight',pad_inches = 0)
            plt.close()
            im_out50 = ImageTk.PhotoImage(file ='C:/Users/........./FuseVis/Guidance images/colorbar.png')
            
            fig1 = plt.figure(figsize=(0.2, 23))
            ax2 = fig1.add_axes([0.05, 0.80, 0.9, 0.15])
            cb2 = mpl.colorbar.ColorbarBase(ax2, cmap=cmap,norm=cNorm_guide,orientation='vertical')
            plt.plot()
            plt.savefig('C:/Users/........./FuseVis/Guidance images/colorbar.png', bbox_inches = 'tight',pad_inches = 0)
            plt.close()
            im_out51 = ImageTk.PhotoImage(file ='C:/Users/........./FuseVis/Guidance images/colorbar.png')
            
            canvas.create_image(1152,650,image=im_out51,anchor=NW)
            canvas.image50 = im_out51
            
            canvas.create_image(1752,650,image=im_out50,anchor=NW)
            canvas.image51 = im_out50
        
            canvas.create_image(1752,4,image=im_out50,anchor=NW)
            canvas.image52 = im_out50  
        
            canvas.create_image(1152,4,image=im_out51,anchor=NW)
            canvas.image53 = im_out51   
            
        return self.arg12, self.arg11

def load_model(model):   
    if model == "Weighted Averaging":
        gamma = 1
        gamma1 = 1
        canvas.delete(ALL)       
        #Insert texts
        canvas.create_text(125,270,fill="black" ,font=("Purisa", 12),text="a) MRI image") 
        canvas.create_text(425,270,fill="black" ,font=("Purisa", 12),text="b) Zoom MRI")
        canvas.create_text(725,270,fill="black" ,font=("Purisa", 12),text="c) G1: Guidance (Fused pixel wrt MRI pixel)")  
        canvas.create_text(1025,270,fill="black",font=("Purisa", 12),text="d) Zoom Guidance G1") 
        canvas.create_text(1325,270,fill="black",font=("Purisa", 11),text="e) J1: Jacobian (Fused pixel wrt MRI image)")
        canvas.create_text(1625,270,fill="black",font=("Purisa", 11),text="f) Zoom Jacobian J1")
        canvas.create_text(125,590,fill="black" ,font=("Purisa", 12),text="g) Fused image")
        canvas.create_text(425,590,fill="black" ,font=("Purisa", 12),text="h) Zoom Fused")        
        canvas.create_text(725,590,fill="black" ,font=("Purisa", 12),text="i) Guidance R:G1, G: G2, B: Fused") 
        canvas.create_text(1025,590,fill="black",font=("Purisa", 12),text="j) Zoom Guidance RGB")
        canvas.create_text(125, 920,fill="black",font=("Purisa", 12),text="l) PET image")
        canvas.create_text(425, 920,fill="black",font=("Purisa", 12),text="m) Zoom PET")        
        canvas.create_text(725,920,fill="black" ,font=("Purisa", 12),text="n) G2: Guidance (Fused pixel wrt PET pixel)")
        canvas.create_text(1025,920,fill="black",font=("Purisa", 12),text="o) Zoom Guidance G2") 
        canvas.create_text(1325,920,fill="black",font=("Purisa", 11),text="p) J2: Jacobian (Fused pixel wrt PET image)")  
        canvas.create_text(1625,920,fill="black",font=("Purisa", 11),text="q) Zoom Jacobian J2")     
        
        fused_numpy = imageio.imread("C:/Users/........./FuseVis/Fused images/WeightedAveraging/Fused.png")
 

        #Insert MRI image to the canvas
        img_mri = ImageTk.PhotoImage(file ="C:/Users/........./FuseVis/Input1/Input1.png") # load the image
        canvas.create_image(0, 0, image=img_mri, anchor=NW) 
        canvas.im1 = img_mri
     
        #Insert PET image to the canvas
        img_pet = ImageTk.PhotoImage(file ="C:/Users/........./FuseVis/Input2/Input2.png") # load the image
        canvas.create_image(0, 650, image=img_pet, anchor=NW)
        canvas.im2 = img_pet

        #load the fused image and insert into the canvas
        im_out1 = ImageTk.PhotoImage(file ="C:/Users/........./FuseVis/Fused images/WeightedAveraging/Fused.png") #load the image
        canvas.create_image(0, 320, image=im_out1, anchor=NW)
        canvas.image1 = im_out1
        
        
        #load the model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #model1 =torch.load('C:/Users/horan/Desktop/FuseVis/.ipynb_checkpoints/FunFuseAn/checkpoint.pth')
        model0.eval()
        
        #predict the fused image
        fused_tensor0 = model0(input1_tensor.to(device), input2_tensor.to(device),w1_tensor, w2_tensor)
        fused_tensor0_numpy = fused_tensor0.data.cpu().numpy()
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model1 =torch.load('C:/Users/........./FuseVis/Checkpoint/FunFuseAn/checkpoint_lambda_0.99_gamma_ssim_0.47_gamma_l2_0.5.pth')
        model1.eval()

        fused_tensor1 = model1(input1_tensor.to(device), input2_tensor.to(device))
        fused_tensor1_numpy = fused_tensor1.data.cpu().numpy()
        
    
        #load the Fused wrt MRI and Fused wrt PET guidance images 
        hf = h5py.File('C:/Users/........./FuseVis/Weighted Averaging/Guidance_Input1.h5', 'r')
        guide_fused_mri =  np.array(hf.get('MRI_dataset'))
        hf.close()
        guide_mri_min = np.min(guide_fused_mri)
        guide_mri_max = np.max(guide_fused_mri)
        hf = h5py.File('C:/Users/........./FuseVis/Weighted Averaging/Guidance_Input2.h5', 'r')
        guide_fused_pet =  np.array(hf.get('PET_dataset'))
        hf.close()
        guide_pet_min = np.min(guide_fused_pet)
        guide_pet_max = np.max(guide_fused_pet)  
        if (guide_mri_max > guide_pet_max):
            max_ = guide_mri_max
        else:
            max_ = guide_pet_max
            
        min_ = 0
        #color map details for the zoom images
        cmap = plt.get_cmap('viridis')
        cNorm = mpl.colors.PowerNorm(gamma=gamma, vmin=min_, vmax=max_)
        scalarMap = mpl.cm.ScalarMappable(norm=cNorm, cmap=cmap)
        
        fig = plt.figure(figsize=(0.2, 23))
        ax1 = fig.add_axes([0.05, 0.80, 0.9, 0.15])
        cb1 = mpl.colorbar.ColorbarBase(ax1,cmap=cmap,norm=cNorm,orientation='vertical')
        plt.plot()
        plt.savefig('C:/Users/........./FuseVis/Guidance images/Weighted Averaging/colorbar.png', bbox_inches = 'tight',pad_inches = 0)
        plt.close()
        im_out50 = ImageTk.PhotoImage(file ='C:/Users/........./FuseVis/Guidance images/Weighted Averaging/colorbar.png')
        canvas.create_image(1152,650,image=im_out50,anchor=NW)
        canvas.image50 = im_out50
        
        im_out2 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(255*scalarMap.to_rgba(guide_fused_mri))).resize((256,256)))
        canvas.create_image(600,0,image=im_out2,anchor=NW)
        canvas.image2 = im_out2
        
        guide_fused_mri_orig = guide_fused_mri
          
        im_out3 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(255*scalarMap.to_rgba(guide_fused_pet))).resize((256,256)))
        canvas.create_image(600,650,image=im_out3,anchor=NW)
        canvas.image3 = im_out3
        
        guide_fused_pet_orig = guide_fused_pet
          
        
        #display the Fused RGB image
        fused_RGB = np.zeros((256,256,3),dtype=float)
        guide_fused_mri_norm = (guide_fused_mri - np.min(guide_fused_mri)) / (np.max(guide_fused_mri) - np.min(guide_fused_mri))
        guide_fused_pet_norm = (guide_fused_pet - np.min(guide_fused_pet)) / (np.max(guide_fused_pet) - np.min(guide_fused_pet))
        fused_numpy_norm = (fused_numpy - np.min(fused_numpy)) / (np.max(fused_numpy) - np.min(fused_numpy))
        fused_RGB[:,:,0]  = guide_fused_mri_norm
        fused_RGB[:,:,1]  = guide_fused_pet_norm 
        fused_RGB[:,:,2]  = fused_numpy_norm

        im_out4 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(fused_RGB*255)).resize((256,256)))
        canvas.create_image(600,320,image=im_out4,anchor=NW) 
        canvas.image4 = im_out4
        
        
        #display the scatter plot
        guide_mri_flat   = guide_fused_mri.flatten()
        guide_pet_flat   = guide_fused_pet.flatten() 
        plt.xlim(min_, max_)
        plt.ylim(min_, max_)
        plt.scatter(guide_mri_flat,guide_pet_flat,  marker='o', s=2, facecolors='0')
        plt.xlabel('G1: Guidance MRI (Fused pixel wrt MRI pixel)')
        plt.ylabel('G2: Guidance PET (Fused pixel wrt PET pixel)')
        plt.title('k) Scatter plot between guidance images', fontsize=12)
        plt.savefig('C:/Users/........./FuseVis/Guidance images/Weighted Averaging/Scatter_Plot.png', pad_inches = 0)
        plt.close()
        im_out5 = ImageTk.PhotoImage(file ='C:/Users/........./FuseVis/Guidance images/Weighted Averaging/Scatter_Plot.png')
        canvas.create_image(1250,300,image=im_out5,anchor=NW)
        canvas.image5 = im_out5
        
        canvas.create_image(1752,650,image=im_out50,anchor=NW)
        canvas.image51 = im_out50
        
        canvas.create_image(1752,0,image=im_out50,anchor=NW)
        canvas.image52 = im_out50  
        
        id_ = canvas.create_image(1152,0,image=im_out50,anchor=NW)
        canvas.image53 = im_out50         
        
        #insert button to the middleframe and link it to "Start Mouseover"
        start_mouseover(fused_tensor0, model, min_, max_, guide_fused_mri_orig, guide_fused_pet_orig,fused_tensor1, fused_RGB, id_,gamma, fused_numpy_norm,gamma1)
 
    if model == "FunFuseAn":
        gamma = 1
        gamma1=1
        canvas.delete(ALL)       
        #Insert texts
        canvas.create_text(125,270,fill="black" ,font=("Purisa", 12),text="a) MRI image") 
        canvas.create_text(425,270,fill="black" ,font=("Purisa", 12),text="b) Zoom MRI")
        canvas.create_text(725,270,fill="black" ,font=("Purisa", 12),text="c) G1: Guidance (Fused pixel wrt MRI pixel)")  
        canvas.create_text(1025,270,fill="black",font=("Purisa", 12),text="d) Zoom Guidance G1") 
        canvas.create_text(1325,270,fill="black",font=("Purisa", 11),text="e) J1: Jacobian (Fused pixel wrt MRI image)")
        canvas.create_text(1625,270,fill="black",font=("Purisa", 11),text="f) Zoom Jacobian J1")
        canvas.create_text(125,590,fill="black" ,font=("Purisa", 12),text="g) Fused image")
        canvas.create_text(425,590,fill="black" ,font=("Purisa", 12),text="h) Zoom Fused")        
        canvas.create_text(725,590,fill="black" ,font=("Purisa", 12),text="i) Guidance R:G1, G: G2, B: Fused") 
        canvas.create_text(1025,590,fill="black",font=("Purisa", 12),text="j) Zoom Guidance RGB")
        canvas.create_text(125, 920,fill="black",font=("Purisa", 12),text="l) PET image")
        canvas.create_text(425, 920,fill="black",font=("Purisa", 12),text="m) Zoom PET")        
        canvas.create_text(725,920,fill="black" ,font=("Purisa", 12),text="n) G2: Guidance (Fused pixel wrt PET pixel)")
        canvas.create_text(1025,920,fill="black",font=("Purisa", 12),text="o) Zoom Guidance G2") 
        canvas.create_text(1325,920,fill="black",font=("Purisa", 11),text="p) J2: Jacobian (Fused pixel wrt PET image)")  
        canvas.create_text(1625,920,fill="black",font=("Purisa", 11),text="q) Zoom Jacobian J2")
        
        fused_numpy = imageio.imread("C:/Users/........./FuseVis/Fused images/FunFuseAn/Fused.png")
 
        #Insert MRI image to the canvas
        img_mri = ImageTk.PhotoImage(file ="C:/Users/........./FuseVis/Input1/Input1.png") # load the image
        canvas.create_image(0, 0, image=img_mri, anchor=NW) 
        canvas.im1 = img_mri
     
        #Insert PET image to the canvas
        img_pet = ImageTk.PhotoImage(file ="C:/Users/........./FuseVis/Input2/Input2.png") # load the image
        canvas.create_image(0, 650, image=img_pet, anchor=NW)
        canvas.im2 = img_pet

        #load the fused image and insert into the canvas
        im_out1 = ImageTk.PhotoImage(file ="C:/Users/........./FuseVis/Fused images/FunFuseAn/Fused.png") #load the image
        canvas.create_image(0, 320, image=im_out1, anchor=NW)
        canvas.image1 = im_out1
        
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model1 =torch.load('C:/Users/........./FuseVis/Checkpoint/FunFuseAn/checkpoint_lambda_0.99_gamma_ssim_0.47_gamma_l2_0.5.pth')
        model1.eval()

        fused_tensor1 = model1(input1_tensor.to(device), input2_tensor.to(device))
        fused_tensor1_numpy = fused_tensor1.data.cpu().numpy()
        
    
        #load the Fused wrt MRI and Fused wrt PET guidance images 
        hf = h5py.File('C:/Users/........./FuseVis/FunFuseAn/Guidance_Input1.h5', 'r')
        guide_fused_mri =  np.array(hf.get('MRI_dataset'))
        hf.close()
        guide_mri_min = np.min(guide_fused_mri)
        guide_mri_max = np.max(guide_fused_mri)
        hf = h5py.File('C:/Users/........./FuseVis/FunFuseAn/Guidance_Input2.h5', 'r')
        guide_fused_pet =  np.array(hf.get('PET_dataset'))
        hf.close()
        guide_pet_min = np.min(guide_fused_pet)
        guide_pet_max = np.max(guide_fused_pet)  

        if (guide_mri_max > guide_pet_max):
            max_ = guide_mri_max
        else:
            max_ = guide_pet_max
        
        min_ = 0
        #color map details for the zoom images
        cmap = plt.get_cmap('viridis')
        #cNorm = mpl.colors.Normalize(vmin=min_, vmax=max_) #re-wrapping normalization
        cNorm = mpl.colors.PowerNorm(gamma=gamma, vmin=min_, vmax=max_)
        scalarMap = mpl.cm.ScalarMappable(norm=cNorm, cmap=cmap)
        
        fig = plt.figure(figsize=(0.2, 23))
        ax1 = fig.add_axes([0.05, 0.80, 0.9, 0.15])
        cb1 = mpl.colorbar.ColorbarBase(ax1,cmap=cmap,norm=cNorm,orientation='vertical')
        plt.plot()
        plt.savefig('C:/Users/........./FuseVis/Guidance images/FunFuseAn/colorbar.png', bbox_inches = 'tight',pad_inches = 0)
        plt.close()
        im_out50 = ImageTk.PhotoImage(file ='C:/Users/........./FuseVis/Guidance images/FunFuseAn/colorbar.png')
        canvas.create_image(1152,650,image=im_out50,anchor=NW)
        canvas.image50 = im_out50
        
        im_out2 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(255*scalarMap.to_rgba(guide_fused_mri))).resize((256,256)))
        canvas.create_image(600,0,image=im_out2,anchor=NW)
        canvas.image2 = im_out2
        
        guide_fused_mri_orig = guide_fused_mri
          
        im_out3 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(255*scalarMap.to_rgba(guide_fused_pet))).resize((256,256)))
        canvas.create_image(600,650,image=im_out3,anchor=NW)
        canvas.image3 = im_out3
        
        guide_fused_pet_orig = guide_fused_pet
          
        
        #display the Fused RGB image
        fused_RGB = np.zeros((256,256,3),dtype=float)
        guide_fused_mri_norm = (guide_fused_mri - np.min(guide_fused_mri)) / (np.max(guide_fused_mri) - np.min(guide_fused_mri))
        guide_fused_pet_norm = (guide_fused_pet - np.min(guide_fused_pet)) / (np.max(guide_fused_pet) - np.min(guide_fused_pet))
        fused_numpy_norm = (fused_numpy - np.min(fused_numpy)) / (np.max(fused_numpy) - np.min(fused_numpy))
        fused_RGB[:,:,0]  = guide_fused_mri_norm
        fused_RGB[:,:,1]  = guide_fused_pet_norm 
        fused_RGB[:,:,2]  = fused_numpy_norm

        im_out4 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(fused_RGB*255)).resize((256,256)))
        canvas.create_image(600,320,image=im_out4,anchor=NW) 
        canvas.image4 = im_out4
        
        
        #display the scatter plot
        guide_mri_flat   = guide_fused_mri.flatten()
        guide_pet_flat   = guide_fused_pet.flatten() 
        plt.xlim(min_, max_)
        plt.ylim(min_, max_)
        plt.scatter(guide_mri_flat,guide_pet_flat, marker='o', s=2, facecolors='0')
        plt.xlabel('G1: Guidance MRI (Fused pixel wrt MRI pixel)')
        plt.ylabel('G2: Guidance PET (Fused pixgamma1el wrt PET pixel)')
        plt.title('k) Scatter plot between guidance images', fontsize=12)
        plt.savefig('C:/Users/........./FuseVis/Guidance images/FunFuseAn/Scatter_Plot.png', pad_inches = 0)
        plt.close()
        im_out5 = ImageTk.PhotoImage(file ='C:/Users/........./FuseVis/Guidance images/FunFuseAn/Scatter_Plot.png')
        canvas.create_image(1250,300,image=im_out5,anchor=NW)
        canvas.image5 = im_out5
        
        canvas.create_image(1752,650,image=im_out50,anchor=NW)
        canvas.image51 = im_out50
        
        canvas.create_image(1752,0,image=im_out50,anchor=NW)
        canvas.image52 = im_out50  
        
        id_ = canvas.create_image(1152,0,image=im_out50,anchor=NW)
        canvas.image53 = im_out50         
        
        #insert button to the middleframe and link it to "Start Mouseover"
        start_mouseover(fused_tensor1, model, min_, max_, guide_fused_mri_orig, guide_fused_pet_orig,fused_tensor1, fused_RGB, id_,gamma,fused_numpy_norm,gamma1)
        
        
    if model == "MaskNet":
        gamma = 1
        gamma1 =1
        canvas.delete(ALL)
        
        #Insert texts
        canvas.create_text(125,270,fill="black" ,font=("Purisa", 12),text="a) MRI image") 
        canvas.create_text(425,270,fill="black" ,font=("Purisa", 12),text="b) Zoom MRI")
        canvas.create_text(725,270,fill="black" ,font=("Purisa", 12),text="c) G1: Guidance (Fused pixel wrt MRI pixel)")  
        canvas.create_text(1025,270,fill="black",font=("Purisa", 12),text="d) Zoom Guidance G1") 
        canvas.create_text(1325,270,fill="black",font=("Purisa", 11),text="e) J1: Jacobian (Fused pixel wrt MRI image)")
        canvas.create_text(1625,270,fill="black",font=("Purisa", 11),text="f) Zoom Jacobian J1")
        canvas.create_text(125,590,fill="black" ,font=("Purisa", 12),text="g) Fused image")
        canvas.create_text(425,590,fill="black" ,font=("Purisa", 12),text="h) Zoom Fused")        
        canvas.create_text(725,590,fill="black" ,font=("Purisa", 12),text="i) Guidance R:G1, G: G2, B: Fused") 
        canvas.create_text(1025,590,fill="black",font=("Purisa", 12),text="j) Zoom Guidance RGB")
        canvas.create_text(125, 920,fill="black",font=("Purisa", 12),text="l) PET image")
        canvas.create_text(425, 920,fill="black",font=("Purisa", 12),text="m) Zoom PET")        
        canvas.create_text(725,920,fill="black" ,font=("Purisa", 12),text="n) G2: Guidance (Fused pixel wrt PET pixel)")
        canvas.create_text(1025,920,fill="black",font=("Purisa", 12),text="o) Zoom Guidance G2") 
        canvas.create_text(1325,920,fill="black",font=("Purisa", 11),text="p) J2: Jacobian (Fused pixel wrt PET image)")  
        canvas.create_text(1625,920,fill="black",font=("Purisa", 11),text="q) Zoom Jacobian J2")
        
        fused_numpy = imageio.imread("C:/Users/........./FuseVis/Fused images/MaskNet/Fused.png")
        
        #Insert MRI image to the canvas
        img_mri = ImageTk.PhotoImage(file ="C:/Users/........./FuseVis/Input1/Input1.png") # load the image
        canvas.create_image(0, 0, image=img_mri, anchor=NW)
        #canvas.create_text(725,270,fill="black",font=("Purisa", 12),text="MRI image")  
        canvas.im1 = img_mri
          

        #Insert PET image to the canvas
        img_pet = ImageTk.PhotoImage(file ="C:/Users/........./FuseVis/Input2/Input2.png") # load the image
        canvas.create_image(0, 650, image=img_pet, anchor=NW)
        #canvas.create_text(1325,270,fill="black",font=("Purisa", 12),text="PET image")
        canvas.im2 = img_pet
        
        
        #load the fused image and insert into the canvas
        im_out1 = ImageTk.PhotoImage(file ="C:/Users/........./FuseVis/Fused images/MaskNet/Fused.png") #load the image
        canvas.create_image(0, 320, image=im_out1, anchor=NW)
        #canvas.create_text(125,270,fill="black",font=("Purisa", 12),text="Fused image")
        canvas.image1 = im_out1
        
        
        #load the funfusean model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model1 =torch.load('C:/Users/........./FuseVis/Checkpoint/FunFuseAn/checkpoint_lambda_0.99_gamma_ssim_0.47_gamma_l2_0.5.pth')
        model1.eval()
        
        #load the model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model4 =torch.load('C:/Users/........./FuseVis/Checkpoint/MaskNet/checkpoint_lambda_0.99_gamma_ssim_0.5_gamma_l2_0.494.pth')
        model4.eval()
        
        #predict the fused image
        fused_tensor1 = model1(input1_tensor.to(device), input2_tensor.to(device))
        fused_tensor1_numpy = fused_tensor1.data.cpu().numpy()
        
        #predict the fused image
        fused_tensor4 = model4(input1_tensor.to(device), input2_tensor.to(device))
        fused_tensor4_numpy = fused_tensor4.data.cpu().numpy()
        #load the Fused wrt MRI guidance images and normalize it
        hf = h5py.File('C:/Users/........./FuseVis/MaskNet/Guidance_Input1.h5', 'r')
        guide_fused_mri =  np.array(hf.get('MRI_dataset'))
        hf.close()
        hf = h5py.File('C:/Users/........./FuseVis/MaskNet/Guidance_Input2.h5', 'r')
        guide_fused_pet =  np.array(hf.get('PET_dataset'))
        hf.close()
        guide_mri_min = np.min(guide_fused_mri)
        guide_mri_max = np.max(guide_fused_mri)
        guide_pet_min = np.min(guide_fused_pet)
        guide_pet_max = np.max(guide_fused_pet)  
        if (guide_mri_max > guide_pet_max):
            max_ = guide_mri_max
        else:
            max_ = guide_pet_max
            
        min_ = 0
        #color map details for the zoom images
        cmap = plt.get_cmap('viridis')
        #cNorm = mpl.colors.Normalize(vmin=min_, vmax=max_) #re-wrapping normalization
        cNorm = mpl.colors.PowerNorm(gamma=gamma, vmin=min_, vmax=max_)
        scalarMap = mpl.cm.ScalarMappable(norm=cNorm, cmap=cmap)
        
        fig = plt.figure(figsize=(0.2, 23))
        ax1 = fig.add_axes([0.05, 0.80, 0.9, 0.15])
        cb1 = mpl.colorbar.ColorbarBase(ax1, cmap=cmap,norm=cNorm,orientation='vertical')
        plt.plot()
        plt.savefig('C:/Users/........./FuseVis/Guidance images/MaskNet/colorbar.png', bbox_inches = 'tight',pad_inches = 0)
        plt.close()
        im_out50 = ImageTk.PhotoImage(file ='C:/Users/........./FuseVis/Guidance images/MaskNet/colorbar.png')
        canvas.create_image(1152,650,image=im_out50,anchor=NW)
        canvas.image50 = im_out50 
        

        im_out2 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(255*(scalarMap.to_rgba(guide_fused_mri)))).resize((256,256)))
        canvas.create_image(600,0,image=im_out2,anchor=NW)
        #canvas.create_text(125,920,fill="black",font=("Purisa", 12),text="G1: Gradient map (Fused wrt MRI)")  
        canvas.image2 = im_out2
          
        guide_fused_mri_orig = guide_fused_mri
        
        im_out3 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(255*scalarMap.to_rgba(guide_fused_pet))).resize((256,256)))
        canvas.create_image(600,650,image=im_out3,anchor=NW)
        #canvas.create_text(1325,920,fill="black",font=("Purisa", 12),text="G2: Gradient map (Fused wrt PET)")
        canvas.image3 = im_out3
        
        guide_fused_pet_orig = guide_fused_pet
        
        #display the Fused RGB image
        fused_RGB = np.zeros((256,256,3),dtype=float)
        guide_fused_mri_norm = (guide_fused_mri - np.min(guide_fused_mri)) / (np.max(guide_fused_mri) - np.min(guide_fused_mri))
        guide_fused_pet_norm = (guide_fused_pet - np.min(guide_fused_pet)) / (np.max(guide_fused_pet) - np.min(guide_fused_pet))
        fused_numpy_norm = (fused_numpy - np.min(fused_numpy)) / (np.max(fused_numpy) - np.min(fused_numpy))
        fused_RGB[:,:,0]  = guide_fused_mri_norm 
        fused_RGB[:,:,1]  = guide_fused_pet_norm 
        fused_RGB[:,:,2]  = fused_numpy_norm

        im_out4 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(fused_RGB*255)).resize((256,256)))
        canvas.create_image(600,320,image=im_out4,anchor=NW)
        #canvas.create_text(875,920,fill="black",font=("Purisa", 12),text="Guidance R:G1, G: G2, B: Fused")  
        canvas.image4 = im_out4
        
        
        #display the Scatter plot image
        guide_mri_flat   = guide_fused_mri.flatten()
        guide_pet_flat   = guide_fused_pet.flatten() 
        plt.xlim(min_, max_)
        plt.ylim(min_, max_)
        plt.scatter(guide_mri_flat,guide_pet_flat, marker='o', s=2, facecolors='0')
        plt.xlabel('G1: Guidance MRI (Fused pixel wrt MRI pixel)')
        plt.ylabel('G2: Guidance PET (Fused pixel wrt PET pixel)')
        plt.title('k) Scatter plot between guidance images', fontsize=12)
        plt.savefig('C:/Users/........./FuseVis/Guidance images/MaskNet/Scatter_Plot.png', pad_inches = 0)
        plt.close()
        im_out5 = ImageTk.PhotoImage(file ='C:/Users/........./FuseVis/Guidance images/MaskNet/Scatter_Plot.png')
        canvas.create_image(1250,300,image=im_out5,anchor=NW)
        canvas.image5 = im_out5

        canvas.create_image(1752,650,image=im_out50,anchor=NW)
        canvas.image51 = im_out50
        
        canvas.create_image(1752,0,image=im_out50,anchor=NW)
        canvas.image52 = im_out50
        
        id_ = canvas.create_image(1152,0,image=im_out50,anchor=NW)
        canvas.image53 = im_out50
        
        #insert button to the middleframe and link it to "Start Mouseover"
        start_mouseover(fused_tensor4, model, min_, max_, guide_fused_mri_orig, guide_fused_pet_orig, fused_tensor1, fused_RGB, id_,gamma,fused_numpy_norm, gamma1)
        
    if model == "DeepFuse":
        gamma = 1
        gamma1 =1
        canvas.delete(ALL)
        
        #Insert texts
        canvas.create_text(125,270,fill="black" ,font=("Purisa", 12),text="a) MRI image") 
        canvas.create_text(425,270,fill="black" ,font=("Purisa", 12),text="b) Zoom MRI")
        canvas.create_text(725,270,fill="black" ,font=("Purisa", 12),text="c) G1: Guidance (Fused pixel wrt MRI pixel)")  
        canvas.create_text(1025,270,fill="black",font=("Purisa", 12),text="d) Zoom Guidance G1") 
        canvas.create_text(1325,270,fill="black",font=("Purisa", 11),text="e) J1: Jacobian (Fused pixel wrt MRI image)")
        canvas.create_text(1625,270,fill="black",font=("Purisa", 11),text="f) Zoom Jacobian J1")
        canvas.create_text(125,590,fill="black" ,font=("Purisa", 12),text="g) Fused image")
        canvas.create_text(425,590,fill="black" ,font=("Purisa", 12),text="h) Zoom Fused")        
        canvas.create_text(725,590,fill="black" ,font=("Purisa", 12),text="i) Guidance R:G1, G: G2, B: Fused") 
        canvas.create_text(1025,590,fill="black",font=("Purisa", 12),text="j) Zoom Guidance RGB")
        canvas.create_text(125, 920,fill="black",font=("Purisa", 12),text="l) PET image")
        canvas.create_text(425, 920,fill="black",font=("Purisa", 12),text="m) Zoom PET")        
        canvas.create_text(725,920,fill="black" ,font=("Purisa", 12),text="n) G2: Guidance (Fused pixel wrt PET pixel)")
        canvas.create_text(1025,920,fill="black",font=("Purisa", 12),text="o) Zoom Guidance G2") 
        canvas.create_text(1325,920,fill="black",font=("Purisa", 11),text="p) J2: Jacobian (Fused pixel wrt PET image)")  
        canvas.create_text(1625,920,fill="black",font=("Purisa", 11),text="q) Zoom Jacobian J2")
        
        fused_numpy = imageio.imread("C:/Users/........./FuseVis/Fused images/DeepFuse/Fused.png")
        
        #Insert MRI image to the canvas
        img_mri = ImageTk.PhotoImage(file ="C:/Users/........./FuseVis/Input1/Input1.png") # load the image
        canvas.create_image(0, 0, image=img_mri, anchor=NW)
        #canvas.create_text(725,270,fill="black",font=("Purisa", 12),text="MRI image")  
        canvas.im1 = img_mri
          

        #Insert PET image to the canvas
        img_pet = ImageTk.PhotoImage(file ="C:/Users/........./FuseVis/Input2/Input2.png") # load the image
        canvas.create_image(0, 650, image=img_pet, anchor=NW)
        #canvas.create_text(1325,270,fill="black",font=("Purisa", 12),text="PET image")
        canvas.im2 = img_pet
        
        
        #load the fused image and insert into the canvas
        im_out1 = ImageTk.PhotoImage(file ="C:/Users/........./FuseVis/Fused images/DeepFuse/Fused.png") #load the image
        canvas.create_image(0, 320, image=im_out1, anchor=NW)
        #canvas.create_text(125,270,fill="black",font=("Purisa", 12),text="Fused image")
        canvas.image1 = im_out1
        
        
        #load the funfusean model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model1 =torch.load('C:/Users/........./FuseVis/Checkpoint/FunFuseAn/checkpoint_lambda_0.99_gamma_ssim_0.47_gamma_l2_0.5.pth')
        model1.eval()
        
        #load the model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model5 =torch.load('C:/Users/........./FuseVis/Checkpoint/DeepFuse/checkpoint_lambda_0.99_gamma_ssim_0.497_gamma_l2_0.5.pth')
        model5.eval()
        
        #predict the fused image
        fused_tensor1 = model1(input1_tensor.to(device), input2_tensor.to(device))
        fused_tensor1_numpy = fused_tensor1.data.cpu().numpy()
        
        #predict the fused image
        model5.setInput(input1_tensor.to(device), input2_tensor.to(device))          # cnn output
        fused_tensor5 = model5.forward()
        fused_tensor5_numpy = fused_tensor5.data.cpu().numpy()
        #load the Fused wrt MRI guidance images and normalize it
        hf = h5py.File('C:/Users/........./FuseVis/DeepFuse/Guidance_Input1.h5', 'r')
        guide_fused_mri =  np.array(hf.get('MRI_dataset'))
        hf.close()
        hf = h5py.File('C:/Users/........./FuseVis/DeepFuse/Guidance_Input2.h5', 'r')
        guide_fused_pet =  np.array(hf.get('PET_dataset'))
        hf.close()
        guide_mri_min = np.min(guide_fused_mri)
        guide_mri_max = np.max(guide_fused_mri)
        guide_pet_min = np.min(guide_fused_pet)
        guide_pet_max = np.max(guide_fused_pet)  
        if (guide_mri_max > guide_pet_max):
            max_ = guide_mri_max
        else:
            max_ = guide_pet_max
            
        min_ = 0
        #color map details for the zoom images
        cmap = plt.get_cmap('viridis')
        #cNorm = mpl.colors.Normalize(vmin=min_, vmax=max_) #re-wrapping normalization
        cNorm = mpl.colors.PowerNorm(gamma=gamma, vmin=min_, vmax=max_)
        scalarMap = mpl.cm.ScalarMappable(norm=cNorm, cmap=cmap)
        
        fig = plt.figure(figsize=(0.2, 23))
        ax1 = fig.add_axes([0.05, 0.80, 0.9, 0.15])
        cb1 = mpl.colorbar.ColorbarBase(ax1, cmap=cmap,norm=cNorm,orientation='vertical')
        plt.plot()
        plt.savefig('C:/Users/........./FuseVis/Guidance images/DeepFuse/colorbar.png', bbox_inches = 'tight',pad_inches = 0)
        plt.close()
        im_out50 = ImageTk.PhotoImage(file ='C:/Users/........./FuseVis/Guidance images/DeepFuse/colorbar.png')
        canvas.create_image(1152,650,image=im_out50,anchor=NW)
        canvas.image50 = im_out50 
        

        im_out2 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(255*scalarMap.to_rgba(guide_fused_mri))).resize((256,256)))
        canvas.create_image(600,0,image=im_out2,anchor=NW)
        #canvas.create_text(125,920,fill="black",font=("Purisa", 12),text="G1: Gradient map (Fused wrt MRI)")  
        canvas.image2 = im_out2
        
        guide_fused_mri_orig = guide_fused_mri
          
        
        im_out3 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(255*scalarMap.to_rgba(guide_fused_pet))).resize((256,256)))
        canvas.create_image(600,650,image=im_out3,anchor=NW)
        canvas.image3 = im_out3
        
        guide_fused_pet_orig = guide_fused_pet
        
        
        #display the Fused RGB image
        fused_RGB = np.zeros((256,256,3),dtype=float)
        guide_fused_mri_norm = (guide_fused_mri - np.min(guide_fused_mri)) / (np.max(guide_fused_mri) - np.min(guide_fused_mri))
        guide_fused_pet_norm = (guide_fused_pet - np.min(guide_fused_pet)) / (np.max(guide_fused_pet) - np.min(guide_fused_pet))
        fused_numpy_norm = (fused_numpy - np.min(fused_numpy)) / (np.max(fused_numpy) - np.min(fused_numpy))
        fused_RGB[:,:,0]  = guide_fused_mri_norm 
        fused_RGB[:,:,1]  = guide_fused_pet_norm 
        fused_RGB[:,:,2]  = fused_numpy_norm

        im_out4 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(fused_RGB*255)).resize((256,256)))
        canvas.create_image(600,320,image=im_out4,anchor=NW)
        #canvas.create_text(875,920,fill="black",font=("Purisa", 12),text="Guidance R:G1, G: G2, B: Fused")  
        canvas.image4 = im_out4
        
        
        #display the Scatter plot image
        guide_mri_flat   = guide_fused_mri.flatten()
        guide_pet_flat   = guide_fused_pet.flatten() 
        plt.xlim(min_, max_)
        plt.ylim(min_, max_)
        plt.scatter(guide_mri_flat,guide_pet_flat, marker='o', s=2, facecolors='0')
        plt.xlabel('G1: Guidance MRI (Fused pixel wrt MRI pixel)')
        plt.ylabel('G2: Guidance PET (Fused pixel wrt PET pixel)')
        plt.title('k) Scatter plot between guidance images', fontsize=12)
        plt.savefig('C:/Users/........./FuseVis/Guidance images/DeepFuse/Scatter_Plot.png', pad_inches = 0)
        plt.close()
        im_out5 = ImageTk.PhotoImage(file ='C:/Users/........./FuseVis/Guidance images/DeepFuse/Scatter_Plot.png')
        canvas.create_image(1250,300,image=im_out5,anchor=NW)
        canvas.image5 = im_out5

        canvas.create_image(1752,650,image=im_out50,anchor=NW)
        canvas.image51 = im_out50
        
        canvas.create_image(1752,0,image=im_out50,anchor=NW)
        canvas.image52 = im_out50
        
        id_ = canvas.create_image(1152,0,image=im_out50,anchor=NW)
        canvas.image53 = im_out50
        
        #insert button to the middleframe and link it to "Start Mouseover"
        start_mouseover(fused_tensor5, model, min_, max_, guide_fused_mri_orig, guide_fused_pet_orig, fused_tensor1, fused_RGB, id_,gamma,fused_numpy_norm,gamma1)
        #button_start_mouseover.grid(row=1, column=0, pady=0)
        
    if model == "DeepPedestrian":
        gamma = 1
        gamma1 = 1
        canvas.delete(ALL)
        
        #Insert texts
        canvas.create_text(125,270,fill="black" ,font=("Purisa", 12),text="a) MRI image") 
        canvas.create_text(425,270,fill="black" ,font=("Purisa", 12),text="b) Zoom MRI")
        canvas.create_text(725,270,fill="black" ,font=("Purisa", 12),text="c) G1: Guidance (Fused pixel wrt MRI pixel)")  
        canvas.create_text(1025,270,fill="black",font=("Purisa", 12),text="d) Zoom Guidance G1") 
        canvas.create_text(1325,270,fill="black",font=("Purisa", 11),text="e) J1: Jacobian (Fused pixel wrt MRI image)")
        canvas.create_text(1625,270,fill="black",font=("Purisa", 11),text="f) Zoom Jacobian J1")
        canvas.create_text(125,590,fill="black" ,font=("Purisa", 12),text="g) Fused image")
        canvas.create_text(425,590,fill="black" ,font=("Purisa", 12),text="h) Zoom Fused")        
        canvas.create_text(725,590,fill="black" ,font=("Purisa", 12),text="i) Guidance R:G1, G: G2, B: Fused") 
        canvas.create_text(1025,590,fill="black",font=("Purisa", 12),text="j) Zoom Guidance RGB")
        canvas.create_text(125, 920,fill="black",font=("Purisa", 12),text="l) PET image")
        canvas.create_text(425, 920,fill="black",font=("Purisa", 12),text="m) Zoom PET")        
        canvas.create_text(725,920,fill="black" ,font=("Purisa", 12),text="n) G2: Guidance (Fused pixel wrt PET pixel)")
        canvas.create_text(1025,920,fill="black",font=("Purisa", 12),text="o) Zoom Guidance G2") 
        canvas.create_text(1325,920,fill="black",font=("Purisa", 11),text="p) J2: Jacobian (Fused pixel wrt PET image)")  
        canvas.create_text(1625,920,fill="black",font=("Purisa", 11),text="q) Zoom Jacobian J2")
        
        fused_numpy = imageio.imread("C:/Users/........./FuseVis/Fused images/DeepPedestrian/Fused.png")
        
        #Insert MRI image to the canvas
        img_mri = ImageTk.PhotoImage(file ="C:/Users/........./FuseVis/Input1/Input1.png") # load the image
        canvas.create_image(0, 0, image=img_mri, anchor=NW)
        #canvas.create_text(725,270,fill="black",font=("Purisa", 12),text="MRI image")  
        canvas.im1 = img_mri
          

        #Insert PET image to the canvas
        img_pet = ImageTk.PhotoImage(file ="C:/Users/........./FuseVis/Input2/Input2.png") # load the image
        canvas.create_image(0, 650, image=img_pet, anchor=NW)
        #canvas.create_text(1325,270,fill="black",font=("Purisa", 12),text="PET image")
        canvas.im2 = img_pet
        
        
        #load the fused image and insert into the canvas
        im_out1 = ImageTk.PhotoImage(file ="C:/Users/........./FuseVis/Fused images/DeepPedestrian/Fused.png") #load the image
        canvas.create_image(0, 320, image=im_out1, anchor=NW)
        #canvas.create_text(125,270,fill="black",font=("Purisa", 12),text="Fused image")
        canvas.image1 = im_out1
        
        
        #load the funfusean model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model1 =torch.load('C:/Users/........./FuseVis/Checkpoint/FunFuseAn/checkpoint_lambda_0.99_gamma_ssim_0.47_gamma_l2_0.5.pth')
        model1.eval()
        
        #load the model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model6 =torch.load('C:/Users/........./FuseVis/Checkpoint/DeepPedestrian/checkpoint_lambda_0.99_gamma_ssim_0.52_gamma_l2_0.5.pth')
        model6.eval()
        
        #predict the fused image
        fused_tensor1 = model1(input1_tensor.to(device), input2_tensor.to(device))
        fused_tensor1_numpy = fused_tensor1.data.cpu().numpy()
        
        #predict the fused image
        fused_tensor6 = model6(input1_tensor.to(device), input2_tensor.to(device))
        fused_tensor6_numpy = fused_tensor6.data.cpu().numpy()
        #load the Fused wrt MRI guidance images and normalize it
        hf = h5py.File('C:/Users/........./FuseVis/DeepPedestrian/Guidance_Input1.h5', 'r')
        guide_fused_mri =  np.array(hf.get('MRI_dataset'))
        hf.close()
        hf = h5py.File('C:/Users/........./FuseVis/DeepPedestrian/Guidance_Input2.h5', 'r')
        guide_fused_pet =  np.array(hf.get('PET_dataset'))
        hf.close()
        guide_mri_min = np.min(guide_fused_mri)
        guide_mri_max = np.max(guide_fused_mri)
        guide_pet_min = np.min(guide_fused_pet)
        guide_pet_max = np.max(guide_fused_pet)  
        if (guide_mri_max > guide_pet_max):
            max_ = guide_mri_max
        else:
            max_ = guide_pet_max
            
        min_ = 0
        #color map details for the zoom images
        cmap = plt.get_cmap('viridis')
        #cNorm = mpl.colors.Normalize(vmin=min_, vmax=max_) #re-wrapping normalization
        cNorm = mpl.colors.PowerNorm(gamma=gamma, vmin=min_, vmax=max_)
        scalarMap = mpl.cm.ScalarMappable(norm=cNorm, cmap=cmap)
        
        fig = plt.figure(figsize=(0.2, 23))
        ax1 = fig.add_axes([0.05, 0.80, 0.9, 0.15])
        cb1 = mpl.colorbar.ColorbarBase(ax1, cmap=cmap,norm=cNorm,orientation='vertical')
        plt.plot()
        plt.savefig('C:/Users/........./FuseVis/Guidance images/DeepPedestrian/colorbar.png', bbox_inches = 'tight',pad_inches = 0)
        plt.close()
        im_out50 = ImageTk.PhotoImage(file ='C:/Users/........./FuseVis/Guidance images/DeepPedestrian/colorbar.png')
        canvas.create_image(1152,650,image=im_out50,anchor=NW)
        canvas.image50 = im_out50 
        

        im_out2 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(255*scalarMap.to_rgba(guide_fused_mri))).resize((256,256)))
        canvas.create_image(600,0,image=im_out2,anchor=NW)
        #canvas.create_text(125,920,fill="black",font=("Purisa", 12),text="G1: Gradient map (Fused wrt MRI)")  
        canvas.image2 = im_out2
        
        guide_fused_mri_orig = guide_fused_mri
          

        im_out3 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(255*scalarMap.to_rgba(guide_fused_pet))).resize((256,256)))
        canvas.create_image(600,650,image=im_out3,anchor=NW)
        #canvas.create_text(1325,920,fill="black",font=("Purisa", 12),text="G2: Gradient map (Fused wrt PET)")
        canvas.image3 = im_out3
        
        guide_fused_pet_orig = guide_fused_pet
        
        
        #display the Fused RGB image
        fused_RGB = np.zeros((256,256,3),dtype=float)
        guide_fused_mri_norm = (guide_fused_mri - np.min(guide_fused_mri)) / (np.max(guide_fused_mri) - np.min(guide_fused_mri))
        guide_fused_pet_norm = (guide_fused_pet - np.min(guide_fused_pet)) / (np.max(guide_fused_pet) - np.min(guide_fused_pet))
        fused_numpy_norm = (fused_numpy - np.min(fused_numpy)) / (np.max(fused_numpy) - np.min(fused_numpy))
        fused_RGB[:,:,0]  = guide_fused_mri_norm 
        fused_RGB[:,:,1]  = guide_fused_pet_norm 
        fused_RGB[:,:,2]  = fused_numpy_norm

        im_out4 = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(fused_RGB*255)).resize((256,256)))
        canvas.create_image(600,320,image=im_out4,anchor=NW)
        #canvas.create_text(875,920,fill="black",font=("Purisa", 12),text="Guidance R:G1, G: G2, B: Fused")  
        canvas.image4 = im_out4
        
        
        #display the Scatter plot image
        guide_mri_flat   = guide_fused_mri.flatten()
        guide_pet_flat   = guide_fused_pet.flatten() 
        plt.xlim(min_, max_)
        plt.ylim(min_, max_)
        plt.scatter(guide_mri_flat,guide_pet_flat, marker='o', s=2, facecolors='0')
        plt.xlabel('G1: Guidance MRI (Fused pixel wrt MRI pixel)')
        plt.ylabel('G2: Guidance PET (Fused pixel wrt PET pixel))')
        plt.title('k) Scatter plot between guidance images', fontsize=12)
        plt.savefig('C:/Users/........./FuseVis/Guidance images/DeepPedestrian/Scatter_Plot.png', pad_inches = 0)
        plt.close()
        im_out5 = ImageTk.PhotoImage(file ='C:/Users/........./FuseVis/Guidance images/DeepPedestrian/Scatter_Plot.png')
        canvas.create_image(1250,300,image=im_out5,anchor=NW)
        canvas.image5 = im_out5

        canvas.create_image(1752,650,image=im_out50,anchor=NW)
        canvas.image51 = im_out50
        
        canvas.create_image(1752,0,image=im_out50,anchor=NW)
        canvas.image52 = im_out50
        
        id_ = canvas.create_image(1152,0,image=im_out50,anchor=NW)
        canvas.image513 = im_out50
        
        #insert button to the middleframe and link it to "Start Mouseover"
        start_mouseover(fused_tensor6, model, min_, max_, guide_fused_mri_orig, guide_fused_pet_orig, fused_tensor1, fused_RGB, id_,gamma,fused_numpy_norm,gamma1)        
        

click = StringVar(root)
click.set("Select Model")
models = ["Weighted Averaging","FunFuseAn", "DeepFuse", "MaskNet", "DeepPedestrian"]
drop = OptionMenu(buttonframe, click, *models, command = load_model)
drop.grid(row=1, column=2, pady=0)



root.mainloop()  #keep the GUI open