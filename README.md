# FuseVis: Interpreting neural networks for image fusion using per-pixel saliency visualization
The project presents a visualization tool to interpret neural networks for image fusion. The tool, named as **FuseVis**, can be used by end-user to compute per-pixel saliency maps that examine the influence of the input image pixels on each pixel of the fused image. The tool can also be adapted to interpret any deep neural network that involves image processing. The work is based on the MDPI journal [paper](https://www.mdpi.com/2073-431X/9/4/98). A video on how to use the tool can be downloaded from clicking [here](
https://tu-dresden.de/ing/informatik/smt/cgv/ressourcen/dateien/mitarbeiter/nishant-kumar/FuseVis_teaser.mp4).

![GitHub Logo](/docs/Tool.png)


The tool performs the following key tasks:

*  Fast computation of per-pixel jacobian based saliency maps for the fused image with respect to input image pairs.
*  Visualize neural networks by considering the backpropagation heuristics using an interactive user interface that helps these networks to be more transparent in a real-time setup.

**Note**: Please cite the paper if you are using this code in your research.

## Prerequisites
* Python 2.7
* Pytorch 1.0 and above
* Tkinter
* numpy
* matplotlib
* skimage
* PIL
* imageio
* h5py

## Data
The training of each of the evaluated fusion network was done with 500 MRI-PET image pairs available at Alzheimer’s Disease Neuroimaging Initiative (ADNI) as mentioned in the paper. Although, the data is available for public use, it is still required to apply for getting the access to the repository by filling out a questionaire. In case you are interested to obtain the data, please apply for access at [this link](http://adni.loni.usc.edu/data-samples/access-data/). For conducting the inference/testing of networks, in addition to the ADNI test data, we also used image pairs from [Harvard Whole Brain Atlas](http://www.med.harvard.edu/AANLIB/) which is also publicly available and donot required any written permission.

## Usage
Step 1: The first step to use FuseVis tool is to train your own neural network with your own dataset where you feed image pairs to the fusion network and obtain a fused image as a prediction from the network. In case you are interested to use one of the evaluated fusion networks, for example: DeepFuse, then you need to run the pytorch based notebook implementation DeepFuse.ipynb provided in this repository to perform the training.

Step 2: Once the network is trained, you can now run FuseVis.ipynb to obtain the user interface. Some key points to note:
a) Make sure you run the jupyter notebook offline since tkinter fails to work in online platforms such as Google Colab.
b) If you just trained one fusion network, you need to make some modifications to FuseVis.ipynb. They are as follows:
        * You need to remove the code snippet related to the architecture of the model which was not trained and hence donot have a checkpoint path. Weighted Averaging method is an exception since it is a training free method.
        * You need to remove the complete code snippet under the 'if' statement of the definition of the 'load_model()' function. For example: if you didn't trained DeepPedestrian network, you need to remove the code under the following 'if' statement:
        
        ![Code snippet](/docs/Sample_Code_snippet.png)
        
        * Make sure you atleast train the FunFuseAn network and have its checkpoint since the implementation of FuseVis has been performed in a way that the fused tensor from the FunFuseAn network is mandatory to define the local window for obtaining zoomed images. 
        
        * You can remove the model which was not trained from the following code snippet:
         ![Code snippet1](/docs/Model_definition.png)

Note: You can also define your own fusion based neural network based on your requirements and use it with FuseVis tool to interpret the influence of the input images on the fused image.  

## How to Cite
Kumar, N.; Gumhold, S. FuseVis: Interpreting Neural Networks for Image Fusion Using Per-Pixel Saliency Visualization. Computers 2020, 9, 98.

## Evaluated Neural Networks
![Logo1](/docs/Networks.png)
