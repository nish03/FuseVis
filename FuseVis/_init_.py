def _init_():
	#check if GPUs available
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    #define the image dimensions 
    image_length = 256
    image_width  = 256
    gray_channels = 1
    return image_length, image_width, gray_channels, device
    
 