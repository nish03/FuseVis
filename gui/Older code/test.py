from tkinter import *
import tkinter as tk
import tkinter.font as tkFont
from PIL import ImageTk, Image

#define the window
root = Tk()  
root.title('Visualisation of fusion networks')
root.configure(background='white')


#Label the images
#fontStyle = tkFont.Font(family="Lucida Grande", size=15)
#w1 = tk.Label(root, bg='white', font=fontStyle, text="Fused Image")
#w1.grid(row=0, column=1)
#w1.pack()

#define the frame
canvasframe = Frame(root)  # define Input and output frame
buttonframe = Frame(root)  # define button frame
canvasframe.pack()  # pack the Input and Output frame
buttonframe.pack()  # pack the button frame


#define the canvas
canvas = Canvas(canvasframe, width=1250, height=620, bg = 'white')
canvas.grid(row=0, column=0)

#Insert fused image to the canvas
img_fused = ImageTk.PhotoImage(file ="C:/Users/cgvadmin/Desktop/Suraka/Fused.png") # load the image
canvas.create_image(0, 0, image=img_fused, anchor=NW)

#Insert MRI image to the canvas
img_mri = ImageTk.PhotoImage(file ="C:/Users/cgvadmin/Desktop/Suraka/MRI.png") # load the image
canvas.create_image(500, 0, image=img_mri, anchor=NW)

#Insert PET image to the canvas
img_pet = ImageTk.PhotoImage(file ="C:/Users/cgvadmin/Desktop/Suraka/PET.png") # load the image
canvas.create_image(1000, 0, image=img_pet, anchor=NW)


def start_mouseover():  # function called when user clicks the button 
    # link the function to the left-mouse-click event
    canvas.bind("<B1-Motion>", Coordinates)

def Coordinates(event): # function called when left-mouse-button is clicked with a mouseover
    x_coord = event.x  # save x and y coordinates selected by the user   
    y_coord = event.y
    print('mouse position is at' + '(' + str(y_coord) + ',' + str(x_coord) + ')', end='\r')
    #display the output MRI Jacobian image
    img_MR_out = ImageTk.PhotoImage(file ='C:/Users/cgvadmin/Desktop/Suraka/Fused_MRI/im_' + str(y_coord) + '_' + str(x_coord) + '.png') # load the image
    canvas.create_image(10,260,image=img_MR_out,anchor=NW)
    canvas.image1 = img_MR_out
    #display the output PET Jacobian image
    img_PET_out = ImageTk.PhotoImage(file ='C:/Users/cgvadmin/Desktop/Suraka/Fused_PET/im_' + str(y_coord) + '_' + str(x_coord) + '.png') # load the image
    canvas.create_image(645,260,image=img_PET_out,anchor=NW)
    canvas.image2 = img_PET_out
    # Display a small dot showing position of point.
    radius = 2
    i = canvas.create_oval(x_coord-radius, y_coord-radius, x_coord+radius, y_coord+radius, fill = 'red')
    canvas.after(50,canvas.delete,i)

# insert button to the middleframe and link it to "Start Mouseover"
button_start_mouseover = Button(buttonframe, text="Start Mouseover",command=start_mouseover)
button_start_mouseover.grid(row=1, column=0, pady=0)


root.mainloop()  #keep the GUI open