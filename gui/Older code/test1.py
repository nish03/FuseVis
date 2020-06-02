import tkinter as tk


class MouseControl:        
    def __init__(self, canvas):            
        self.canvas = canvas
        self.canvas.bind('<Button-1>', self.clicked)  
        self.canvas.bind('<Double-1>', self.double_click)  
        self.canvas.bind('<ButtonRelease-1>', self.button_released)  
        self.canvas.bind('<B1-Motion>', self.moved)  

    def clicked(self, event):      
        print('single mouse click event at ({}, {})'.format(event.x, event.y))

    def double_click(self, event):
        print('double mouse click event')

    def button_released(self, event):        
        print('button released')

    def moved(self, event):        
        print('mouse position is at ({:03}. {:03})'.format(event.x, event.y), end='\r')    

def main():
    root = tk.Tk()
    window = tk.Canvas(root, width=400, height=400, bg='grey')
    mouse = MouseControl(window)
    window.place(x=0, y=0)
    window.mainloop()


if __name__ == "__main__":
    main()