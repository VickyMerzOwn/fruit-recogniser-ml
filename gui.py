from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog

root = Tk()
root.title("A GUI Fruit Recogiser")

def open():
	root.filename = filedialog.askopenfilename(initialdir="~/Desktop/submission", title = "select a File", filetypes = (("jpg file", "*.jpg"),("all files", "*.*")))
	image = ImageTk.PhotoImage(Image.open(root.filename))
	image_label = Label(image = image).pack()

open_file_button = Button(root, text = "Choose File", command = open)


root.mainloop()
