import tkinter as tk
from tkinter import filedialog
#from main import *

def selectFile():
    openFiles.append(filedialog.askopenfilename(initialdir = "/User/Desktop", title = "Select File", filetypes = (("obj files", "*.obj"), ("jpeg files", "*.jpg"), ("png files", "*.png"))))

def populateFiles():
    filesList.delete(0, tk.END)
    filesList.insert(tk.END, *openFiles)

def populatePlayer():
    player.delete("all")
    for x in video:
        player.create_rectangle(x, 0, x + 5, 50, fill = "#908010", width = 0)
        if x >= root.winfo_pointerx() - 303 and x <= root.winfo_pointerx() - 297:
            player.create_polygon(x, 60, x - 10, 70, x - 10, 90, x + 70, 90, x + 70, 70, x + 10, 70, x, 60, fill = "#101030", width = 3, outline = "#151535")
            player.create_text(x + 10, 78, text = str(video[x].__name__), fill = "#303040")
    player.create_rectangle(root.winfo_pointerx() - 295, 0, root.winfo_pointerx() - 300, 50, fill = "#905010", width = 0)

root = tk.Tk()
root.title("VFX")
root.geometry("1500x800")

topbar = tk.Frame(root, height = 30, bg = "#101030")
topbar.pack(fill = tk.X, side = tk.TOP)

toolbar = tk.Frame(root, width = 300, bg = "#303040")
toolbar.pack(fill = tk.Y, side = tk.LEFT)

preferences = tk.Frame(root, width = 500, bg = "#303040")
preferences.pack(fill = tk.Y,  side = tk.RIGHT)

player = tk.Canvas(root, height = 100, bg = "#404050", highlightthickness = -2)
player.pack(fill = tk.X, side = tk.BOTTOM)

files = tk.Frame(preferences, width = 500, height = 500, bg = "#404050")
files.pack(side = tk.TOP, padx = 10, pady = 10)
scrollbar = tk.Scrollbar(files)
scrollbar.pack(side = tk.RIGHT, fill = tk.Y)
filesList = tk.Listbox(files, xscrollcommand = scrollbar.set, bg = "#404050")
filesList.pack(side = tk.LEFT, fill = tk.BOTH)
scrollbar.config(command = filesList.yview)

class Class:
    pass

openFiles = []
video = {100:Class}

menubar = tk.Menu(root)
filemenu = tk.Menu(menubar, tearoff = 0)
filemenu.add_command(label = "Import", command = selectFile)
menubar.add_cascade(label = "File", menu = filemenu)
root.config(menu = menubar)

while True:
    populateFiles()
    populatePlayer()
    root.update()
