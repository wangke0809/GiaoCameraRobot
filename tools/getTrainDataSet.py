import cv2
import tkinter as tk
import json

window = tk.Tk()

window.title('Make DataSet')
window.geometry('900x300')

person = {0: 'wk', 1: 'dln', 2: 'kml', 3: 'zsy', 4: 'hh', 5: 'wly', 6: 'lch', 7: 'wzh', 8: 'ys'}

def onClick(data):
    print(data)


b = tk.Button(window, text='come in', font=('Arial', 12), width=10, height=2, command=lambda: onClick('come in'))
b.pack()
b = tk.Button(window, text='come out', font=('Arial', 12), width=10, height=2, command=lambda: onClick('come out'))
b.pack()
b = tk.Button(window, text='stay', font=('Arial', 12), width=10, height=2, command=lambda: onClick('stay'))
b.pack()
b = tk.Button(window, text='save', font=('Arial', 12), width=10, height=2, command=lambda: onClick('save'))
b.pack()
for p in person:
    b = tk.Button(window, text=person[p], font=('Arial', 12), width=10, height=2, command=lambda: onClick(p))
    b.pack(side='left')

window.mainloop()
