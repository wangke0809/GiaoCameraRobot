import cv2
import tkinter as tk
import json, glob
import os, sys
sys.path.append('..')
import config


person = config.PersonNames

index = 0

l = [0] * 12

def main():
    window = tk.Tk()

    window.title('Make DataSet')
    window.geometry('900x300')

    imagesWithBox = './imagesWithBox'
    imagesList = os.listdir(imagesWithBox)

    labelsJson = './labels.json'

    if not os.path.exists(labelsJson):
        labels = {}
    else:
        f = open(labelsJson, 'r')
        labels = json.load(f)

    needHandleList = []

    for path in imagesList:
        name = path.replace('.png', '')
        if name not in labels.keys():
            needHandleList.append(path)

    if len(needHandleList) == 0:
        exit(0)

    img = cv2.imread(imagesWithBox + '/' + needHandleList[index])
    cv2.imshow('Image', img)

    content = tk.StringVar()
    content.set("There are %d images need to handle" % len(needHandleList))
    w = tk.Label(window, textvariable=content)
    w.pack()


    def onClick(data):
        global index, l
        if data == 'save':
            if index >= len(needHandleList):
                content.set(" " * 100)
                content.set("Finished!")
                return
            # if l[0] == 0 and l[1] == 0 and l[2] == 0:
            #     return
            mark = False
            for i in l[3:]:
                if i == 1:
                    mark = True
            # if not mark:
            #     return
            labels[needHandleList[index].replace('.png', '')] = l
            with open(labelsJson, 'w') as file:
                json.dump(labels, file)

            index += 1
            if index >= len(needHandleList):
                return
            l = [0] * 12
            img = cv2.imread(imagesWithBox + '/' + needHandleList[index])
            cv2.imshow('Image', img)
        elif data == 'come in':
            l[0] = 1
            l[1] = 0
            l[2] = 0
        elif data == 'come out':
            l[0] = 0
            l[1] = 1
            l[2] = 0
        elif data == 'stay':
            l[0] = 0
            l[1] = 0
            l[2] = 1
        else:
            l[3 + int(data)] = 1 if l[3 + int(data)] == 0 else 0
        if index < len(needHandleList):
            content.set(
                needHandleList[index].replace('.png', '') + " " + str(l) + '%d left' % (len(needHandleList) - index))


    b = tk.Button(window, text='In', font=('Arial', 12), width=10, height=2, command=lambda: onClick('come in'))
    b.pack()
    b = tk.Button(window, text='Out', font=('Arial', 12), width=10, height=2, command=lambda: onClick('come out'))
    b.pack()
    b = tk.Button(window, text='Stay', font=('Arial', 12), width=10, height=2, command=lambda: onClick('stay'))
    b.pack()
    b = tk.Button(window, text='Save', font=('Arial', 12), width=10, height=2, command=lambda: onClick('save'))
    b.pack()
    for p in person:
        b = tk.Button(window, text=person[p], font=('Arial', 12), width=10, height=2, command=lambda p=p: onClick(p))
        b.pack(side='left')

    window.mainloop()

if __name__ == '__main__':
    main()
