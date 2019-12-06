import cv2

cap = cv2.VideoCapture(0)

mouseX = -1
mouseY = -1
mouseX1 = -1
mouseY1 = -1
mouseX2 = -1
mouseY2 = -1
mouseMode = -1
printConfig = False


def callBack(event, x, y, flags, param):
    global mouseX1, mouseY1, mouseX, mouseY, mouseX2, mouseY2, mouseMode, printConfig
    mouseX = x
    mouseY = y
    if event == cv2.EVENT_LBUTTONDOWN:
        mouseMode = 0
        mouseX1 = x
        mouseY1 = y
    elif event == cv2.EVENT_LBUTTONUP:
        mouseMode = 1
        mouseX2 = x
        mouseY2 = y
    elif event == cv2.EVENT_RBUTTONDOWN:
        mouseMode = -1
        printConfig = False


cv2.namedWindow('Image')
cv2.setMouseCallback('Image', callBack)

while (cap.isOpened()):
    ret, img = cap.read()
    if ret == True:
        img = cv2.putText(img, "(%d, %d)" % (mouseX, mouseY2), (mouseX, mouseY), cv2.FONT_HERSHEY_SIMPLEX,
                          0.7, (255, 255, 255), 1, cv2.LINE_AA)
        if mouseMode == 0:
            p1 = (mouseX1, mouseY1)
            p2 = (mouseX, mouseY)
            img = cv2.rectangle(img, p1, p2, 1)
        if mouseMode == 1:
            p1 = (mouseX1, mouseY1)
            p2 = (mouseX2, mouseY2)
            img = cv2.rectangle(img, p1, p2, 1)
            if not printConfig:
                print("MonitorRegion = {'x': %d, 'y': %d, 'w': %d, 'h': %d}" % (
                    mouseX1, mouseY1, mouseX2 - mouseX1, mouseY2 - mouseY1))
                printConfig = True
        cv2.imshow('Image', img)
        k = cv2.waitKey(100)
        if k == ord('c') or k == ord('C'):
            mouseMode = -1
            printConfig = False
        elif k != -1:
            break

cap.release()
cv2.waitKey(0)
cv2.destroyAllWindows()
exit(0)
