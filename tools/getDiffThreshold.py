import cv2
import matplotlib.pyplot as plt
import sys

sys.path.append('..')
import config
from camera import Camera

regionX = config.MonitorRegion['x']
regionY = config.MonitorRegion['y']
regionW = config.MonitorRegion['w']
regionH = config.MonitorRegion['h']

camera = Camera('../checkpoint_50.pth', config.TinyFaceModelDevice)
cap = cv2.VideoCapture(0)
x = 0
ax = []
ay = []
plt.ion()

while (cap.isOpened()):
    ret, img = cap.read()
    if ret == True:
        img2 = img[regionY:regionY + regionH, regionX:regionX + regionW]
        diff = camera.detectDiff(img2.copy())
        p1 = (regionX, regionY)
        p2 = (regionX + regionW, regionY + regionH)
        img = cv2.rectangle(img, p1, p2, 1)
        cv2.imshow('Image', img)
        if x > 10:
            ax.append(x)
            ay.append(diff)
            plt.clf()
            plt.plot(ax, ay)
            plt.pause(0.01)
        x += 1
        k = cv2.waitKey(10)
        if k != -1:
            break

plt.ioff()
cap.release()
cv2.waitKey(0)
cv2.destroyAllWindows()
exit(0)
