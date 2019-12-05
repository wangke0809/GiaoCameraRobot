import cv2
import time

cap = cv2.VideoCapture(0)  

cv2.namedWindow("Image")
while(cap.isOpened()): 
    ret,img = cap.read() 
    if ret == True: 
        cv2.imshow('Image',img)
        k = cv2.waitKey(100)
        if k == ord('a') or k == ord('A'):

            cv2.imwrite(str(time.strftime('%Y%m%d_%H_%M_%S.png')),img)
            break
cv2.waitKey(0)
# if cap.isOpened():
#     time.sleep(15)
#     ret,img = cap.read()
#     if ret == True:
#         cv2.imwrite('test2.png',img)

cap.release() 
