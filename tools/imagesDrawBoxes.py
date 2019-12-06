import cv2
import sys

sys.path.append('..')
import config
from utils import draw_bboxes
from tinyface import TinyFace
import glob

tf = TinyFace('../checkpoint_50.pth', device=config.TinyFaceModelDevice)

regionX = config.MonitorRegion['x']
regionY = config.MonitorRegion['y']
regionW = config.MonitorRegion['w']
regionH = config.MonitorRegion['h']

imagesPath = '../images'
imagesWithBox = './imagesWithBox'
pngList = glob.glob(imagesPath + '/*.png')
savedList = glob.glob(imagesWithBox + '/*.png')
savedListSet = set()
needHandleList = []

for savedListPath in savedList:
    path = savedListPath.replace(imagesWithBox, '')
    savedListSet.add(path)
for pngListPath in pngList:
    path = pngListPath.replace(imagesPath, '')
    if not savedListSet.__contains__(path):
        needHandleList.append(pngListPath)

print("png num: ", len(pngList), 'handled num: ', len(savedListSet), 'needHandle num: ', len(needHandleList))

for pngPath in needHandleList:
    print("handle", pngPath)
    img = cv2.imread(pngPath)
    img = img[regionY:regionY + regionH, regionX:regionX + regionW]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    bboxes = tf.detect_faces(img, conf_th=0.9, scales=[1])
    print(bboxes)
    img = draw_bboxes(img, bboxes, thickness=1)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    savePath = pngPath.replace(imagesPath, imagesWithBox)
    print(savePath)
    cv2.imwrite(savePath, img)

print("png num: ", len(pngList), 'handled num: ', len(savedListSet), 'needHandle num: ', len(needHandleList))
