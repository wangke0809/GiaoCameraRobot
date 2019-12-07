import torch
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision import models
import time, glob
import cv2
import numpy as np
from model import ResNet50
import json
from getTrainDataSet import person

OUT_DIM = 12

resnet = models.resnet50()
resnet.load_state_dict(torch.load('./resnet50-19c8e357.pth'))

net = ResNet50(resnet, OUT_DIM)

net.load_state_dict(torch.load('models/latestModel.pth'))

trainDataPath = './imagesWithBox'
allTrainDataList = glob.glob(trainDataPath + '/*.png')

f = open('./labels.json', 'r')
labels = json.load(f)

transform = Compose([ToTensor(), Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])


def getStr(data):
    data -= 3
    if data == -3:
        return 'come in, '
    elif data == -2:
        return 'come out, '
    elif data == -1:
        return 'stay, '
    else:
        return person[data]


for path in allTrainDataList:
    name = path.replace(trainDataPath, '').replace('.png', '')[1:]
    # print(name)
    imgO = cv2.imread(path)
    img = cv2.cvtColor(imgO, cv2.COLOR_BGR2RGB)
    img = transform(img).unsqueeze_(0)
    t1 = time.time()
    predLabel = net(img)
    gtLable = labels[name]
    print('use %.4f' % (time.time() - t1))
    predictions = predLabel.data.cpu().numpy()[0]
    idx_list = np.where(predictions > np.percentile(predictions, 90))[0]
    # print(gtLable)
    gtStr = 'GT '
    for i in range(len(gtLable)):
        if gtLable[i] == 1:
            gtStr += getStr(i)
            gtStr += ' '
    print(gtStr)
    # print(idx_list)
    predStr = 'PD '
    for i in idx_list:
        predStr += getStr(i)
        predStr += ' '
    print(predStr)
    imgO = cv2.putText(imgO, gtStr, (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                      0.7, (0, 0, 0), 1, cv2.LINE_AA)
    imgO = cv2.putText(imgO, predStr, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                       0.7, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imshow("image", imgO)
    cv2.waitKey(0)

