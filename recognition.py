import torch
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision import models
import time, glob
import cv2
import numpy as np
from tools.model import ResNet50
import json
from tools.getTrainDataSet import person
from logger import Logger
import config

log = Logger.getLogger('Recognition')


class Recognition(object):

    def __init__(self, device='cpu'):
        self.device = device
        tstamp = time.time()
        log.info('[Recognition] loading with %s', self.device)
        resnet = models.resnet50()
        resnet.load_state_dict(torch.load('tools/resnet50-19c8e357.pth'))
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net = ResNet50(resnet, 12)
        net = net.to(self.device)
        net.load_state_dict(torch.load('tools/models/latestModel.pth', map_location=torch.device(self.device)))
        self.net = net
        self.transform = Compose([ToTensor(), Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        log.info('[Recognition] finished loading (%.4f sec)' % (time.time() - tstamp))

    def detect(self, img, bgr2rgb=True):
        tstamp = time.time()
        if bgr2rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(img).unsqueeze_(0)
        img = img.to(self.device)
        predLabel = self.net(img)
        predictions = predLabel.data.cpu().numpy()[0]
        predType = predictions[0:3]
        predTypeId = np.where(predType > np.percentile(predType, 90))[0][0]
        predNames = predictions[3:]
        predNamesId = np.where(predNames > np.percentile(predNames, 90))[0][0]
        log.info('predictions: ' + str(predictions))
        ret1 = (predTypeId, True if predType[predTypeId] > 0 else False)
        ret2 = (predNamesId, True if predNames[predNamesId] > 0 else False)
        log.info('recognition using %.6f sec' % (time.time() - tstamp))
        return ret1, ret2


if __name__ == '__main__':
    rec = Recognition()
    # img = cv2.imread('tools/111.png')
    # rec.detect(img)
