import torch
from torch import optim
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision import models
import torch.utils.data as data
import time, glob, math
import cv2
from sklearn.metrics import average_precision_score
import numpy as np
import os
from model import ResNet50
import sys, shutil, json

sys.path.append('..')
import logger, config

log = logger.Logger.getLogger('train')

# https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

# [come in, come out, stay, person1~9]
OUT_DIM = 3 + len(config.PersonNames)
BATCH_SIZE = 1
NUM_EPOCHS = 20
PERCENTILE = 0
LEARNING_RATE = 0.0001


######################### dataloader #########################

class ImageDataSet(data.Dataset):

    def __init__(self, imgPath, imgList):
        super().__init__()
        self.imgPath = imgPath
        self.imgList = imgList
        self.dataLen = len(self.imgList)
        self.transform = Compose([ToTensor(), Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        f = open('./labels.json', 'r')
        self.labels = json.load(f)

    def __len__(self):
        return self.dataLen

    def __getitem__(self, index):
        name = self.imgList[index].replace(self.imgPath, '').replace('.png', '')[1:]
        l = self.labels[name]

        img = cv2.imread(self.imgList[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(img)
        # label = torch.zeros((1, OUT_DIM))
        label = torch.from_numpy(np.array(l).astype(np.float32))

        return img, label.squeeze()


######################### train #########################


resnet = models.resnet50()
resnet.load_state_dict(torch.load('./resnet50-19c8e357.pth'))

net = ResNet50(resnet, OUT_DIM)

try:
    net.load_state_dict(torch.load('models/latestModel.pth'))
    log.info('load pre trained model')
except:
    log.info('pre trained model isn\'t exist')

try:
    os.mkdir('models')
except:
    pass

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = net.to(device)

trainDataPath = './imagesWithBox'
allTrainDataList = glob.glob(trainDataPath + '/*.png')

trainDataSet = ImageDataSet(trainDataPath, allTrainDataList)
trainDataLoader = data.DataLoader(dataset=trainDataSet, batch_size=BATCH_SIZE, shuffle=False)

optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
lossFunc = torch.nn.MultiLabelSoftMarginLoss()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)


def compute_mAP(labels, outputs):
    y_true = labels.cpu().numpy()
    y_pred = outputs.cpu().numpy()
    AP = []
    for i in range(y_pred.shape[0]):
        AP.append(average_precision_score(y_true[i], y_pred[i]))
    return np.mean(AP)

for epoch in range(NUM_EPOCHS):
    runningLoss = 0.0
    epochStartTime = time.time()
    net.train()
    mAP = []
    log.info("set learning rate: %.6f" % optimizer.param_groups[0]['lr'])

    for i, (imagesBatch, labelsBatch) in enumerate(trainDataLoader):
        startTime = time.time()
        imagesBatch = imagesBatch.to(device)
        labelsBatch = labelsBatch.to(device)

        optimizer.zero_grad()

        # with torch.set_grad_enabled(True):
        predBatch = net(imagesBatch)
        loss = lossFunc(predBatch, labelsBatch)

        loss.backward()
        optimizer.step()

        runningLoss += loss.item() * imagesBatch.size(0)
        # predictions = predBatch.detach().numpy()[0]
        # idx_list = np.where(predictions > np.percentile(predictions, PERCENTILE))
        # print(idx_list)
        m = compute_mAP(labelsBatch.data, predBatch.data)
        mAP.append(m)

        elapsedTime = time.time() - startTime
        log.info(
            "Epoch:[{:3d}/{:3d}]: {:3d}/{:3d} | loss:{:.8f} | mAP:{:.2f} | Time: {:.4f}s".format(epoch + 1, NUM_EPOCHS,
                                                                                                 i + 1,
                                                                                                 math.ceil(len(
                                                                                                     trainDataLoader.dataset) / BATCH_SIZE),
                                                                                                 runningLoss / (i + 1),
                                                                                                 np.mean(mAP[
                                                                                                         0 - BATCH_SIZE:]),
                                                                                                 elapsedTime))

    modelName = 'models/{}_{}_model.pth'.format(time.strftime("%Y-%m-%d_%H_%M_%S", time.localtime()), epoch + 1)
    torch.save(net.state_dict(), modelName)
    shutil.copy(modelName, 'models/latestModel.pth')
    scheduler.step(loss)
    epochLoss = runningLoss / len(trainDataLoader.dataset)
    epochElapsedTime = time.time() - epochStartTime
    log.info(
        "Epoch:[{:3d}/{:3d}] | loss:{:.8f} | mAP:{:.2f} | Time: {:.4f}s".format(epoch + 1, NUM_EPOCHS, epochLoss,
                                                                                np.mean(mAP), epochElapsedTime))
