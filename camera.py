from tinyface import TinyFace
from recognition import Recognition
import cv2, time
import numpy as np
from logger import Logger
from utils import draw_bboxes

log = Logger.getLogger('camera')


class Camera(object):

    def __init__(self, modelPath, device='cpu'):
        self.TF = TinyFace(modelPath, device=device)
        self.REC = Recognition(device=device)
        self.lastImg = None

    def detectDiff(self, img):
        tStart = time.time()
        if self.lastImg is None:
            ret, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
            binary_img = binary_img / 255
            self.lastImg = binary_img
            return 99999999
        diff = 0
        ret, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        if self.lastImg is not None:
            binary_img = binary_img / 255
            diff = binary_img - self.lastImg
            diff = np.abs(np.sum(diff))
            self.lastImg = binary_img
        else:
            self.lastImg = binary_img / 255

        log.info("diff: %6d, using %.6f sec" % (diff, time.time() - tStart))

        return diff

    def detectFaces(self, img):
        imgRegion = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        bboxes = self.TF.detect_faces(imgRegion, conf_th=0.9, scales=[1])
        bboxesLength = len(bboxes)
        predType, predName = (-1, True), (-1, True)
        if bboxesLength> 0:
            imgRegion = draw_bboxes(img, bboxes, thickness=1)
            predType, predName = self.REC.detect(imgRegion, True)
        return bboxesLength, predType, predName
