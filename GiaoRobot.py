import cv2
import time
import config
from logger import Logger
from push import Push
from storage import Storage
from camera import Camera
from utils import draw_bboxes

log = Logger.getLogger("Giao")


def main():
    # open camera
    cap = cv2.VideoCapture(0)

    last_send_time = 0

    camera = Camera(config.TinyFaceModelPath, config.TinyFaceModelDevice)
    push = Push(config.dingTalkWebhookAccessToken)
    storage = Storage(config.SinaSCSAccessKey, config.SinaSCSSecretKey, config.SinaSCSBucketName,
                      config.SinaSCSBucketUrl)

    regionX = config.MonitorRegion['x']
    regionY = config.MonitorRegion['y']
    regionW = config.MonitorRegion['w']
    regionH = config.MonitorRegion['h']

    while (cap.isOpened()):
        try:
            ret, img = cap.read()
            if ret == True:
                imgRegion = img[regionY:regionY + regionH, regionX:regionX + regionW]
                diff = camera.detectDiff(imgRegion)
                if diff < config.MonitorDiffThreshold:
                    time.sleep(0.2)
                    continue
                faceNum, predType, predName = camera.detectFaces(imgRegion)
                log.info("faceNum: %d", faceNum)
                if faceNum > 0:
                    if predType[1]:
                        typeStr = '确定'
                    else:
                        typeStr = '可能'
                    if predType[0] == 0:
                        typeStr += '进入'
                    elif predType[0] == 1:
                        typeStr += '离开'
                    else:
                        typeStr += '站起来了'
                    if predName[1]:
                        nameStr = '确定是'
                    else:
                        nameStr = '可能是'
                    nameStr += config.PersonNames[predName[0]]
                    content = nameStr + '，' + typeStr
                    log.info('push content %s', content)

                if faceNum > 0 and (time.time() - last_send_time) > 30:
                    last_send_time = time.time()
                    saveName = str(time.strftime('images/%Y%m%d_%H_%M_%S.png'))
                    cv2.imwrite(saveName, img)
                    saveName = str(time.strftime('images/%Y%m%d_%H_%M_%S.jpg'))
                    cv2.imwrite(saveName, img)
                    url = storage.saveImage(saveName)
                    log.info("send giao! %s" % url)
                    push.sendImage(config.PushTitle, content, url)
        except Exception as e:
            log.error("error: %s", e)

        # time.sleep(0.5)

    cap.release()


if __name__ == '__main__':
    main()
