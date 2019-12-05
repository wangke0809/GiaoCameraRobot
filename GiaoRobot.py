import cv2
import time
import config
from logger import Logger
from push import Push
from storage import Storage
from camera import Camera

log = Logger.getLogger("Giao")


def main():
    # open camera
    cap = cv2.VideoCapture(0)

    last_send_time = 0

    camera = Camera(config.TinyFaceModelPath, config.TinyFaceModelDevice)
    push = Push(config.dingTalkWebhookAccessToken)
    storage = Storage(config.SinaSCSAccessKey, config.SinaSCSSecretKey, config.SinaSCSBucketName,
                      config.SinaSCSBucketUrl)

    while (cap.isOpened()):
        try:
            ret, img = cap.read()
            if ret == True:
                # img = img[config.MonitorRegion]
                imgRegion = img[185:305, 270:490]
                diff = camera.detectDiff(imgRegion)
                if diff < 150:
                    log.info("different < 150")
                    time.sleep(0.2)
                    continue
                bboxes = camera.detectFaces(imgRegion)
                log.info("bboxes len %d", len(bboxes))
                if len(bboxes) > 0 and (time.time() - last_send_time) > 30:
                    last_send_time = time.time()
                    saveName = str(time.strftime('images/%Y%m%d_%H_%M_%S.png'))
                    cv2.imwrite(saveName, img)
                    saveName = str(time.strftime('images/%Y%m%d_%H_%M_%S.jpg'))
                    cv2.imwrite(saveName, img)
                    url = storage.saveImage(saveName)
                    log.info("send giao! %s" % url)
                    push.sendImage(config.PushTitle, url)
        except Exception as e:
            log.error("error: %s", e)

        time.sleep(0.5)

    cap.release()


if __name__ == '__main__':
    main()
