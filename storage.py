from sinastorage.bucket import SCSBucket
import sinastorage
import time


class Storage(object):

    def __init__(self, accessKey, secretKey, bucketName, bucketUrl):
        sinastorage.setDefaultAppInfo(accessKey, secretKey)
        self.s = SCSBucket(bucketName)
        self.bucketUrl = bucketUrl

    def saveImage(self, path):
        name = str(time.strftime('%Y%m%d/%H:%M:%S.jpg'))
        self.s.putFile(name, path, acl='public-read')
        return self.bucketUrl % name

# def putFileCallback(size, progress):
#     print(size, progress)
