from sinastorage.bucket import SCSBucket
import sinastorage
import sys
sys.path.append("../")
import config
import time

uploadedAmount = 0.0
def putFileCallback(size, progress):
    print(size, progress)

sinastorage.setDefaultAppInfo(config.SinaSCSAccessKey, config.SinaSCSSecretKey)

s = SCSBucket(config.SinaSCSBucketName)
# s.putFile('1.jpg', 'selfie.jpg', putFileCallback)
ret = s.putFile(str(time.strftime('%Y%m%d/%H:%M:%S')) + '.jpg', 'selfie.jpg', acl='public-read')
print('complete!')
print(ret.info())

