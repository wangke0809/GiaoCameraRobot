<p align="center">
  <img src="https://raw.githubusercontent.com/wangke0809/giaocamerarobot/master/docs/giao.png"/>
</p>

GiaoCameraRobot is the mascot of SIPL Lab.

<p>
  <a href=""><img src="https://raw.githubusercontent.com/wangke0809/giaocamerarobot/master/docs/fake-like-sipl-passing.png" alt="passing"></a>
  <a href=""><img  src="https://raw.githubusercontent.com/wangke0809/giaocamerarobot/master/docs/fake-like-siplcodecov.png"  alt="codecov"></a>
  <a href=""><img src="https://raw.githubusercontent.com/wangke0809/giaocamerarobot/master/docs/license.png" alt="passing"></a>
</p>

English | [中文]

[中文]: https://feelncut.com/2019/12/09/giaocamerarobot.html

## Overview

I named her Xiao [Giao], and she could monitor every movement in the laboratory through the camera.

[Giao]: https://baike.baidu.com/item/giao/458428

Xiao Giao uses OpenCV to obtain camera data, uses TinyFace to detect faces, and recognizes people and actions by fine-tuning ResNet50. Send DingTalk notifications when someone is detected entering or leaving.

![](https://raw.githubusercontent.com/wangke0809/giaocamerarobot/master/docs/1.png)


## Getting Started
### Installation

- Clone this repo:
```bash
git clone https://github.com/wangke0809/GiaoCameraRobot.git
cd GiaoCameraRobot
```

- Install [PyTorch](http://pytorch.org) 0.4+ and torchvision.
  - For pip users, please type the command `pip install -r requirements.txt`.
  
### Configuration
 
 ```bash
 cp config.py.example config.py
 ```
 
 Get the region to be monitored:
 
 ```bash
 cd tools
 python getMonitorRegion.py
 ```
 
Get image detection threshold：

 ```bash
 cd tools
 python getDiffThreshold.py
 ```
 
  Fill the region and threshold into config.py
  
Download [TinyFace pre-trained model](https://drive.google.com/open?id=1vdKzrfQ4cXeI157NEJoeI1ECZ66GFEKE) into current directory, and download [ResNet50 pre-trained model](https://download.pytorch.org/models/resnet50-19c8e357.pth) into `tools` directory. 

  
### Run
  
```bash
python GiaoRobot.py
```

## Train the Model

Of course, you can't run normally by following the above steps, because the model has not been trained yet! You need to close the person and action detection module first， and collect enough training samples through face detection.

### Step 1: Prepare training dataset

```bash
cd tools
python imagesDrawBoxes.py
```

### Step 2: Annotate training dataset

 ```bash
cd tools
python getTrainDataSet.py
```

![](https://raw.githubusercontent.com/wangke0809/giaocamerarobot/master/docs/2.png)

### Step 3: Train the model
  
 ```bash
cd tools
python trainModel.py
```   

## Related Projects

- [face-detection-pytorch](https://github.com/cs-giung/face-detection-pytorch)
