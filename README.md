# Vehicle Recognition using Single Shot Detector (SSD)

**[Overview](#overview)** | **[Abstract](#abstract)** | **[Model](#model)** | **[Experiments](#experiments)** | **[Results](#experimental-results)** | **[PowerPoint](#powerpoint)** | **[Conclusion](#conclusion)** | **[SSD Setup](#ssd-model-setup)** 

---

# Overview

<p align="justify">
Vehicle recognition using Single Shot Detector (SSD) in Autonomous Vehicles (AV's) was created as a final project for CSC 7991: Introduction to Deep Learning. The purpose of this project was to conduct experiments using a SSD to detect vehicles in nighttime, snowy, and drone videos. The goal is to demonstrate the SSD's ability to detect vehicles at varying distances in harsh conditions. The SSD for this project was implemented from its creators Wei Liu et al., which paper, model, and model setup is listed below.   
</p>

<p align="center"> 
<img src="https://raw.githubusercontent.com/kyle-w-brown/vehicle-recongition-ssd/master/img/nighttime.png" width="33%"> <img src="https://raw.githubusercontent.com/kyle-w-brown/vehicle-recongition-ssd/master/img/drone.PNG" width="31%"> <img src="https://raw.githubusercontent.com/kyle-w-brown/vehicle-recongition-ssd/master/img/snow.png" width="32%">
</p>


## [Project Website](https://kyle-w-brown.github.io/vehicle-recongition-ssd/)

---

# Abstract

<p align="justify">
Over the past decade, deep neural networks have evolved and now dominate many vision tasks such as image recognition, object detection and semantic segmentation. This is now proving to be very useful since it’s giving computer systems the ability to recognize visual data (video or image) and make decisions accordingly. Many different deep neural architectures are now specialized in accomplishing a main objective. Convolutional neural network (CNN) has been developed to be one of the best networks in image recognition, once the model is built and trained on specific class data, feeding it test data the model will recognize the classes it has learned from the test samples. 
</p>

---

# Model

##  SSD: Single Shot MultiBox Detector

[![Build Status](https://travis-ci.org/weiliu89/caffe.svg?branch=ssd)](https://travis-ci.org/weiliu89/caffe)
[![License](https://img.shields.io/badge/license-BSD-blue.svg)](LICENSE)

By [Wei Liu](http://www.cs.unc.edu/~wliu/), [Dragomir Anguelov](https://www.linkedin.com/in/dragomiranguelov), [Dumitru Erhan](http://research.google.com/pubs/DumitruErhan.html), [Christian Szegedy](http://research.google.com/pubs/ChristianSzegedy.html), [Scott Reed](http://www-personal.umich.edu/~reedscot/), [Cheng-Yang Fu](http://www.cs.unc.edu/~cyfu/), [Alexander C. Berg](http://acberg.com).

## [Caffe Website](http://caffe.berkeleyvision.org/)

### Introduction

<p align="justify"> 
SSD is an unified framework for object detection with a single network. Authors Liu et al, have made the code available to train/evaluate a network for object detection task. For more details, please refer to our arXiv paper and slide link below. Our team decided on SSD because it is more fine grained compared to YOLO which is more suited for large object detection in a video or image. SSD is a feedforward convolutional network that uses a fixed number of bounding boxes and scores to attempt to detect VoC object classes within those bounding boxes. The VOC dataset consists of 11,530 training images. Finally a non-maximum suppression step to produce the final detections and remove overlapping bounding boxes. SSD is simple relative to methods that require object proposals because it completely eliminates proposal generation and subsequent pixel or feature resampling stages and encapsulates all computation in a single network. 
</p>

* [arXiv paper](http://arxiv.org/abs/1512.02325) 
* [slide](http://www.cs.unc.edu/~wliu/papers/ssd_eccv2016_slide.pdf)

<br>

<img src="http://www.cs.unc.edu/~wliu/papers/ssd.png" alt="SSD Framework" width="600px">


| System | VOC2007 test *mAP* | **FPS** (Titan X) | Number of Boxes | Input resolution
|:-------|:-----:|:-------:|:-------:|:-------:|
| [Faster R-CNN (VGG16)](https://github.com/ShaoqingRen/faster_rcnn) | 73.2 | 7 | ~6000 | ~1000 x 600 |
| [YOLO (customized)](http://pjreddie.com/darknet/yolo/) | 63.4 | 45 | 98 | 448 x 448 |
| SSD300* (VGG16) | 77.2 | 46 | 8732 | 300 x 300 |
| SSD512* (VGG16) | **79.8** | 19 | 24564 | 512 x 512 |

<img src="http://www.cs.unc.edu/~wliu/papers/ssd_results.png" alt="SSD results on multiple datasets" width="800px">

_Note: SSD300* and SSD512* are the latest models. Current code should reproduce these results._

<br>

**[SSD: Single Shot MultiBox Detector, Liu et al. arVix](https://github.com/kyle-w-brown/vehicle-recongition-ssd/blob/master/SSD%20-%20Wei%20Liu%20et%20al%2C%20arVix%202016/1512.02325.pdf)**

---

# Experiments

<p align="justify">
Safety comes from the autonomous vehicle being able to detect and respond to many traffic situations. The more input the computer can handle and analyze at a fast processing time will give the autonomous vehicle increased decision making. To visualize how a CNN performs on an input with poor weather conditions, our team gathered dashcam videos from Youtube of vehicles operating in harsh conditions. The input Youtube videos concentrate on snowy, and nighttime videos to apply Single Shot Detection (SSD) to classify the vehicles and bound them in green boxes. 
</p>

The main advantages we believe this method will have are:
1.	Real Time Vehicle Detection: Ability to use live feed camera input into cycle GAN and CNN and using onboard CPU and GPU computing to visualize vehicles on the road in real time.
2.	In-network Architecture: All components in our method are within one network and trained in an end-to-end fashion.
3.	Data Augmentation: 720p video input is resized to 300x300x3 image segments which at a much lower resolution and requires less storage and computing power.  


# Nighttime Experiment

<div align="center">
  <img src="https://raw.githubusercontent.com/kyle-w-brown/vehicle-recongition-ssd/master/img/nighttime.png" width="45%"> <img src="https://raw.githubusercontent.com/kyle-w-brown/vehicle-recongition-ssd/master/img/nighttime-four.png" width="45%"><br><br>
</div>


[![SSD: Experiments](img/yt.png)](https://www.youtube.com/watch?v=t-bmOthOJxY)

<br>

# Snowy Condition Experiments

## Snowy Conditions Vehicles Parked Experiment

<div align="center">
  <img src="https://raw.githubusercontent.com/kyle-w-brown/vehicle-recongition-ssd/master/img/snow-parked.PNG" width="45%"> <img src="https://raw.githubusercontent.com/kyle-w-brown/vehicle-recongition-ssd/master/img/snow-parked-two.PNG" width="45%"><br><br>
</div>

[![SSD: Experiments](img/yt.png)](https://www.youtube.com/watch?v=0biampSWKNA)

<br>

## Snowy Conditions Vehicles Moving Experiment

<div align="center">
  <img src="https://raw.githubusercontent.com/kyle-w-brown/vehicle-recongition-ssd/master/img/snow-two.png" width="45%"> <img src="https://raw.githubusercontent.com/kyle-w-brown/vehicle-recongition-ssd/master/img/snow-three.png" width="45%"><br><br>
</div>

[![SSD: Experiments](img/yt.png)](https://www.youtube.com/watch?v=QVAgJ0P7epY)

<br>

# Drone Experiments 

## Aerial Drone Experiment

<div align="center">
  <img src="https://raw.githubusercontent.com/kyle-w-brown/vehicle-recongition-ssd/master/img/drone.PNG" width="45%"> <img src="https://raw.githubusercontent.com/kyle-w-brown/vehicle-recongition-ssd/master/img/drone-two.PNG" width="45%"><br><br>
</div>

[![SSD: Experiments](img/yt.png)](https://www.youtube.com/watch?v=fAo_G2XjxKY)

<br>

## Parking Lot Drone Experiment

<div align="center">
  <img src="https://raw.githubusercontent.com/kyle-w-brown/vehicle-recongition-ssd/master/img/drone-plaza.PNG" width="45%"> <img src="https://raw.githubusercontent.com/kyle-w-brown/vehicle-recongition-ssd/master/img/drone-plaza-two.PNG" width="45%"><br><br>
</div>

[![SSD: Experiments](img/yt.png)](https://www.youtube.com/watch?v=vWDNv6RghDU)

---

<br>

# Group 2 PowerPoint
 
[![SSD: Group 2 PowerPoint](img/ppt.PNG)](https://github.com/kyle-w-brown/vehicle-recongition-ssd/blob/master/CSC%207991%20Final%20Presentation/SSD-Vehicle-Recognition_Group-2.pptx?raw=true)

<div align="center">
  <img src="https://raw.githubusercontent.com/kyle-w-brown/vehicle-recongition-ssd/master/img/ppt.png" width="85%"><br><br>
</div>

<div align="center">
  <img src="https://raw.githubusercontent.com/kyle-w-brown/vehicle-recongition-ssd/master/img/ssd-two.png" width="85%"><br><br>
</div>

<div align="center">
  <img src="https://raw.githubusercontent.com/kyle-w-brown/vehicle-recongition-ssd/master/img/ssd-three.png" width="85%"><br><br>
</div>

<div align="center">
  <img src="https://raw.githubusercontent.com/kyle-w-brown/vehicle-recongition-ssd/master/img/ssd-four.png" width="85%"><br><br>
</div>

<div align="center">
  <img src="https://raw.githubusercontent.com/kyle-w-brown/vehicle-recongition-ssd/master/img/ssd-five.png" width="85%"><br><br>
</div>

<div align="center">
  <img src="https://raw.githubusercontent.com/kyle-w-brown/vehicle-recongition-ssd/master/img/ssd-six.png" width="85%"><br><br>
</div>

<div align="center">
  <img src="https://raw.githubusercontent.com/kyle-w-brown/vehicle-recongition-ssd/master/img/ssd-apps.png" width="85%"><br><br>
</div>

<br>

---


# Experimental Results

| Conditions | Accuracy |
|:-------|:-----:|
| Drone Aerial | 18% |
| Drone Parking Lot | 82% |
| Nighttime | 43% |
| Snow Moving | 44% | 
| Snow Parked | 17% | 


## Summary Experimental Results

<p align="justify">
Our preliminary results using SSD for object detection has exceeded project expectations. Our observed results include: Snowy – 44% and 16%, Night time – 43%, and Drone – 18% and 82%. GANs and SSDs are still a developing technology which comes with many challenges including: Unproven technology, no economies of scale, and CPU/GPU limitations. 
</p>

---

# Conclusion

<p align="justify">
We proposed the implementation of a single-shot detection (SSD) for vehicle recognition. SSD's provide an advantage for real-time vehicle detection, in-network architecture, and data augmentation that uses on-board computing within one network to resize video images. The SSD, is a fast single-shot object detector for multiple categories. SSD's are a feed-forward convolutional network that uses a fixed number of bounding boxes and scores to detect VoC object classes using bounding boxes. A key feature of our experiments is the use of bounding box outputs attached to multiple feature maps trained from VoC dataset against night time, snowy, and drone videos. The experiment conducted used VOC dataset with SSDs 11,530 training images. The inputs for the experiments were dashcam videos from YouTube tested with CNN using SSD for vehicle box bounding area. With similar technological impacts, GAN's and SSD's are promising for the development of Advanced Driving Assistance Systems (ADAS), military transportation and bomb disposal, and emergency applications. A future direction is to explore its use as part of a system using recurrent neural networks to detect and track objects in video simultaneously.
</p>

---

<br>

# Citing SSD

```sh
@inproceedings{liu2016ssd,
  title = {SSD: Single Shot MultiBox Detector},
  author = {Liu, Wei and Anguelov, Dragomir and Erhan, Dumitru and Szegedy, Christian and Reed, Scott and Fu, Cheng-Yang and Berg, Alexander C.},
  booktitle = {ECCV},
  year = {2016}
}
```


---

# SSD Model Setup

### Contents
1. [Installation](#installation)
2. [Preparation](#preparation)
3. [Train/Eval](#traineval)
4. [Models](#models)

### Installation
1. Get the code. We will call the directory that you cloned Caffe into `$CAFFE_ROOT`
  ```Shell
  git clone https://github.com/weiliu89/caffe.git
  cd caffe
  git checkout ssd
  ```

2. Build the code. Please follow [Caffe instruction](http://caffe.berkeleyvision.org/installation.html) to install all necessary packages and build it.
  ```Shell
  # Modify Makefile.config according to your Caffe installation.
  cp Makefile.config.example Makefile.config
  make -j8
  # Make sure to include $CAFFE_ROOT/python to your PYTHONPATH.
  make py
  make test -j8
  # (Optional)
  make runtest -j8
  ```

### Preparation
1. Download [fully convolutional reduced (atrous) VGGNet](https://gist.github.com/weiliu89/2ed6e13bfd5b57cf81d6). By default, we assume the model is stored in `$CAFFE_ROOT/models/VGGNet/`

2. Download VOC2007 and VOC2012 dataset. By default, we assume the data is stored in `$HOME/data/`
  ```Shell
  # Download the data.
  cd $HOME/data
  wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
  wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
  wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
  # Extract the data.
  tar -xvf VOCtrainval_11-May-2012.tar
  tar -xvf VOCtrainval_06-Nov-2007.tar
  tar -xvf VOCtest_06-Nov-2007.tar
  ```

3. Create the LMDB file.
  ```Shell
  cd $CAFFE_ROOT
  # Create the trainval.txt, test.txt, and test_name_size.txt in data/VOC0712/
  ./data/VOC0712/create_list.sh
  # You can modify the parameters in create_data.sh if needed.
  # It will create lmdb files for trainval and test with encoded original image:
  #   - $HOME/data/VOCdevkit/VOC0712/lmdb/VOC0712_trainval_lmdb
  #   - $HOME/data/VOCdevkit/VOC0712/lmdb/VOC0712_test_lmdb
  # and make soft links at examples/VOC0712/
  ./data/VOC0712/create_data.sh
  ```

### Train/Eval
1. Train your model and evaluate the model on the fly.
  ```Shell
  # It will create model definition files and save snapshot models in:
  #   - $CAFFE_ROOT/models/VGGNet/VOC0712/SSD_300x300/
  # and job file, log file, and the python script in:
  #   - $CAFFE_ROOT/jobs/VGGNet/VOC0712/SSD_300x300/
  # and save temporary evaluation results in:
  #   - $HOME/data/VOCdevkit/results/VOC2007/SSD_300x300/
  # It should reach 77.* mAP at 120k iterations.
  python examples/ssd/ssd_pascal.py
  ```
  If you don't have time to train your model, you can download a pre-trained model at [here](https://drive.google.com/open?id=0BzKzrI_SkD1_WVVTSmQxU0dVRzA).

2. Evaluate the most recent snapshot.
  ```Shell
  # If you would like to test a model you trained, you can do:
  python examples/ssd/score_ssd_pascal.py
  ```

3. Test your model using a webcam. Note: press <kbd>esc</kbd> to stop.
  ```Shell
  # If you would like to attach a webcam to a model you trained, you can do:
  python examples/ssd/ssd_pascal_webcam.py
  ```
  [Here](https://drive.google.com/file/d/0BzKzrI_SkD1_R09NcjM1eElLcWc/view) is a demo video of running a SSD500 model trained on [MSCOCO](http://mscoco.org) dataset.

4. Check out [`examples/ssd_detect.ipynb`](https://github.com/weiliu89/caffe/blob/ssd/examples/ssd_detect.ipynb) or [`examples/ssd/ssd_detect.cpp`](https://github.com/weiliu89/caffe/blob/ssd/examples/ssd/ssd_detect.cpp) on how to detect objects using a SSD model. Check out [`examples/ssd/plot_detections.py`](https://github.com/weiliu89/caffe/blob/ssd/examples/ssd/plot_detections.py) on how to plot detection results output by ssd_detect.cpp.

5. To train on other dataset, please refer to data/OTHERDATASET for more details. We currently add support for COCO and ILSVRC2016. We recommend using [`examples/ssd.ipynb`](https://github.com/weiliu89/caffe/blob/ssd/examples/ssd_detect.ipynb) to check whether the new dataset is prepared correctly.

### Models
We have provided the latest models that are trained from different datasets. To help reproduce the results in [Table 6](https://arxiv.org/pdf/1512.02325v4.pdf), most models contain a pretrained `.caffemodel` file, many `.prototxt` files, and python scripts.

1. PASCAL VOC models:
   * 07+12: [SSD300*](https://drive.google.com/open?id=0BzKzrI_SkD1_WVVTSmQxU0dVRzA), [SSD512*](https://drive.google.com/open?id=0BzKzrI_SkD1_ZDIxVHBEcUNBb2s)
   * 07++12: [SSD300*](https://drive.google.com/open?id=0BzKzrI_SkD1_WnR2T1BGVWlCZHM), [SSD512*](https://drive.google.com/open?id=0BzKzrI_SkD1_MjFjNTlnempHNWs)
   * COCO<sup>[1]</sup>: [SSD300*](https://drive.google.com/open?id=0BzKzrI_SkD1_NDlVeFJDc2tIU1k), [SSD512*](https://drive.google.com/open?id=0BzKzrI_SkD1_TW4wTC14aDdCTDQ)
   * 07+12+COCO: [SSD300*](https://drive.google.com/open?id=0BzKzrI_SkD1_UFpoU01yLS1SaG8), [SSD512*](https://drive.google.com/open?id=0BzKzrI_SkD1_X3ZXQUUtM0xNeEk)
   * 07++12+COCO: [SSD300*](https://drive.google.com/open?id=0BzKzrI_SkD1_TkFPTEQ1Z091SUE), [SSD512*](https://drive.google.com/open?id=0BzKzrI_SkD1_NVVNdWdYNEh1WTA)

2. COCO models:
   * trainval35k: [SSD300*](https://drive.google.com/open?id=0BzKzrI_SkD1_dUY1Ml9GRTFpUWc), [SSD512*](https://drive.google.com/open?id=0BzKzrI_SkD1_dlJpZHJzOXd3MTg)

3. ILSVRC models:
   * trainval1: [SSD300*](https://drive.google.com/open?id=0BzKzrI_SkD1_a2NKQ2d1d043VXM), [SSD500](https://drive.google.com/open?id=0BzKzrI_SkD1_X2ZCLVgwLTgzaTQ)

<sup>[1]</sup>We use [`examples/convert_model.ipynb`](https://github.com/weiliu89/caffe/blob/ssd/examples/convert_model.ipynb) to extract a VOC model from a pretrained COCO model.


---

# SSD300 Python Model Setup

Single Shot MultiBox Detector implemented with TensorFlow

## Dependencies ##
python3.6.1
* numpy
* skimage
* TensorFlow
* matplotlib
* OpenCV

## Usage ##
1. Import required modules
```
import tensorflow as tf
import numpy as np

from util.util import *
from model.SSD300 import *
```

2. Load test-image  
```
img = load_image('./test.jpg')
img = img.reshape((300, 300, 3))
```

3. Start Session  
```
with tf.Session() as sess:
        ssd = SSD300(sess)
        sess.run(tf.global_variables_initializer())
        for ep in range(EPOCH):
            ...
```

4. Training or Evaluating
you must just call ssd.eval() !
```
...

_, _, batch_loc, batch_conf, batch_loss = ssd.eval(minibatch, actual_data, is_training=True)

...
```


## Test Training ##
you have to extract data-set from zip files.
decompress all zip files in datasets/ and move to voc2007/ dir.
```
$ ls voc2007/ | wc -l    #  => 4954
$ ./setup.sh
$ python train.py
```
