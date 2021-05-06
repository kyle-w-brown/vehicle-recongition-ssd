**[Overview](#overview)** | **[Abstract](#abstract)** | **[Model](#model)** | **[Experiments](#experiments)** | **[Results](#experimental-results)** | **[PowerPoint](#powerpoint)** | **[Conclusion](#conclusion)** |

---

# Overview

<p align="justify">
Vehicle recognition using Single Shot Detector (SSD) in Autonomous Vehicles (AV's) was created as a final project for CSC 7991: Introduction to Deep Learning. The purpose of this project was to conduct experiments using a SSD to detect vehicles in nighttime, snowy, and drone videos. The goal is to demonstrate the SSD's ability to detect vehicles at varying distances in harsh conditions. The SSD for this project was implemented from its creators Wei Liu et al., which paper, model, and model setup is listed below.   
</p>

<p align="center"> 
<img src="https://raw.githubusercontent.com/kyle-w-brown/vehicle-recongition-ssd/master/img/nighttime.png" width="33%"> <img src="https://raw.githubusercontent.com/kyle-w-brown/vehicle-recongition-ssd/master/img/drone.PNG" width="31%"> <img src="https://raw.githubusercontent.com/kyle-w-brown/vehicle-recongition-ssd/master/img/snow.png" width="32%">
</p>

---

<br>

# Abstract

<p align="justify">
Over the past decade, deep neural networks have evolved and now dominate many vision tasks such as image recognition, object detection and semantic segmentation. This is now proving to be very useful since it’s giving computer systems the ability to recognize visual data (video or image) and make decisions accordingly. Many different deep neural architectures are now specialized in accomplishing a main objective. Convolutional neural network (CNN) has been developed to be one of the best networks in image recognition, once the model is built and trained on specific class data, feeding it test data the model will recognize the classes it has learned from the test samples. 
</p>




---

<br>

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

<p align="center">
<iframe width="550" height="550" src="https://nbviewer.jupyter.org/github/kyle-w-brown/vehicle-recongition-ssd/blob/master/SSD%20-%20Wei%20Liu%20et%20al%2C%20arVix%202016/1512.02325.pdf" title="arXiv:1512.02325v5" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</p>

---

<br>

# Experiments

<p align="justify">
Safety comes from the autonomous vehicle being able to detect and respond to many traffic situations. The more input the computer can handle and analyze at a fast processing time will give the autonomous vehicle increased decision making. To visualize how a CNN performs on an input with poor weather conditions, our team gathered dashcam videos from Youtube of vehicles operating in harsh conditions. The input Youtube videos concentrate on snowy, and nighttime videos to apply Single Shot Detection (SSD) to classify the vehicles and bound them in green boxes. 
</p>

The main advantages we believe this method will have are:
1.	Real Time Vehicle Detection: Ability to use live feed camera input into cycle GAN and CNN and using onboard CPU and GPU computing to visualize vehicles on the road in real time.
2.	In-network Architecture: All components in our method are within one network and trained in an end-to-end fashion.
3.	Data Augmentation: 720p video input is resized to 300x300x3 image segments which at a much lower resolution and requires less storage and computing power.  


<br>

## Nighttime Experiment

<div align="center">
  <img src="https://raw.githubusercontent.com/kyle-w-brown/vehicle-recongition-ssd/master/img/nighttime.png" width="45%"> <img src="https://raw.githubusercontent.com/kyle-w-brown/vehicle-recongition-ssd/master/img/nighttime-four.png" width="45%"><br><br>
</div>

<p align="center">
<iframe width="590" height="315" src="https://www.youtube.com/embed/t-bmOthOJxY" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</p>

<br>

## Snow Conditions Experiments

### Snow Conditions Vehicles Parked Experiment

<div align="center">
  <img src="https://raw.githubusercontent.com/kyle-w-brown/vehicle-recongition-ssd/master/img/snow-parked.PNG" width="45%"> <img src="https://raw.githubusercontent.com/kyle-w-brown/vehicle-recongition-ssd/master/img/snow-parked-two.PNG" width="45%"><br><br>
</div>

<p align="center">
<iframe width="560" height="315" src="https://www.youtube.com/embed/0biampSWKNA?start=6" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</p>

<br>

### Snow Conditions Vehicles Moving Experiment

<div align="center">
  <img src="https://raw.githubusercontent.com/kyle-w-brown/vehicle-recongition-ssd/master/img/snow-two.png" width="45%"> <img src="https://raw.githubusercontent.com/kyle-w-brown/vehicle-recongition-ssd/master/img/snow-three.png" width="45%"><br><br>
</div>

<p align="center">
<iframe width="590" height="315" src="https://www.youtube.com/embed/QVAgJ0P7epY" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</p>

<br>

## Drone Experiments 

### Aerial Drone Experiment

<div align="center">
  <img src="https://raw.githubusercontent.com/kyle-w-brown/vehicle-recongition-ssd/master/img/drone.PNG" width="45%"> <img src="https://raw.githubusercontent.com/kyle-w-brown/vehicle-recongition-ssd/master/img/drone-two.PNG" width="45%"><br><br>
</div>

<p align="center">
<iframe width="590" height="315" src="https://www.youtube.com/embed/fAo_G2XjxKY" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</p>

<br>

### Parking Lot Drone Experiment

<div align="center">
  <img src="https://raw.githubusercontent.com/kyle-w-brown/vehicle-recongition-ssd/master/img/drone-plaza.PNG" width="45%"> <img src="https://raw.githubusercontent.com/kyle-w-brown/vehicle-recongition-ssd/master/img/drone-plaza-two.PNG" width="45%"><br><br>
</div>

<p align="center">
<iframe width="590" height="315" src="https://www.youtube.com/embed/vWDNv6RghDU" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</p>

---

<br>

# PowerPoint

## Link to MS PowerPoint: [Vehicle Recognition using SSD's for AV's](https://github.com/kyle-w-brown/vehicle-recongition-ssd/blob/master/CSC%207991%20Final%20Presentation/SSD-Vehicle-Recognition_Group-2.pptx?raw=true) 


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

<p align="center">
<iframe width="650" height="415" src="https://nbviewer.jupyter.org/github/kyle-w-brown/vehicle-recongition-ssd/blob/master/CSC%207991%20Final%20Presentation/SSD-Vehicle-Recognition_Group-2.pdf" title="arXiv:1512.02325v5" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</p>

---

<br>

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

<br>

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
