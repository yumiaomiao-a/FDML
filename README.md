# FDML
This is the official repository of the article "FDML: Feature Disentangling and Multi-view Learning for Face Forgery Detection".

Recent advances in realistic facial manipulation techniques have led to a growing interest in forgery detection due to security concerns. The presence of source-dependent information in both the forged images and the learned representations inevitably confuses the detector. To alleviate this issue, we present a Feature Disentangling and Multi-view Learning framework via multi-view learning and disentangled representation learning in pursuit of better in-domain detection precision as well as cross-domain generalization.

This paper is currently under review, and we will update the paper status here in time. If you use this repository for your research, please consider citing our paper.

This repository is currently under maintenance, if you are experiencing any problems, please open an issue.

####### Download
- git clone https://github.com/yumiaomiao-a/FDML.git
- cd FDML

## Prerequisites:  
We recommend using the Anaconda to manage the environment.  
- conda create -n fdml python=3.6  
- conda activate fdml  
- conda install -c pytorch pytorch=1.7.1 torchvision=0.5.0 numpy=1.19.5 opencv-python=3.4.2.16


## Dataset Preparation
You need to download the publicly available face forensics datasets. In this work, we conduct experiments on Celeb-DF and FaceForensics++ datasets, their official download links are as follows:
- https://github.com/yuezunli/celeb-deepfakeforensics
- https://github.com/ondyari/FaceForensics
