# Development and Validation of a Multi-Task Deep Learning Algorithm for the Detection and Recognition of Precancerous Lesion in Esophagus and Colon

This folder includes source code of tensorflow implementation for "Development and Validation of a Multi-Task Deep Learning Algorithm for the Detection and Recognition of Precancerous Lesion in Esophagus and Colon"

## 1. Overview
Esophagoscopy and colonoscopy are gold standards for the prevention of esophagus and colorectal cancers, because of their remarkable performances in reducing the morbidity and mortality. However, the screening accuracy is greatly influenced by man-induced factors such as lack of experience, fatigue, negligence, etc. In this study, we demonstrate that a multi-task deep learning algorithm can simultaneously complete the detection and recognition of lesions in esophagus or colon with only one network inference, in real time and with high sensitivity and specificity. For the detection and recognition of colonic polyps, we develop a multi-task deep learning algorithm by using 2,894 (1,337 abnormal, 1,557 normal) colonoscopy images, and validate it on five datasets containing 38,648 colonoscopy images (achieving an AUC of 99.18%). For the diagnosis of esophageal lesions, we develop a multi-task deep learning algorithm by using 2,428 (1,332 abnormal, 1,096 normal) esophagoscope images, and validate it on newly collected 187 endoscopic images (achieving per-image-sensitivity, 94.51%; per-image-specificity, 97.92%; AUC, 99.46%; mIoU, 77.81%). A total of 5,847 and 746 patients are included in the studies of colonoscopy and esophagoscopy, respectively. Furthermore, in order to verify the potential clinical value of the algorithm, we conduct a study of endoscopists screening early esophageal lesions with the assistance of the algorithm. The average diagnostic ability of eight endoscopists (2 seniors, 3 mid-levels, 3 juniors) is all improved in terms of sensitivity (74.43% vs. 87.35%), specificity (89.6% vs. 92.3%), accuracy (82.2% vs. 89.9%). Finally, we have used the developed system for real-time visual assistance during colonoscopy and esophagoscopy (see Supplementary Videos 1 to 10). The algorithm can process at least 135 frames per second with a latency of 7.157 ms in real-time video analysis. The system is able to aid endoscopists screening precancerous lesions in esophagus and colon.

## 2. System requirements
 ###  2.1 Hardware Requirements
    The package requires only a standard computer with GPU and enough RAM to support the operations defined by a user. 
    For optimal performance, we recommend a computer with the following specs:
    RAM: 32+ GB
    CPU: 8+ cores, 3.6+ GHz/core
    GPU：11+ GB (such as GeForce RTX 2080 Ti GPU)
  
  ###  2.2 Software Requirements
 ####   2.2.1 OS Requirements
          This package is supported for Windows operating systems.
 ####   2.2.2 Installing CUDA 10.0 on Windows 10.
 ####   2.2.3 Installing Python 3.6+ on Windows 10.
 ####   2.2.4 Python Package Versions
	    numpy 1.15.4
	    tensorflow 1.8.0
	    scipy 1.1.0
  	    scikit-learn 0.20.2	
	    scikit-image 0.14.1
	    opencv-python 3.3.0.10
	    matplotlib 3.0.2
	    pillow 5.3.0
	 
	 
##  3. Installation Guide
  A working version of CUDA, python and tensorflow. This should be easy and simple installation. 
  CUDA(https://developer.nvidia.com/cuda-downloads)
  tensorflow(https://www.tensorflow.org/install/) 
  python(https://www.python.org/downloads/)
  

##  4. Usage of source code
  Enter into folder "Esophagus" for classification and segmentation of esophagus precancerous lesions.

 ###   4.1 Training
 ####   4.1.1 run genTFrecords.py to make TFrecords file. The TFrecords file will be saved in "data" sub-folder.
 ####   4.1.2 run train.py to train a network for detection and recognition. The model will be save in "model" sub-folder.

###  4.2 Test
####    4.2.1 run test.py to obtain initial results. The results will be saved in "results" sub-folder.

 ###   4.3 Evaluation
####    4.3.1 run getEval.py 
to obtain the results of evaluation on detection accuracy (mIou). This script should be run after test.py and the results will be saved in "results" sub-folder.
####    4.3.2 run getEval_class.py 
to obtain the results of evaluation on recognition. This script should be run after test.py and the results will be saved in "results" sub-folder.
####    4.3.3 run genMaskedImg.py 
to obtain intuitionistic detection results. This script should be run after test.py and the results will be saved in "results" sub-folder.
 ####   4.3.4 run visualization.py 
 to obtain the visualization of feature maps from different intermediate layers. The results will be saved in "results" sub-folder. (only presented in Esophagus)
 ####   4.3.5 run test_high_dimension.py first and then run visualization_high_dimension.py 
 to visualize the fc2 layer using PCA (by virtue of tensorboard). (only presented in Esophagus)
