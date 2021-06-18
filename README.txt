A Real-Time Artificial Intelligence System for Classification and Segmentation of Esophagus Precancerous Lesions with Randomized Controlled Trial

This folder includes source code of tensorflow implementation for "A Real-Time Artificial Intelligence System for Classification and Segmentation of Esophagus Precancerous Lesions with Randomized Controlled Trial"

1. Overview
The incidence rate of esophageal cancer has been increasing greatly in the past decade. The gold standard for preventing esophagus cancer is routine esophagoscopy because of its remarkable performance in reducing morbidity and mortality. Recently, artificial intelligence has shown great promise in screening esophagus lesions on laboratory annotated datasets. The next step of artificial intelligence-based screening algorithms needs to consider real-time diagnosis and randomized controlled trial when applied to clinical diagnosis. In this study, we firstly demonstrate a novel multi-task deep learning model that can simultaneously complete the classification and segmentation of esophagus precancerous lesions with only one network inference, in real-time and with high sensitivity (94.51%) and specificity (97.92%). Furthermore, we conduct a study of endoscopists screening esophageal lesions with the assistance of the developed algorithm. The average diagnostic ability of eight endoscopists (2 seniors, 3 mid-levels, 3 juniors) is all improved in terms of sensitivity (74.43% vs. 87.35%), specificity (89.6% vs. 92.3%), and accuracy (82.2% vs. 89.9%). Finally, a randomized controlled trial was conducted and demonstrated that the detection rate of the esophageal abnormal lesion in the research group and the control group was 3.7% and 1.7%, respectively (54 abnormal lesions were found in the research group, 28 in the control group. p＜0.001). The screening system can process at least 135 frames per second with a computational time of 7.157 ms in real-time video assistance and is able to aid endoscopists to screen esophagus lesions.

2. System requirements
  2.1 Hardware Requirements
    The package requires only a standard computer with GPU and enough RAM to support the operations defined by a user. 
    For optimal performance, we recommend a computer with the following specs:
    RAM: 32+ GB
    CPU: 8+ cores, 3.6+ GHz/core
    GPU：11+ GB (such as GeForce RTX 2080 Ti GPU)
  
  2.2 Software Requirements
  2.2.1 OS Requirements
          This package is supported for Windows operating systems.
  2.2.2 Installing CUDA 10.0 on Windows 10.
  2.2.3 Installing Python 3.6+ on Windows 10.
  2.2.4 Python Package Versions
	    numpy 1.15.4
	    tensorflow 1.8.0
	    scipy 1.1.0
  	    scikit-learn 0.20.2	
	    scikit-image 0.14.1
	    opencv-python 3.3.0.10
	    matplotlib 3.0.2
	    pillow 5.3.0
	 
	 
3. Installation Guide
  A working version of CUDA, python and tensorflow. This should be easy and simple installation. 
  CUDA(https://developer.nvidia.com/cuda-downloads)
  tensorflow(https://www.tensorflow.org/install/) 
  python(https://www.python.org/downloads/)
  

4. Usage of source code
  Enter into folder "Esophagus" for classification and segmentation of esophagus precancerous lesions.

  4.1 Training
  4.1.1 run genTFrecords.py to make TFrecords file. The TFrecords file will be saved in "data" sub-folder.
  4.1.2 run train.py to train a network for detection and recognition. The model will be save in "model" sub-folder.

  4.2 Test
  4.2.1 run test.py to obtain initial results. The results will be saved in "results" sub-folder.

  4.3 Evaluation
  4.3.1 run getEval.py to obtain the results of evaluation on detection accuracy (mIou). This script should be run after test.py and the results will be saved in "results" sub-folder.
  4.3.2 run getEval_class.py to obtain the results of evaluation on recognition. This script should be run after test.py and the results will be saved in "results" sub-folder.
  4.3.3 run genMaskedImg.py to obtain intuitionistic detection results. This script should be run after test.py and the results will be saved in "results" sub-folder.
  4.3.4 run visualization.py to obtain the visualization of feature maps from different intermediate layers. The results will be saved in "results" sub-folder. (only presented in Esophagus)
  4.3.5 run test_high_dimension.py first and then run visualization_high_dimension.py to visualize the fc2 layer using PCA (by virtue of tensorboard). (only presented in Esophagus)