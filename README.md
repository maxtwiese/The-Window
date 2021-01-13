# The Window

*Exploration of computer vision for pupil dilation measurement.*

## 1 Introduction

### 1.1 Background

Interest in machine learning for periocular recognition has increased with the donning of masks to prevent the spread of COVID-19--  periocular refers to the surrounding region of the eye including the eyebrow and complete canthus points. With the subsequent advancements in iris segmentation I aim to apply computer vision to pupil dilation tracking. Pupil dilation can be used to measure emotional arousal and autonomic activation. [(1)](Ref1) The goal of this project is to train a model that can measure pupil dilation from a webcam with meaningful feedback in real time. Measuring autonomic activation could be used to improve tele-mental health, especially in time boxed treatment plans such as the Cognitive Behavioral Therapy treatment plan used by the VA.

### 1.2 Data

THe UBIRIS Datasets were produced to introduce noisy images into the corpus of labeled data for computer vision research on ocular recognition. The UBIRISv1 included obstructions and with UBIRISv2 the researchers increased noise from movement and variable distance. The periocular dataset is UBIRISv2 with a wider crop to train finding an eye within the face. From the source:

> The UBIPr dataset  is a version of the UBIRIS.v2 set, with images cropped in a way that they cover wider parts of the ocular region than the original UBIRIS.v2 data. It is particularly suited for experiments related with Periocular Recognition. [(2)](Ref2)

The label data for each image was in separate text file. An example of each file can be seen with **Figure 1**.

```
 CurrentDir;FileName;Size;Gender;CamDist;LPoseAngle;LGazeAngle;LPigm;LEyeClosure;LHairOcclusion;LGlass;LCornerOutPtX;LCornerOutPtY;LIrisCenterPtX;LIrisCenterPtY;LCornerInPtX;LCornerInPtY;LEyebrowOutPtX;LEyebrowOutPtY;LEyebrowMidPtX;LEyebrowMidPtY;LEyebrowInPtX;LEyebrowInPtY;LEyeSizeX;LEyeSizeY;
 C:\Users\Chandra\Desktop\Chandra\My Project\NECOVID\POSE ESTIMATION\Dataset\Periocular dataset UBI\original\2007_12_10;
 IMG_0266.JPG;
 2912;
 4368;
 3;
 Male;
 8;
 0;
 0;
 Medium;
 Light;
 No;
 No;
 219;
 162;
 193;
 257;
 219;
 369;
 159;
 88;
 53;
 197;
 135;
 389;
 401;
 501;
```

*Figure 1.1 Example label text file.*

This dataset is 10,199 images of varying resolution and corresponding label files for each. These images are from the 11,102 images for the UBIRISv2 Dataset that was built from 522 Irises of 261 volunteers. **Table 1.1** describes some demographic information of those volunteers.

| Gender | %     | Age      | %     | Iris Pigmentation | %     |
| :----- | :---: | :------- | :---: | :---------------- | :---: |
| Male   | 54.4% | [0, 20]  | 6.6%  | Light             | 18.3% |
| Female | 45.6% | [21, 25] | 32.9% | Medium            | 42.6% |
|        |       | [26, 30] | 23.8% | Heavy             | 39.1% |
|        |       | [31, 35] | 21.0% |                   |       |
|        |       | [36, 99] | 15.7% |                   |       |

*Table 1.1: Volunteer Demographics*


## 2 Current Work

### 2.1 Environment and Toolkit

| EC2 Instance | vCPU | GPU            | GPU Memory |
| ------------ | ---- | -------------- | ---------- |
| p3.8xlarge   | 32   | 4x NVIDIA V100 | 16 GB      |

*Table 2.1: Instance*

I am training on the PyTorch Vision framework. [(3)](Ref3) The following files in `src` are a part of that framework and used to execute the `engine` module when called in `ddpFasterRCNN.py`:

- `coco_eval.py`
- `coco_utils.py`
- `engine.py`
- `transforms.py`
- `utils.py`

To utilize the 4 GPUs of the p3.8xlarge EC2 instance I have my model wrapped into a Distributed Data Parallel (DDP) that splits my batches across the 4 GPU training 4 models in parallel and learning from their mean loss. The model file, [ddpFasterRCNN.py](src/ddpFasterRCNN.py) is executed using the BASH script [ddp_agent.sh](src/ddp_agent.sh) to establish the proper environment in PyTorch.

### 2.2 Method

#### 2.2.1 Regional Convolutional Neural Network Overview

Iris segmentation is a two part machine learning problem: eye detection and segmentation. A Regional Convolutional Neural Network (R-CNN) is well equipped to solve detecting an eye within an image. To train an R-CNN a machine learning engineer will need bounding boxes for the Region of Interest (RoI) and a label for that RoI. In R-CNNs the general workflow is to search regions of an image and classify within the regions. This is very computationally expensive and has since been improved upon with Fast R-CNN, Faster R-CNN, and Mask R-CNN. Faster R-CNN can process around 5 frames per second to construct a bounding box with an appropriate classification. Mark R-CNN finds a by pixel level mask that covers the exact object of interest instead of a rectangle RoI. It uses Faster R-CNN as a backbone and requires labels not easily built from the information given in our data  to cold start learning. As such, I will build a Faster R-CNN and then solve the segmentation portion with Hough transformations or a Gaussian Mixture Model. The behavior of a Faster R-CNN is to:

1. Intake an image.
2. Build anchor points across the entire image.
3. Build regions of interest at each anchor point for a determined set of lengths and weights (i.e. [2, 4, 8] and [0.5, 1, 2])
4. Remove all RoIs with coordinates outside of the image.
5. Give importance to RoIs based on how much of the bounding box overlaps with the RoI.
6. Search within important regions from that first look.

Training the Faster R-CNN on periocular images would allow it detect an eye without first finding an entire face. It constructs it's entire knowledge fo an eye from surrounding folds, eyebrows, and the portions of the eye I care to measure.

#### 2.2.2 Data Pipeline

Label files for my data, example shown in Figure 1.1, were extremely messy. The header is the first row and is followed by rows for its columns.  They also include more entry rows than they do categories in the header. Size in the header refers to height,  weight,  and channel rows; of which, only channel is accurate information for the image files. I needed to created a `.csv` file from this these 10,199 `.txt` files that has image file names aligned with their bounding box coordinates. I wrote a a python script to pull only the rows I needed as strings, clean the data in pandas, and output a `.csv`.

Eye recognition bounding boxes can be constructed in a few ways. Since this data involves many angles and high noise I use the proposed algorithm of Liu, et al [(4)](Ref4) where the box is contsructed using the inner and outer canthus points of the eye. Using the center of the eye may seem more straight forward, but with multiple camera angles the iris center may not be the visual center of the eye for the network to search from.

*With canthus points (x<sub>1</sub>, y<sub>1</sub>) and (x<sub>2</sub>, y<sub>2</sub>) you calculate the Euclidean distance between those points as D and the midpoint of the line that connects the two canthus points as L<sub>p</sub>). Then you can generate the top left and bottom right corners of your rectangular RoI with chosen constants a and b.*

![Liu_RoI](data/Liu_RoI.png)

Bounding box coordinates built from this algorithm and image file names were output to a helper document by [csvBuilder.py](src/csvBuilder.py) that is used by [Datasets.py](src/Datasets.py) to extend the PyTorch Dataset class to the UBIRIS Periocular images. The dataset reads the images in as PIL Images, resizes them, normalizes for the requirements set in Pytorch's Model Zoo, and converts them to a Tensor. It creates a target dictionary for each image that includes the bounding box, the classification label for that box (here they are all 1 since we are classifying eye or not eye), the area of the box, and background.

I used a pretrained Faster R-CNN with a ResNet-50 backbone from the PyTorch Model Zoo. I froze all layers' weights by setting their gradient requirement to false. I replaced the head function, the classifier, with a fresh Faster-CNN head searching for only two classes. This allowed me to train only one layer and save memory.

### 2.3 Road Blocks

My model can learn forwards, but cannot handle the increased gradient of backward propagation. In forwards tests the learning is extremely early and minute before the loss begins to stabilize to a bound within the hundredth. Without backward propagation I cannot get a reliable read on validation accuracy. Loss for a Faster R-CNN is defined as the sum of loss on the regression and the classification. More specifically it is the sum of negative log of the discrete probability distribution for each classification the of smooth L1 losses for each bounding box coordinate. The lack of interpretability in this metric lead me to the Vision framework from PyTorch. I switched to the PyTorch [engine](src/engine.py) from my built framework after seeking guidance from a senior data scientist on my memory issues in back propagation and lack of interpretable metrics to troubleshoot. The increase in GPU usage from my model to adapt into their better engine has exceeded the threshold of 16GB per GPU over 4 GPUs. The p3.8xlarge should be able to handle this load which means the error comes from implementation of these tools or modeling of the problem. I am to new to Machine Learning and brand new to PyTorch, so I steering my project back to research and scoping for the next month. (01-13-2021)

## 3 Next Steps

### 3.1 Environment and Toolkit

| EC2 Instance | vCPU | GPU            | GPU Memory |
| ------------ | ---- | -------------- | ---------- |
| p3.8xlarge   | 32   | 4x NVIDIA V100 | 32 GB      |

1. Migrate to model management platform for further granularity on where memory is added to GPUs.

2. Mount the S3 container with UBIRIS Periocular dataset to the EC2 instance.

3. As a backstop, there is an upgrade to 32GB GPU offered by AWS. I am need to work to reduce memory on GPUs before I keep buying more GPU Memory.

### 3.2 Method

1. Work on pre-established databases to increase knowledge of platform and problem space.

2. Build a regression CNN from scratch.

3. Reduce amount of work sent to GPU.

4. Replace more than the top layer.

## 4 Sources

<a name="Ref1">(1)</a>Bradley MM, Miccolli L, Escrig MA, Lang PJ. The pupil as a measure of emotional arousal and autonomic activation. [National Institutes of Health](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3612940/)

<a name="Ref2">(2)</a>Chandrashekhar Padole, Hugo Proença; Periocular Recognition: Analysis of Performance Degradation Factors, in Proceedings of the Fifth IAPR/IEEE International Conference on Biometrics – ICB 2012, New Delhi, India, March 30-April 1, 2012.  [UBIPr](http://iris.di.ubi.pt/ubipr.html)

<a name="Ref3">(3)</a> [Pytorch/Vision](https://github.com/pytorch/vision.git)

<a name="Ref4">(4)</a>Liu P, Guo J, Tseng S, Wong K, Lee JD, Yao C, Zhu D (2017) Ocular recognition for blinking eyes. IEEE Trans Image Process 26(10):5070–5081. [doi]https://doi.org/10.1109/TIP.2017.2713041