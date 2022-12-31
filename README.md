# Comparative Analysis of Skin Cancer (Benign vs. Malignant) Detection Using Convolutional Neural Networks
![alt txt](https://github.com/Sajid-Bit/Skin-Cancer/blob/main/images/image2.avif)

As a content creator and educator, I am constantly looking for awesome projects that I find useful and share them with the broader community. I am not the only one doing this. There are lots of people that share fun projects that they find interesting and useful. This is how projects go viral and gain lots of visibility. From my observation, there are a few components that make certain machine learning projects stand out from the rest. If your goal is to build a portfolio or create impactful and unique projects for the community, here are a few areas you can focus on to make your projects compelling and stand out from the rest.

# 1. Convolutional Neural Network (CNN)

We live in a world where people are suffering from many diseases. Cancer is the most threatening of them all. Among all the variants of cancer, skin cancer is spreading rapidly. It happens because of the abnormal growth of skin cells. The increase in ultraviolet radiation on the Earth's surface is also helping skin cancer spread in every corner of the world. Benign and malignant types are the most common skin cancers people suffer from. People go through expensive and time-consuming treatments to cure skin cancer but yet fail to lower the mortality rate. To reduce the mortality rate, early detection of skin cancer in its incipient phase is helpful. <br /> In today's world, deep learning is being used to detect diseases. `The convolutional neural network (CNN)` helps to find skin cancer through image classification more accurately. We used CNN models and a comparison of their working processes for finding the best results .  <br /> 
In this dataset, there are 6594 images of benign and malignant skin cancer. Using different approaches: <br /> 
### We have gained accurate results for
>  VGG16 (93.18%)<br />
> SVM (83.48%)<br />
> ResNet50 (84.39%)<br />
> Sequential_Model_1 (74.24%)<br />
> Sequential_Model_2 (77.00%)<br />
> Sequential_Model_3 (84.09%) <br />
The `VGG16 model` has given us the highest accuracy of 93.18%.

# 2. Work process of the system block diagram
This system begins by preprocessing data taken from Google Drive into its system. Then data is normalized by null value reduction, image resizing, labeling images, and many more. Normalizing data is one of the most important factors in this project as it helps to decrease value loss. After processing, the data system is trained with six different neural network models (SVM, VGG16, ResNet50, sequential 1/2/3). After training data with trained images, it is fine-tuned to get the maximum accuracy. One of the most important tasks in this process is to check the overfitting and underfitting of these models. When the system is ready for the final process, test data is used to predict and get an accurate output. This work process is maintained throughout the study.

![test](https://github.com/Sajid-Bit/Skin-Cancer/blob/main/images/image4.jpg)

# 1 - Convolutional Neural Network (CNN)
Neural networks are one of the most beautiful programming paradigms ever devised. Anyone can instruct the computer what to do in the traditional method of programming, breaking large issues down into many small, carefully defined jobs that the computer can readily complete. In a neural network, on the other hand, users do not tell the computer how to solve their problems [11]. Rather, it learns from observational data and comes up with its own solution to the problem. CNN's weight-sharing function, which reduces the number of network parameters that can be trained and helps to avoid overfitting by the model and increase generalization, is one of the key reasons for considering CNN in such a circumstance.

![alt txt](https://github.com/Sajid-Bit/Skin-Cancer/blob/main/images/JHE2021-5895156.002.jpg)

 # 2 - SVM model
SVM has three major qualities when used to predict the regression equation. To begin, SVM uses a collection of linear functions specified in a high-dimensional area to calculate the regression. After that, SVM uses a Vapnik-insensitive loss function to evaluate risk and perform regression estimation via risk minimization. Finally, SVM employs a risk function that combines empirical error with a regularization component obtained from the Selectively Reliable Multicast Protocol (SRMP) . For classification problems, SVM works as a supervised learning-based binary classifier that outperforms other classification algorithms . An SVM distinguishes between two classes by creating a classification hyperplane in a high-dimensional feature space.
![alt txt](https://github.com/Sajid-Bit/Skin-Cancer/blob/main/images/svm.jpg)



# Dataset

The project dataset is openly available on Kaggle (SIIM-ISIC Melanoma Classification, 2020). It consists of around forty-four thousand images from the same patient sampled over different weeks and stages. The dataset consists of images in various file format. The raw images are in DICOM (Digital Imaging and COmmunications in Medicine), containing patient metadata and skin lesion images. DICOM is a commonly used file format in medical imaging. Additionally, the dataset also includes images in TFRECORDS (TensorFlow Records) and JPEG format.
 
> Simple of the Data set

Figure 2 is labelled as benign **melanoma** in the dataset.
<p align="center">
  <img alt="benign melanoma" src="https://github.com/Sajid-Bit/Skin-Cancer/blob/main/Data/benign/ISIC_0015719.jpg" width="30%">
&nbsp; &nbsp; &nbsp; &nbsp;
  <img alt="benign melanoma" src="https://github.com/Sajid-Bit/Skin-Cancer/blob/main/Data/benign/ISIC_0052212.jpg" width="30%">
</p>

Figure 2 is labelled as **malignant melanom**a in the dataset.

<p align="center">
  <img alt="benign melanoma" src="https://github.com/Sajid-Bit/Skin-Cancer/blob/main/Data/malignant/ISIC_0232101.jpg" width="30%">
&nbsp; &nbsp; &nbsp; &nbsp;
  <img alt="benign melanoma" src="https://github.com/Sajid-Bit/Skin-Cancer/blob/main/Data/malignant/ISIC_0247330.jpg" width="30%">
</p>

## [Data Augmentation](./Notebooks/Data%20Augumentation.ipynb)

In a small size dataset, image augmentation is required to avoid overfitting the training dataset. After data aggregation, we have around **46k images in the training set**. The dataset contains significant class imbalance, with most of the classes have an **"Unknown"** category (Table 2). We have defined our augmentation pipeline to deal with the class imbalance. The augmentation that helps to improve the prediction accuracy of the model is selected. The selected augmentation are as follows:
1. **Transpose**: A spatial level transformation that transposes image by swapping rows and columns.
2. **Flip**: A spatial level transformation that flip image either/both horizontally and/or vertically. Images are randomly flipped either horizontally or vertically to make the model more robust.
3. **Rotate**: A spatial level transformation that randomly turns images for uniform distribution. Random rotation allows the model to become invariant to the object orientation.
4. **RandomBrightness**: A pixel-level transformation that randomly changes the brightness of the image. As in real life, we do not have object under perfect lighting conditions and this augmentation help to mimic real-life scenarios.
5. **RandomContrast**: A pixel-level transformation that randomly changes the contrast of the input image. As in real life, we do not have object under perfect lighting conditions and this augmentation help to mimic real-life scenarios.
6. **MotionBlur**: A pixel-level transformation that applies motion blur using a random-sized kernel.
7. **MedianBlur**: A pixel-level transformation that blurs input image using a median filter.
8. **GaussianBlur**: A pixel-level transformation that blurs input image using a gaussian filter.
9. **GaussNoise**: A pixel-level transformation that applies Gaussian noise to the image. This augmentation will simulate the measurement noise while taking the images
10. **OpticalDistortion**: Optical distortion is also known as Lens error. It mimics the lens distortion effect.
11. **GridDistortion**: An image warping technique driven by mapping between equivalent families of curves or edges arranged in a grid structure.
12. **ElasticTransform**: A pixel-level transformation that divides the images into multiple grids and, based on edge displacement, the grid will be distorted. This transform helps the network to have a better understanding of edges while training.
13. **CLAHE**: A pixel-level transformation that applies Contrast Limited Adaptive Histogram Equalization to the input image. This augmentation improves the contrast of the images.
14. **HueSaturationValue**: A pixel-level transformation that randomly changes hue, saturation and value of the input image.
15. **ShiftScaleRotate**: A spatial level transformation that randomly applies affine transforms: translate, scale and rotate the input. The allow scale and rotate the image by certain angles
16. **Cutout**: A spatial level transformation that does a rectangular cut in the image. This transformation helps the network to focus on the different areas in the images.

![alt txt](https://github.com/Sajid-Bit/Skin-Cancer/blob/main/images/image2.avif)
 
 
