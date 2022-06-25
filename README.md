# Tomato Disease Classification with Deep Learning

Contributors: Evie Klaassen and Ka Yam

This project was completed for MSDS631: Deep Learning, as part of the Master's in Data Science program at the University of San Francisco.

## Background

Tomatoes are one of the most common vegetables grown by home gardeners, as well as one of the most widely produced crops in the California agriculture industry. For this reason, understanding tomato diseases and their respective appearances serves many, and was the motivation behind this project. Various diseases can be identified using the leaves of tomato plants. Using PyTorch and other deep learning techniques, we built a classification model to identify what disease a tomato plant may have, based on images of the plant's leaves.

## Data Sources

We used two datasets of tomato leaf images for this project. The first, and largest, is the [PlantVillage dataset from Kaggle](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset). From this dataset, we used all images of tomato leaves, which had color, greyscale, and segmented iamges with 10 total classes. The second dataset, found [here](https://data.mendeley.com/datasets/369cky7n39/1), consists of color images with 3 classes, and supplemented the PlantVillage dataset. In total, our complete dataset had 54,783 images with 10 classes:

- Target Spot
- Tomato Mosaic Virus
- Late Blight
- Leaf Mold
- Bacterial Spot
- Early Blight
- Tomato Yellow Leaf Curl Virus
- Spider Mites
- Septoria Leaf Spot
- Healthy

Here are some example images from our dataset:

Bacterial Spot:

![Bacterial Spot](https://github.com/evieklaassen/tomato-disease-classifier/blob/main/readme_images/bac_spot.JPG)

Leaf Mold:

![Leaf Mold](https://github.com/evieklaassen/tomato-disease-classifier/blob/main/readme_images/mold.JPG)

Late Blight:

![Late Blight](https://github.com/evieklaassen/tomato-disease-classifier/blob/main/readme_images/late_blight.JPG)

Healthy Tomato Leaf:

![Healthy](https://github.com/evieklaassen/tomato-disease-classifier/blob/main/readme_images/healthy.JPG)

## Preprocessing

We used the [Albumentations](https://albumentations.ai/) package for our data preprocessing. For our candidate models, we needed to resize our images to 224 by 224, and we also performed a series of image augmentations with varying probabilities to expand our dataset and improve the performance of our models. We explored spatial transformations (i.e. HorizontalFlip, VerticalFlip, and Rotate) and pixel-level transformations (i.e. RandomBrightnessContrast, Blur, and GaussNoise).

## Candidate Models

For our model selection process, we experimented with 3 different pre-trained models. We trained each model for 5 epochs on our training dataset, which was 60% of our full dataset, then we evaluated each model on our validation dataset, which was 20% of our full dataset.

#### Fine-Tuned VGG16

VGG-16 is a convolutional neural network that is 16 layers deep. The fine-tuned model classifies 10 classes(of diseases) instead of the 1000 categories it was trained on. To do so, we replaced the final classifier layer with a linear layer going from 4096 features to 10 features. To finetune the model, we trained different layers at different learning rates (1e-4, 5e-4, 1e-3). The later the layer, the more we adjusted the features. At the end of 5 epochs, we achieved a train accuracy of 99.39% and a valid accuracy of 97.20%.

#### GoogleNet


#### Partially Frozen ResNet18

#### Model Performance

Below are the validation loss (cross entropy loss) and accuracy for each of our models:

![Validation Loss](https://github.com/evieklaassen/tomato-disease-classifier/blob/main/readme_images/val_loss.png) ![Validation Accuracy](https://github.com/evieklaassen/tomato-disease-classifier/blob/main/readme_images/val_acc.png)

### Final Model 

## Future Directions
