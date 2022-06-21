# Tomato Disease Classification with Deep Learning

Contributors: Evie Klaassen and Ka Yam

This project was completed for MSDS631: Deep Learning, as part of the Master's in Data Science program at the University of San Francisco.

## Background

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

![Bacterial Spot](https://github.com/evieklaassen/tomato-disease-classifier/blob/main/readme_images/bac_spot.JPG)

![Leaf Mold](/readme_images/mold.jpg)

![Late Blight](/readme_images/late_blight.jpg)

![Healthy](/readme_images/healthy.jpg)

## Preprocessing

We used the [Albumentations](https://albumentations.ai/) package for our data preprocessing. For our candidate models, we needed to resize our images to 224 by 224, and we also performed a series of image augmentations with varying probabilities to expand our dataset and improve the performance of our models.

## Candidate Models

For our model selection process, we experimented with 3 different pre-trained models. We trained each model for 5 epochs on our training dataset, which was 60% of our full dataset, then we evaluated each model on our validation dataset, which was 20% of our full dataset.

#### Fine-Tuned VGG16

#### GoogleNet

#### Partially Frozen ResNet18

#### Model Performance

Below are the validation loss (cross entropy loss) and accuracy for each of our models:

### Final Model 

## Future Directions
