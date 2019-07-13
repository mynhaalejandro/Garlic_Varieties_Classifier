# Garlic Varieties Classifier
Classification of registered garlic varieties on the basis of their visual markers namely color and shape based features using digital image processing techniques in OpenCV.

## Overview
Garlic varieites profiling is a system which able to classify registered garlic varieties in the Philippines namely Batanes, Ilocos Pink, Ilocos White, Mexican, MMSU Gem, Tan Bolters and VFTA 275 M76 on the basis on their visual features using digital image processing techniques. The images are pre-processed and then their color and shape based features are extracted from the image.

A dataset was created using the extracted features to train and test the model. The model used was Decision Tree Classifier and was able to achieve 75% accuracy, 79% precision, and 75% recall scores.

## Dataset
The dataset used is garlic varieties dataset which can be downloaded from here. There are 7 different class of garlic on this dataset. Each class has about 50 images to work with.
## Requirements
- OpenCV
- Numpy
- Scipy
- Matplotlib
- Pandas
- Seaborn
- Itertools
- Tqdm
- Scikit Learn
- Pydotplus
- Pickle

A prototype was created using the Decision Tree Classifier Model. The prototype was created using a microframework for Python called Flask.


