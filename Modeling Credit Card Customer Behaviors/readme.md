## Engr421: Machine learning Hw8 Report

## Name: Ahmed Masry

## ID: 61868

# INTRODUCTION:

In this assignment, I have implemented an MLP using Pytorch in Python for Modeling Credit Card
Customer Behavior. We have 6 binary classification problems; each represents whether a customer will
perform a specific action. In my implementation, I first preprocess the data. Then I use a for loop to do the
training and testing for each problem. At the end, I save my predictions probabilities in a csv file.

# REQUIREMENTS:

- Python
- Pytorch
- Numpy
- Pandas
- Sklearn
- Matplotlib.

# DATA PREPROCESSING:

```
1) Loading the Data:
  o All the training and test data files are loaded using Pandas read_csv function and stored into
Pandas frames.
2) Handling NAN values:
  o The Nan values in each column in the training data have been replaced with the mean of the
remaining values in the column. In the test set, we do the same but using the means of the columns of
the training data as well.
3) Dropping Nun Numeric columns:
  o From my experiments, the AUC value on the validation set is higher when I drop the Nun Numeric
columns. So I dropped them in my final version.
4) Normalizing the data:
  o In order to scale the features values in the training data, I normalized all the columns using
the maximum and minimum of the columns. For the test data, I have done the same using the
mins and maxs from the training data.
5) Splitting the data into Training and Validation:
  o In order to evaluate my model performance during the training, we need a validation set. So
I used the Sklearn train_test_split function to split my training data into two sets: 80% train
data and 20% validation set.
  o I have also noticed that the number of instances with each label are not the same in the
training data. So during splitting, I made sure that the data is stratified.
6) Pytorch Data Loader:
  o Using Pytorch data.Dataset API, I implemented a class called Dataset for storing and
retrieving the data easily.
  o I have also used the Pytorch Data Loader API to mini-batch of size 32 and shuffle my
training data.
```
# MODEL DESCRIPTION

In this assignment, I implemented a Multi-Layer Perceptron (MLP) using the Module API from the pytorch
library. My model has 5 fully-connected hidden Linear layers each followed by a ReLU activation function
in addition to the output layer which is followed by a Sigmoid function (since we are doing binary
classification). The number of hidden nodes in each layer is as follows:

- Layer1: 512 hidden nodes
- Layer2: 256 hidden nodes
- Layer3: 128 hidden nodes
- Layer4: 63 hidden nodes
- Layer5: 32 hidden nodes.
- Output layer: 1 node.

# TRAINING PROCEDURE:

In order to train the MLP model, I used the following

- Optimizer: Adam
- Learning rate: 0.
- Loss function: Binary Cross Entropy.
- Epochs: 5

In each epoch, I train my model on all the batches of the training set. Then I calculate and print the
following:
1 - The average training loss in the epoch
2 - Validation Set Loss.
3 - AUC value. (Calculated using the Sklearn roc_auc_score function)

# RESULTS:

After finishing the training, I calculated the AUC value for each target on the validation set and got the
following values:

- Target1 ~= 0.73
- Target2 ~= 0.77
- Target3 ~= 0.77
- Target4 ~= 0.82
- Target5 ~= 0.72
- Target6 ~= 0.78

Moreover, I plotted the AUC curve for each target (using Matplotlib and Sklearn) as you can see in the next figure.
![alt text](https://github.com/AhmedMasryKU/ML-Final-Projects/blob/master/Modeling%20Late%20Payments%20for%20Credit%20Card%20Bills/figure.png)
# CONSLUSION:

As we can see from the AUC values, my MLP model does a good job in these 6 binary classification
problems.


