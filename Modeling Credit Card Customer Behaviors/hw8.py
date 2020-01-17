from numpy import genfromtxt
import csv
import numpy as np
import math
import matplotlib.pyplot as plt
import copy
import random
import torch
from torch import nn
import datetime
import pandas as pd
from torch.utils import data
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn import metrics

# Datsets.
class Dataset(data.Dataset):
    def __init__(self, features, labels):
        'Initialization'
        self.features = features
        self.labels = labels
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.labels)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        X = self.features[index]
        y = self.labels[index]

        return X, y


class MLP(torch.nn.Module):
    def __init__(self, num_features):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, X):
        # print(X)
        output = self.layers(X)
        # print(output)
        return output



def main():
    # Load train data.
    train_data = pd.read_csv("hw08_training_data.csv")
    train_label = pd.read_csv("hw08_training_label.csv")

    # Load test data.
    test_data = pd.read_csv("hw08_test_data.csv")

    # Preprocess the data.
    # 1- Drop Nun Numeric Columns
    train_data = train_data.select_dtypes(['number'])
    test_data = test_data.select_dtypes(['number'])

    # 2- Replace NA with Columns means in train and test data.
    columns_means = train_data.mean()
    train_data = train_data.fillna(columns_means)
    test_data = test_data.fillna(columns_means)

    # 3- Delete ID column.
    del train_label["ID"]
    del train_data["ID"]
    Saved_IDs = test_data["ID"]
    del test_data["ID"]

    # 4- Normalize training Data.
    columns_max = train_data.max()
    columns_min = train_data.min()
    train_data = (train_data - columns_min) / (columns_max - columns_min)
    # 5- Normalize test data.
    test_data = (test_data - columns_min) / (columns_max - columns_min)

    # 6- Convert Test data into Tensor.
    target_X_test = torch.FloatTensor(test_data.values)

    predictions = []
    for label_num in range(6):
        print("Working on problem ", label_num+1)
        #Get labels and training instances for this particular problem.
        target_train_label = train_label[train_label.columns[label_num]].dropna()
        target_train_data = train_data.iloc[target_train_label.index]

        # Split data into training and validation.
        target_X_train, target_X_valid, target_y_train, target_y_valid = train_test_split(target_train_data,
                                                                                          target_train_label,
                                                                                          test_size=0.2,
                                                                                          stratify=target_train_label,
                                                                                          random_state=42)
        target_dataset = Dataset(torch.FloatTensor(target_X_train.values),
                             torch.Tensor(target_y_train.astype("int64").values))
        target_loader = DataLoader(dataset=target_dataset, batch_size=32, shuffle=True)

        # Convert Validation Sets into Tensor.
        target_X_valid = torch.FloatTensor(target_X_valid.values)
        target_y_valid = torch.Tensor(target_y_valid.astype("int64").values)

        # Define the model, loss function, and the optimizer.
        target_model = MLP(len(target_train_data.columns))
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(target_model.parameters(), lr=0.001)

        # Training loop.
        losses = []
        n_epochs = 5
        for epoch in range(n_epochs):
            epoch_loss = 0
            epoch_num = 0
            for x_batch, y_batch in target_loader:
                predicted_y = target_model(x_batch)
                loss = criterion(predicted_y.squeeze(), y_batch)
                losses.append(loss.data.item())
                epoch_loss += loss.data.item()
                epoch_num += 1

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            # Validation Set calc and plotting.
            predicted_y_val = target_model(target_X_valid)
            val_loss = criterion(predicted_y_val.squeeze(), target_y_valid).data.item()

            predicted_y_val = target_model(target_X_valid)
            auc = roc_auc_score(target_y_valid, predicted_y_val.detach().numpy())
            print("Epoch: ", epoch, "Training Loss: ", epoch_loss / epoch_num, "Validation Loss: ", val_loss, "Auc: ",
                  auc)

        # Predicting Test data.
        test_predictions = target_model(target_X_test)
        target_predictions = test_predictions.detach().numpy()
        predictions.append(target_predictions)
        
        # This part was for plotting purposes only.  
        #y_score = target_model(target_X_valid).detach().numpy()
        #y_score = y_score.reshape((len(y_score), ))
        #y_true = target_y_valid.numpy().reshape((len(target_y_valid.numpy()), ))
        #fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
        #plt.plot(fpr, tpr)
        #plt.ylabel('TP-rate')
        #plt.xlabel('FP-rate')
        #plt.title(label = "Target " + str(label_num+1))
        #plt.show()

    # Saving into file.
    IDs = Saved_IDs.to_numpy()
    IDs = IDs.reshape((len(IDs), 1))
    test_data_final = IDs
    for pred in predictions:
        test_data_final = np.concatenate((test_data_final, pred), axis=1)
    np.savetxt("hw08_test_predictions.csv", test_data_final, delimiter=",")
    print("Predictions saved into file.")


if __name__ == '__main__':
    main()


