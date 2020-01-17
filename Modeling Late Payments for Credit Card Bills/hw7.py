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

def main(target_num):
    print("Woking on target ", target_num)
    # Load train data.
    data_train_file_name = "hw07_target" + str(target_num) + "_training_data.csv"
    label_train_file_name = "hw07_target" + str(target_num) + "_training_label.csv"
    target_train_data = pd.read_csv(data_train_file_name)
    target_train_label = pd.read_csv(label_train_file_name)

    # Load test data.
    data_test_file_name = "hw07_target" + str(target_num) + "_test_data.csv"
    target_test_data = pd.read_csv(data_test_file_name)

    # Preprocess the data.
    # 1- Replace NA with Columns means in train and test data.
    columns_means = target_train_data.mean()
    target_train_data = target_train_data.fillna(columns_means)
    target_test_data = target_test_data.fillna(columns_means)

    # 2- Drop Nun Numeric Columns (Only 3 so no big deal).
    target_train_data = target_train_data.select_dtypes(['number'])
    target_test_data = target_test_data.select_dtypes(['number'])

    # 3- Delete ID column.
    del target_train_label["ID"]
    del target_train_data["ID"]
    Saved_IDs = target_test_data["ID"]
    del target_test_data["ID"]

    # 4- Normalize training Data.
    columns_max = target_train_data.max()
    columns_min = target_train_data.min()
    target_train_data = (target_train_data - columns_min) / (columns_max - columns_min)
    # 5- Normalize test data.
    target_test_data = (target_test_data - columns_min) / (columns_max - columns_min)

    # Split data into training and validation.
    target_X_train, target_X_valid, target_y_train, target_y_valid = train_test_split(target_train_data,
                                                                                      target_train_label, test_size=0.2,
                                                                                      stratify=target_train_label,
                                                                                      random_state=42)
    # Convert Validation Sets into Tensor.
    target_X_valid = torch.FloatTensor(target_X_valid.values)
    target_y_valid = torch.Tensor(target_y_valid.astype("int64").values)

    # Convert Test data into Tensor.
    target_X_test = torch.FloatTensor(target_test_data.values)

    target_dataset = Dataset(torch.FloatTensor(target_X_train.values),
                             torch.Tensor(target_y_train.astype("int64").values))
    target_loader = DataLoader(dataset=target_dataset, batch_size=32, shuffle=True)

    target_model = MLP(len(target_train_data.columns))

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(target_model.parameters(), lr=0.001)

    # Training loop.
    losses = []
    n_epochs = 10
    for epoch in range(n_epochs):
        epoch_loss = 0
        epoch_num = 0
        for x_batch, y_batch in target_loader:
            predicted_y = target_model(x_batch)
            loss = criterion(predicted_y, y_batch)
            losses.append(loss.data.item())
            epoch_loss += loss.data.item()
            epoch_num += 1

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Validation Set calc and plotting.
        predicted_y_val = target_model(target_X_valid)
        val_loss = criterion(predicted_y_val, target_y_valid).data.item()

        predicted_y_val = target_model(target_X_valid)
        auc = roc_auc_score(target_y_valid, predicted_y_val.detach().numpy())
        print("Epoch: ", epoch, "Training Loss: ", epoch_loss / epoch_num, "Validation Loss: ", val_loss, "Auc: ", auc)

    # Predicting Test data and saving into csv file.
    test_predictions = target_model(target_X_test)
    predictions = test_predictions.detach().numpy()
    IDs = Saved_IDs.to_numpy()
    IDs = IDs.reshape((len(IDs), 1))
    test_data_final = np.concatenate((IDs, predictions), axis=1)
    predictions_file_name = "hw07_target" + str(target_num) + "_test_predictions.csv"
    np.savetxt(predictions_file_name, test_data_final, delimiter=",")


if __name__ == '__main__':
    main(1)
    main(2)
    main(3)
