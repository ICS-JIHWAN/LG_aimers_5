import os
import numpy as np
import pandas as pd
import torch


class x_y_merge_data():
    def __init__(self, path):
        super(x_y_merge_data, self).__init__()
        train = pd.read_csv(path + 'x_y_merge.csv').iloc[:, 1:]
        self.train_X = train.iloc[:, :-1]
        self.train_Y = train.iloc[:, -1]

        self.tmp_x, self.tmp_y = self.train_X.values, self.train_Y.values

    def __len__(self):
        return len(self.train_X)

    def __getitem__(self, idx):
        X = torch.from_numpy(self.tmp_x)[idx]
        Y = torch.from_numpy(self.tmp_y)[idx]

        one_hot_Y = self.one_hot_encoding(Y)


        return {'X': X, 'Y': Y, 'one_hot_Y':one_hot_Y}

    def one_hot_encoding(self, Y):
        one_hot = np.zeros(1)

        if Y == 'Normal':
            one_hot = 1.0

        elif Y == 'AbNormal':
            one_hot = 0.0

        return one_hot


if __name__ == '__main__':
    path = "/storage/mskim/aimers/data/"
    data = x_y_merge_data("/storage/mskim/aimers/data/").__getitem__(0)
