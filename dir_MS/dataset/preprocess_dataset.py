import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def data_rename_merge(path):

    dam = pd.read_csv(os.path.join(path, "Dam dispensing.csv"), low_memory=False)  # 62479 rows x 222 columns
    autoClave = pd.read_csv(os.path.join(path, "Auto clave.csv"), low_memory=False)  # 61052 rows x 26 columns
    fill1 = pd.read_csv(os.path.join(path, "Fill1 dispensing.csv"), low_memory=False)  # 61928 rows x 102 columns
    fill2 = pd.read_csv(os.path.join(path, "Fill2 dispensing.csv"), low_memory=False)  # 62318 rows x 132 columns
    y = pd.read_csv(os.path.join(path, "train_y.csv"), low_memory=False)  # 40506 rows x 2 columns

    # Rename columns
    dam.columns = [i + " - Dam" for i in dam.columns]
    autoClave.columns = [i + " - AutoClave" for i in autoClave.columns]
    fill1.columns = [i + " - Fill1" for i in fill1.columns]
    fill2.columns = [i + " - Fill2" for i in fill2.columns]
    X_Dam = dam.rename(columns={"Set ID - Dam": "Set ID"})
    X_AutoClave = autoClave.rename(columns={"Set ID - AutoClave": "Set ID"})
    X_Fill1 = fill1.rename(columns={"Set ID - Fill1": "Set ID"})
    X_Fill2 = fill2.rename(columns={"Set ID - Fill2": "Set ID"})

    # Merge X
    X = pd.merge(X_Dam, X_AutoClave, on="Set ID")
    X = pd.merge(X, X_Fill1, on="Set ID")
    X = pd.merge(X, X_Fill2, on="Set ID")
    X = X.drop(X[X.duplicated(subset="Set ID")].index).reset_index(drop=True)
    X.to_csv("/storage/mskim/aimers/data/x_merge.csv", index=False) # 57867 rows x 479 columns

    # Merge X and y
    x_y_merge = pd.merge(X, y, "inner", on="Set ID")

    # Drop columns with more than half of the values missing
    drop_cols = []
    for column in x_y_merge.columns:
        if (x_y_merge[column].notnull().sum() // 2) < x_y_merge[column].isnull().sum():
            drop_cols.append(column)
    x_y_merge = x_y_merge.drop(drop_cols, axis=1)

    x_y_merge = x_y_merge.drop("LOT ID - Dam", axis=1)
    x_y_merge.to_csv("/storage/mskim/aimers/data/x_y_merge.csv", index=False) # 40506 rows x 189 columns

def data_preprocessing(path):
    x_y_merge = pd.read_csv(os.path.join(path, "x_y_merge.csv"), low_memory=False)

    for col in x_y_merge.columns:
        unique_value_counts = x_y_merge[col].value_counts()
        print(unique_value_counts)


    for col in x_y_merge.columns:
        if len(x_y_merge[col].unique()) < 10:
            unique_value_counts = x_y_merge[col].value_counts()
            plt.figure(figsize=(12, 8))
            grape = unique_value_counts.plot(kind='bar')

            for p in grape.patches:
                height = p.get_height()
                grape.text(p.get_x() + p.get_width() / 2, height, f'{height}', ha='center', va='bottom')


            plt.title('Value Counts of {}'.format(col))
            plt.xticks(rotation=0)

            plt.savefig('/storage/mskim/aimers/data_graph/{}.png'.format(col))

    print(col)

if __name__ == '__main__':

    path = "/storage/mskim/aimers/data"

    # data_rename_merge(path)
    data_preprocessing(path)

    x_y_merge = pd.read_csv(os.path.join(path, "x_y_merge.csv"), low_memory=False)

    correlation = x_y_merge.corr()

    import seaborn as sns
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation, annot=True, cmap='coolwarm')
    plt.show()