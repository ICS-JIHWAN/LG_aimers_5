import os
from pprint import pprint
import matplotlib.pyplot as plt
import seaborn as sb

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def read_excel_file(file_path: str, header: int = None) -> pd.DataFrame:
    csv_file = file_path.replace(".xlsx", ".csv")

    if not os.path.exists(csv_file):
        print("Converting excel to csv...")
        if header:
            df = pd.read_excel(file_path, header=header)
        else:
            df = pd.read_excel(file_path)

        df.to_csv(csv_file, index=False)
        print(f"  {file_path} -> {csv_file}")
        return df
    else:
        print(f"  Reading {csv_file}")
        return pd.read_csv(csv_file, low_memory=False)



def read_excel_file(file_path: str, header: int = None) -> pd.DataFrame:
    csv_file = file_path.replace(".xlsx", ".csv")

    if not os.path.exists(csv_file):
        print("Converting excel to csv...")
        if header:
            df = pd.read_excel(file_path, header=header)
        else:
            df = pd.read_excel(file_path)

        df.to_csv(csv_file, index=False)
        print(f"  {file_path} -> {csv_file}")
        return df
    else:
        print(f"  Reading {csv_file}")
        return pd.read_csv(csv_file, low_memory=False)


if __name__ == '__main__':
    # ==== Configuration ====
    ROOT_DIR = "/storage/jhchoi/lgaimers_5"
    RANDOM_STATE = 110
    # ========================

    # ==== Read & Make dataset ====
    X_Dam = read_excel_file(
        os.path.join(ROOT_DIR, "data/Dam dispensing.xlsx"), header=1
    )
    X_AutoClave = read_excel_file(
        os.path.join(ROOT_DIR, "data/Auto clave.xlsx"), header=1
    )
    X_Fill1 = read_excel_file(
        os.path.join(ROOT_DIR, "data/Fill1 dispensing.xlsx"), header=1
    )
    X_Fill2 = read_excel_file(
        os.path.join(ROOT_DIR, "data/Fill2 dispensing.xlsx"), header=1
    )
    # y   -> train       [Set ID: train ID, target: ground truth]
    # sub -> submission  [Set ID: test ID,  target: fill with prediction]
    y   = pd.read_csv(os.path.join(ROOT_DIR, "data/train_y.csv"))
    sub = pd.read_csv(os.path.join(ROOT_DIR, "submission.csv"))

    # Rename columns
    X_Dam.columns = [i + " - Dam" for i in X_Dam.columns]
    X_AutoClave.columns = [i + " - AutoClave" for i in X_AutoClave.columns]
    X_Fill1.columns = [i + " - Fill1" for i in X_Fill1.columns]
    X_Fill2.columns = [i + " - Fill2" for i in X_Fill2.columns]
    X_Dam = X_Dam.rename(columns={"Set ID - Dam": "Set ID"})
    X_AutoClave = X_AutoClave.rename(columns={"Set ID - AutoClave": "Set ID"})
    X_Fill1 = X_Fill1.rename(columns={"Set ID - Fill1": "Set ID"})
    X_Fill2 = X_Fill2.rename(columns={"Set ID - Fill2": "Set ID"})

    # Merge X
    X = pd.merge(X_Dam, X_AutoClave, on="Set ID")
    X = pd.merge(X, X_Fill1, on="Set ID")
    X = pd.merge(X, X_Fill2, on="Set ID")
    X = X.drop(X[X.duplicated(subset="Set ID")].index).reset_index(drop=True)
    # =======================

    # ==== Preprocessing ====
    # Drop columns with more than half of the values missing
    drop_cols = []
    for column in X.columns:
        if (X[column].notnull().sum() // 2) < X[column].isnull().sum():
            drop_cols.append(column)
    X = X.drop(drop_cols, axis=1)

    # Additional column drops
    add_drop_cols = []
    add_drop_cols += ["LOT ID - Dam", "LOT ID - AutoClave", "LOT ID - Fill1", "LOT ID - Fill2"]  # Lot ID

    X = X.drop(add_drop_cols, axis=1)

    # Merge X and y -> df_train, df_test
    df_train = pd.merge(X, y, "inner", on="Set ID")
    df_test  = pd.merge(X, sub, "inner", on="Set ID")

    # Label Encoder
    feature_le = LabelEncoder()
    target_le = LabelEncoder()

    features_int = []
    features_str = []
    for col in df_train.columns:
        try:
            df_train[col] = df_train[col].astype(int)
            features_int.append(col)
        except:
            if col == 'target':
                continue
                # df_train[col] = target_le.fit_transform(df_train[col])
            else:
                df_train[col] = feature_le.fit_transform(df_train[col])
                features_str.append(col)

    # Time sorted by collect date of Dam
    df_train = df_train.sort_values(by=["Collect Date - Dam"])
    df_test  = df_test.sort_values(by=["Collect Date - Dam"])

    df_train_X, df_train_y = df_train[features_str + features_int], df_train['target']
    df_test_X = df_test[features_str + features_int]

    # X features after preprocessing
    all_features   = [col for col in df_train_X.columns]
    dam_features   = [col for col in all_features if col[-3:] == "Dam"]
    clave_features = [col for col in all_features if col[-5:] == "Clave"]
    fill1_features = [col for col in all_features if col[-5:] == "Fill1"]
    fill2_features = [col for col in all_features if col[-5:] == "Fill2"]

    # correlation coefficient
    plt.rcParams['figure.figsize'] = (50, 50)
    df_look_heatmap = df_train_X[dam_features[:8]]  # 상관관계를 보고자 하는 data frame
    sb.heatmap(df_look_heatmap.corr(),
               annot_kws={
                   'fontsize': 13,
                   'fontweight': 'bold',
                   'fontfamily': 'serif'
               },
               annot=True,
               cmap='Greens',
               vmin=-1, vmax=1)


