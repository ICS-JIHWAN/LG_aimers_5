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


ROOT_DIR = "/storage/jhchoi/lgaimers_5/data"
RANDOM_STATE = 110

X_Dam = read_excel_file(os.path.join(ROOT_DIR, "Dam dispensing.xlsx"), header=1)

X_AutoClave = read_excel_file(
    os.path.join(ROOT_DIR, "Auto clave.xlsx"), header=1
)

X_Fill1 = read_excel_file(
    os.path.join(ROOT_DIR, "Fill1 dispensing.xlsx"), header=1
)

X_Fill2 = read_excel_file(
    os.path.join(ROOT_DIR, "Fill2 dispensing.xlsx"), header=1
)

y = pd.read_csv(os.path.join(ROOT_DIR, "train_y.csv"))

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

# Merge X and y
df_merged = pd.merge(X, y, "inner", on="Set ID")

# Drop columns with more than half of the values missing
drop_cols = []
for column in df_merged.columns:
    if (df_merged[column].notnull().sum() // 2) < df_merged[
        column
    ].isnull().sum():
        drop_cols.append(column)
df_merged = df_merged.drop(drop_cols, axis=1)

# Drop Lot ID
df_merged = df_merged.drop("LOT ID - Dam", axis=1)
df_merged = df_merged.drop("LOT ID - AutoClave", axis=1)
df_merged = df_merged.drop("LOT ID - Fill1", axis=1)
df_merged = df_merged.drop("LOT ID - Fill2", axis=1)

features = df_merged.columns

features_dam = []  # 79
features_calve = []  # 21
features_fill1 = []  # 39
features_fill2 = []  # 49
features_other = []  # 2
for f in features:
    if f[-3:] == 'Dam':  # Dam features
        features_dam.append(f)
    elif f[-9:] == 'AutoClave':  # AutoClave features
        features_calve.append(f)
    elif f[-5:] == 'Fill1':  # Fill1 features
        features_fill1.append(f)
    elif f[-5:] == 'Fill2':  # Fill2 features
        features_fill2.append(f)
    else:  # Target and...
        features_other.append(f)

feature_idx = 9
print(
    f"{features_dam[feature_idx]} // {features_calve[feature_idx]} // {features_fill1[feature_idx]} // {features_fill2[feature_idx]}")

# Column : model.suffix
fs = ['Model.Suffix - Dam', 'Model.Suffix - AutoClave', 'Model.Suffix - Fill1', 'Model.Suffix - Fill2']
df_melted = df_merged[fs].melt(var_name='Categories', value_name='Value')
value_counts = df_melted.value_counts().reset_index(name='Frequency')

fig, axs = plt.subplots(4, 1, figsize=(10, 15))
for i, col in enumerate(fs):
    df = value_counts[value_counts['Categories'] == col]
    axs[i].bar(df['Value'], df['Frequency'])

    for j, value in enumerate(df['Frequency'].values):
        axs[i].text(value_counts.index[j], value, str(value), ha='center', va='bottom')
plt.show()
plt.close()

# Label Encoder
le = LabelEncoder()

features_int = []
features_str = []
for col in df_merged.columns:
    try:
        df_merged[col] = df_merged[col].astype(int)
        features_int.append(col)
    except:
        df_merged[col] = le.fit_transform(df_merged[col])
        features_str.append(col)

# correlation coefficient
plt.rcParams['figure.figsize'] = (50, 50)
sb.heatmap(df_merged.corr(),
           annot_kws={
               'fontsize': 13,
               'fontweight': 'bold',
               'fontfamily': 'serif'
           },
           annot=True,
           cmap='Greens',
           vmin=-1, vmax=1)
