import os
from pprint import pprint

import numpy as np
import pandas as pd
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

if __name__ == '__main__':

    path = "/storage/mskim/aimers/data"
    RANDOM_STATE = 200
    x_merge = pd.read_csv(os.path.join(path, "x_merge.csv"), low_memory=False)
    x_y_merge = pd.read_csv(os.path.join(path, "x_y_merge.csv"), low_memory=False)

    normal_ratio = 1.0  # 1.0 means 1:1 ratio

    df_normal = x_y_merge[x_y_merge["target"] == "Normal"]
    df_abnormal = x_y_merge[x_y_merge["target"] == "AbNormal"]

    num_normal = len(df_normal)
    num_abnormal = len(df_abnormal)
    print(f"  Total: Normal: {num_normal}, AbNormal: {num_abnormal}")

    df_normal = df_normal.sample(n=int(num_abnormal * normal_ratio), replace=False, random_state=RANDOM_STATE)
    df_concat = pd.concat([df_normal, df_abnormal], axis=0).reset_index(drop=True)
    df_concat.value_counts("target")

    df_concat = df_concat.sort_values(by=["Collect Date - Dam"])
    df_train, df_val = train_test_split(df_concat, test_size=0.3, stratify=df_concat["target"], random_state=RANDOM_STATE)


    model = RandomForestClassifier(random_state=RANDOM_STATE)

    features = []

    for col in df_train.columns:
        try:
            df_train[col] = df_train[col].astype(int)
            features.append(col)
        except:
            continue

    if "Set ID" in features:
        features.remove("Set ID")

    train_x = df_train[features]
    train_y = df_train["target"]

    model.fit(train_x, train_y)

    df_test_y = pd.read_csv(os.path.join(path, "submission.csv"))

    df_test = pd.merge(x_merge, df_test_y, "inner", on="Set ID")
    df_test_x = df_test[features]

    for col in df_test_x.columns:
        try:
            df_test_x.loc[:, col] = df_test_x[col].astype(int)
        except:
            continue

    test_pred = model.predict(df_test_x)

    # 제출 데이터 읽어오기 (df_test는 전처리된 데이터가 저장됨)
    df_sub = pd.read_csv("submission.csv")
    df_sub["target"] = test_pred

    # 제출 파일 저장
    df_sub.to_csv("submission.csv", index=False)

    # 제출 데이터 읽어오기 (df_test는 전처리된 데이터가 저장됨)
    df_sub = pd.read_csv("submission.csv")
    df_sub["target"] = test_pred

    # 제출 파일 저장
    df_sub.to_csv("submission.csv", index=False)
