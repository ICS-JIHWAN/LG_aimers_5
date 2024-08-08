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

    # change code

    from sklearn.preprocessing import LabelEncoder

    target = df_train['target'].values.tolist()
    label_encoder = LabelEncoder()
    train_y = np.array(label_encoder.fit_transform(target))

    import keras
    from keras.models import Sequential
    from keras.layers import Activation, Dropout, BatchNormalization, Input, Dense
    from keras.optimizers import Adam
    from keras import backend as K

    def f1_score(y_true, y_pred):
        y_pred = K.round(y_pred)  # 이진 분류의 경우 y_pred를 0 또는 1로 반올림
        tp = K.sum(K.cast(y_true * y_pred, 'float'))
        fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'))
        fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'))
        precision = tp / (tp + fp + K.epsilon())
        recall = tp / (tp + fn + K.epsilon())
        f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())
        return f1


    model = Sequential()
    model.add(Dense(units=149, activation='relu', input_dim=149))
    model.add(Dropout(0.1))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(units=1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[f1_score])

    model.fit(train_x, train_y, epochs=50, validation_split=0.05)

    ########--------------------------------------########

    # model.fit(train_x, train_y)

    df_test_y = pd.read_csv(os.path.join(path, "submission.csv"))

    df_test = pd.merge(x_merge, df_test_y, "inner", on="Set ID")
    df_test_x = df_test[features]

    for col in df_test_x.columns:
        try:
            df_test_x.loc[:, col] = df_test_x[col].astype(int)
        except:
            continue

    test_pred = model.predict(df_test_x)

    # change code

    test_pred_1 = test_pred.reshape(-1).astype(bool)
    result = ["AbNormal" if x else "Normal" for x in test_pred_1]

    df_sub = pd.read_csv("submission.csv")
    df_sub["target"] = test_pred

    df_sub.to_csv("submission.csv", index=False)

    df_sub = pd.read_csv("submission.csv")
    df_sub["target"] = result

    df_sub.to_csv("submission.csv", index=False)

    ########--------------------------------------########


    # df_sub = pd.read_csv("submission.csv")
    # df_sub["target"] = test_pred
    #
    # df_sub.to_csv("submission.csv", index=False)
    #
    # df_sub = pd.read_csv("submission.csv")
    # df_sub["target"] = test_pred
    #
    # df_sub.to_csv("submission.csv", index=False)