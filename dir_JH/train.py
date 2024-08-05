import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from catboost import CatBoostClassifier


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
    ROOT_DIR = "/storage/jhchoi/lgaimers_5"
    RANDOM_STATE = 110

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

    # Merge X and y -> df_train, df_test
    df_train = pd.merge(X, y, "inner", on="Set ID")
    df_test  = pd.merge(X, sub, "inner", on="Set ID")

    # ==== Column drop ====
    # Drop columns with more than half of the values missing
    drop_cols = []
    for column in df_train.columns:
        if (df_train[column].notnull().sum() // 2) < df_train[column].isnull().sum():
            drop_cols.append(column)
    df_train = df_train.drop(drop_cols, axis=1)
    df_test  = df_test.drop(drop_cols, axis=1)

    # Additional column drops
    add_drop_cols = []
    add_drop_cols += ["LOT ID - Dam", "LOT ID - AutoClave", "LOT ID - Fill1", "LOT ID - Fill2"]  # Lot ID

    df_train = df_train.drop(add_drop_cols, axis=1)
    df_test  = df_test.drop(add_drop_cols, axis=1)

    # 시간 정렬
    df_merged = df_merged.sort_values(by=["Collect Date - Dam"])

    # label encoder
    le = LabelEncoder()
    target_le = LabelEncoder()

    features_int = []
    features_str = []
    for col in df_merged.columns:
        try:
            df_merged[col] = df_merged[col].astype(int)
            features_int.append(col)
        except:
            if col == 'target':
                df_merged[col] = target_le.fit_transform(df_merged[col])
            else:
                df_merged[col] = le.fit_transform(df_merged[col])
                features_str.append(col)

    # validation
    # df_train, df_val = train_test_split(
    #     df_merged,
    #     test_size=0.3,
    #     stratify=df_merged["target"],
    #     random_state=RANDOM_STATE,
    # )

    df_merged_X, df_merged_y = df_merged[features_str + features_int], df_merged['target']
    label_mapping = dict(zip(le.classes_, range(len(le.classes_))))

    # class weights
    unique, counts = np.unique(df_merged_y, return_counts=True)
    class_weights = {0: 1.0, 1: float(counts[0]) / counts[1]}
    print(f'Class weights: {class_weights}')

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=43)

    for train_index, test_index in sss.split(df_merged_X, df_merged_y):
        train_x, val_x = df_merged_X.iloc[train_index], df_merged_X.iloc[test_index]
        train_y, val_y = df_merged_y.iloc[train_index], df_merged_y.iloc[test_index]

        # model & train
        model = CatBoostClassifier(
            iterations=100,
            learning_rate=0.1,
            depth=6,
            loss_function='Logloss',  # 이진 분류를 위한 로그 손실 함수 (Cross Entropy Loss)
            class_weights=class_weights,
            random_seed=42,
        )
        # model = RandomForestClassifier(random_state=RANDOM_STATE)

        train_x = train_x.sort_values(by=["Collect Date - Dam"])
        model.fit(train_x, train_y)

        # test
        val_x = val_x.sort_values(by=["Collect Date - Dam"])
        pred_y = model.predict(val_x)

        f1 = f1_score(val_y, pred_y, average='weighted')
        print(f'F1 Score: {f1:.2f}')


        df_test_X, submission = df_test[features_str + features_int], df_test['target']
        for col in df_test_X.columns:
            try:
                df_test_X[col] = df_test_X[col].astype(int)
            except:
                df_test_X[col] = le.fit_transform(df_test_X[col])

        df_test_X = df_test_X.sort_values(by=["Collect Date - Dam"])
        test_pred = model.predict(df_test_X)

        df_sub = pd.read_csv(os.path.join('/storage/jhchoi/lgaimers_5', "submission.csv"))
        df_sub['target'] = target_le.inverse_transform(test_pred)

        df_sub.to_csv('/storage/jhchoi/lgaimers_5/sub/submission.csv', index=False)

        df_sub = pd.read_csv(os.path.join('/storage/jhchoi/lgaimers_5/sub', "submission.csv"))
