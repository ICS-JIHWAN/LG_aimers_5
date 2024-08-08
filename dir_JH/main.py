import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

ROOT_DIR = "data"
RANDOM_STATE = 1004

# data read
train_path = '/storage/jhchoi/lgaimers_5/data/train.csv'
test_path = '/storage/jhchoi/lgaimers_5/data/test.csv'

df_train = pd.read_csv(train_path)
print(f'Train 총 데이터 개수  : {len(df_train)}')
print(f'Train 총 feature 개수 : {len(df_train.columns)}')
print(df_train.head())

# test의 경우 Set ID column 제거 필수 !!
df_test = pd.read_csv(test_path)
print(f'Test 총 데이터 개수  : {len(df_test)}')
print(f'Test 총 feature 개수 : {len(df_test.columns)}')
print(df_test.head())

# Columns Drop
# 결측치 columns 제거
drop_cols = []
for column in df_train.columns:
    if (df_train[column].notnull().sum() // 2) < df_train[
        column
    ].isnull().sum():
        drop_cols.append(column)

# 동일한 값이 들어있는 데이터 columns 제거
# 이상하게 및에 있는 columns를 지우면 점수가 낮게 나옴 어떤 columns인지 모르겠음
# [Wip Line, Insp. Seq No., Insp Judge Code]
# drop_cols += ["Wip Line_Dam", "Wip Line_Fill1", "Wip Line_Fill2", "Wip Line_AutoClave"]
# drop_cols += ["Insp. Seq No._Dam", "Insp. Seq No._Fill1", "Insp. Seq No._Fill2", "Insp. Seq No._AutoClave"]
# drop_cols += ["Insp Judge Code_Dam", "Insp Judge Code_Fill1", "Insp Judge Code_Fill2", "Insp Judge Code_AutoClave"]
# drop_cols += ["HEAD NORMAL COORDINATE X AXIS(Stage1) Collect Result_Dam", "HEAD NORMAL COORDINATE X AXIS(Stage1) Collect Result_Fill1", "HEAD NORMAL COORDINATE X AXIS(Stage1) Collect Result_Fill2"]  # 결측치 존재 및
# drop_cols += ["1st Pressure Judge Value_AutoClave", "2nd Pressure Judge Value_AutoClave", "3rd Pressure Judge Value_AutoClave"] # 다 동일한 값이기 때문에

df_train = df_train.drop(drop_cols, axis=1)
print(df_train.head())

# normal에서 Outlier 제거
dam_normal_outlier = [
    "Dispense Volume(Stage1) Collect Result_Dam",
    "Stage1 Circle2 Distance Speed Collect Result_Dam", "Stage1 Circle3 Distance Speed Collect Result_Dam",
    "Stage1 Circle4 Distance Speed Collect Result_Dam",
    "Stage1 Line1 Distance Speed Collect Result_Dam", "Stage1 Line2 Distance Speed Collect Result_Dam",
    "Stage1 Line3 Distance Speed Collect Result_Dam", "Stage1 Line4 Distance Speed Collect Result_Dam",
    "Stage3 Circle2 Distance Speed Collect Result_Dam", "Stage3 Circle3 Distance Speed Collect Result_Dam",
    "Stage3 Circle4 Distance Speed Collect Result_Dam",
    "Stage3 Line1 Distance Speed Collect Result_Dam", "Stage3 Line2 Distance Speed Collect Result_Dam",
    "Stage3 Line3 Distance Speed Collect Result_Dam", "Stage3 Line4 Distance Speed Collect Result_Dam"
]

df_filtered = df_train.copy()
filtered_index = []
for col in dam_normal_outlier:
    df_drop_outlier = df_train[df_train["target"] == "Normal"]

    Q1 = df_drop_outlier[col].quantile(0.25)
    Q3 = df_drop_outlier[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    filtered_index += (
        df_drop_outlier[(df_drop_outlier[col] < lower_bound) | (df_drop_outlier[col] > upper_bound)].index).tolist()

df_filtered = df_filtered.drop(list(set(filtered_index)))
print(f"총 제거된 데이터 개수 : {len(list(set(filtered_index)))}")
df_train = df_filtered.copy()

# Target label 개수 같게
normal_ratio = 1.0  # 1.0 means 1:1 ratio

df_normal = df_train[df_train["target"] == "Normal"]
df_abnormal = df_train[df_train["target"] == "AbNormal"]

num_normal = len(df_normal)
num_abnormal = len(df_abnormal)
print(f"  Total: Normal: {num_normal}, AbNormal: {num_abnormal}")

df_normal = df_normal.sample(n=int(num_abnormal * normal_ratio), replace=False, random_state=RANDOM_STATE)
df_concat = pd.concat([df_normal, df_abnormal], axis=0).reset_index(drop=True)
df_concat.value_counts("target")

df_train, df_val = train_test_split(
    df_concat,
    test_size=0.3,
    stratify=df_concat["target"],
    random_state=RANDOM_STATE,
)


def print_stats(df: pd.DataFrame):
    num_normal = len(df[df["target"] == "Normal"])
    num_abnormal = len(df[df["target"] == "AbNormal"])

    print(f"  Total: Normal: {num_normal}, AbNormal: {num_abnormal}" + f" ratio: {num_abnormal / num_normal}")


# Print statistics
print(f"  \tAbnormal\tNormal")
print_stats(df_train)
print_stats(df_val)

model = RandomForestClassifier(random_state=RANDOM_STATE)

features = []
for col in df_train.columns:
    try:
        df_train[col] = df_train[col].astype(int)
        features.append(col)
    except:
        continue

train_x = df_train[features]
train_y = df_train["target"]

model.fit(train_x, train_y)

val_x = df_val[features]
val_y = df_val["target"]

pred_y = model.predict(val_x)

# F1 스코어 계산
f1 = f1_score(val_y, pred_y, pos_label='Normal')

print("F1 스코어:", f1)

df_test_x = df_test[features]
print(df_test_x)

test_pred = model.predict(df_test_x)
print(test_pred)

# 제출 데이터 읽어오기 (df_test는 전처리된 데이터가 저장됨)
df_sub = pd.read_csv("/storage/jhchoi/lgaimers_5/data/submission.csv")
df_sub["target"] = test_pred

# 제출 파일 저장
df_sub.to_csv("/storage/jhchoi/lgaimers_5/submission.csv", index=False)