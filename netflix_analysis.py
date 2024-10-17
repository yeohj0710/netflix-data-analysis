import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
from collections import Counter

# 1. 데이터 불러오기
data = pd.read_csv("netflix_titles.csv")

# 데이터 sample 확인
print("Data Sample:")
print(data.head())

# 1-1. EDA: Descriptive 통계
print("\nDescriptive Statistics:")
print(data.describe())

# 범주형 변수의 값 분포 확인
print("\nType Distribution:")
print(data["type"].value_counts())
print("\nRating Distribution:")
print(data["rating"].value_counts())

# 1-2. EDA: 변수 시각화

# 개봉 연도 분포 시각화
sns.histplot(data["release_year"], bins=30, kde=True)
plt.title("Release Year Distribution")
plt.xlabel("Release Year")
plt.ylabel("Number of Titles")
plt.show()

# 장르별 작품 수 계산
data["genre_list"] = data["listed_in"].str.split(", ")

# 장르별 제목 수 세기
genre_counts = Counter()
for genres in data["genre_list"].dropna():
    genre_counts.update(genres)

# 상위 10개 장르
top_genres = genre_counts.most_common(10)
genres, counts = zip(*top_genres)

# 상위 10개 장르 시각화
plt.figure(figsize=(12, 6))
sns.barplot(x=list(counts), y=list(genres))
plt.title("Top 10 Genres on Netflix")
plt.xlabel("Number of Titles")
plt.ylabel("Genre")
plt.tight_layout()
plt.show()

# 1-3. EDA: 상관관계 분석

# 작품 길이를 숫자형으로 변환
data["duration_numeric"] = data["duration"].str.extract("(\d+)").astype(float)

# 개봉 연도와 작품 길이 간의 상관관계 분석
numeric_data = data[["release_year", "duration_numeric"]].dropna()
correlation = numeric_data.corr()

# 상관관계 heatmap 시각화
sns.heatmap(correlation, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# 2. 데이터 전처리 및 학습 데이터 준비

# 사용할 열 선택
features = data[["type", "release_year", "duration_numeric"]]
target = data["rating"]

# 결측치가 있는 행 제거
features = features.dropna()
target = target.loc[features.index]  # 목표 변수와 특징 변수 인덱스 맞추기

# 범주형 변수 인코딩
le_type = LabelEncoder()
features["type"] = le_type.fit_transform(features["type"])

le_rating = LabelEncoder()
target_encoded = le_rating.fit_transform(target)

# 클래스 불균형 처리: 데이터 수가 적은 클래스 제거
target_series = pd.Series(target_encoded, index=features.index)
rating_counts = target_series.value_counts()
common_ratings = rating_counts[rating_counts >= 50].index  # 50개 이상인 클래스 선택
filtered_indices = target_series.isin(common_ratings)

# 필터링된 특징과 목표 변수로 업데이트하고 인덱스 재설정
features = features[filtered_indices].reset_index(drop=True)
target = target_series[filtered_indices].reset_index(drop=True)

# 학습용 데이터와 테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.25, stratify=target
)

# 3. 모델 학습
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 4. 모델 평가 및 성능 분석
y_pred = model.predict(X_test)

# Confusion Matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Accuracy
print("\nAccuracy:", accuracy_score(y_test, y_pred))

# Precision
print(
    "Precision:", precision_score(y_test, y_pred, average="weighted", zero_division=1)
)

# Recall
print("Recall:", recall_score(y_test, y_pred, average="weighted", zero_division=1))

# F1-Score
print("F1-Score:", f1_score(y_test, y_pred, average="weighted", zero_division=1))

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=1))
