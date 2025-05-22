# نصب کتابخانه‌های لازم
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import requests
from io import StringIO

# دانلود دیتاست از URL
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/19/car+evaluation/car.data"

# دریافت فایل از URL
response = requests.get(url)
data = StringIO(response.text)

# نام ستون‌ها
columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']

# بارگذاری دیتاست
df = pd.read_csv(data, names=columns)

# نمایش 5 نمونه اول
print("نمونه‌های اول دیتاست:")
print(df.head())

# تبدیل ویژگی‌های دسته‌ای به متغیرهای عددی
encoder = LabelEncoder()
for column in df.columns:
    df[column] = encoder.fit_transform(df[column])

# نمایش تغییرات
print("\nدیتاست پس از Label Encoding:")
print(df.head())

# تحلیل ویژگی‌ها - آمار توصیفی
print("\nآمار توصیفی دیتاست:")
print(df.describe())

# رسم باکس پلات برای شناسایی داده‌های پرت
plt.figure(figsize=(10, 6))
sns.boxplot(data=df)
plt.xticks(rotation=45)
plt.title('Boxplot for Feature Analysis')
plt.show()

# محاسبه ماتریس همبستگی
correlation_matrix = df.corr()

# نمایش ماتریس همبستگی
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

# تقسیم داده‌ها به ویژگی‌ها (X) و هدف (y)
X = df.drop('class', axis=1)
y = df['class']

# انجام PCA برای کاهش ابعاد
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# نمایش نتایج PCA
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
plt.title('PCA for Dimensionality Reduction')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# تقسیم داده‌ها به مجموعه‌های آموزشی و تست
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# مدل 1: Logistic Regression
log_reg_model = LogisticRegression()
log_reg_model.fit(X_train, y_train)

# پیش‌بینی با مدل Logistic Regression
y_pred_log_reg = log_reg_model.predict(X_test)

# ارزیابی مدل Logistic Regression
print("\nLogistic Regression Evaluation:")
print(f'Accuracy: {accuracy_score(y_test, y_pred_log_reg)}')
print(classification_report(y_test, y_pred_log_reg))

# مدل 2: Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# پیش‌بینی با مدل Random Forest
y_pred_rf = rf_model.predict(X_test)

# ارزیابی مدل Random Forest
print("\nRandom Forest Evaluation:")
print(f'Accuracy: {accuracy_score(y_test, y_pred_rf)}')
print(classification_report(y_test, y_pred_rf))