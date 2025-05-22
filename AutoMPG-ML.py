import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# بارگذاری دیتاست از UCI با استفاده از sep='\s+' و رشته خام
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin', 'car_name']
df = pd.read_csv(url, sep=r'\s+', names=columns)  
# نمایش 5 نمونه اول دیتاست
print("نمونه‌های اول دیتاست:")
print(df.head())

# پیش‌پردازش داده‌ها
# تبدیل مقادیر نادرست در ستون horsepower به NaN
df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')
# پر کردن مقادیر NaN در ستون horsepower بدون استفاده از inplace=True
df['horsepower'] = df['horsepower'].fillna(df['horsepower'].mean())  
# تقسیم داده‌ها به ویژگی‌ها (X) و هدف (y)
X = df.drop(['mpg', 'car_name'], axis=1)  # حذف هدف و نام خودرو
y = df['mpg']  # هدف

# تقسیم داده‌ها به آموزش و تست (80% آموزش و 20% تست)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ساخت مدل رگرسیون خطی
model = LinearRegression()

# آموزش مدل
model.fit(X_train, y_train)

# پیش‌بینی با داده‌های تست
y_pred = model.predict(X_test)

# ارزیابی مدل
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'\nMean Squared Error: {mse}')
print(f'R² Score: {r2}')

# تجزیه و تحلیل ویژگی‌ها
# ماتریس همبستگی
plt.figure(figsize=(10, 6))
sns.heatmap(X.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

# نمودارهای پراکندگی
sns.pairplot(df[['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration']])
plt.show()
