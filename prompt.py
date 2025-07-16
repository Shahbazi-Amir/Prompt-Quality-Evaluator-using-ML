import pandas as pd

df = pd.read_csv('prompt_data.csv')
from collections import Counter

# Load the dataset
df = pd.read_csv('prompt_data.csv')

# Function to count repeated words in a response
def count_repeated_words(response):
    words = response.lower().split()
    word_counts = Counter(words)
    repeated = sum(count - 1 for word, count in word_counts.items() if count > 1)
    return repeated

# Add the new column 'repeated_words'
df['repeated_words'] = df['response'].apply(count_repeated_words)

# Save the updated dataset
df.to_csv('prompt_data_updated.csv', index=False)

# Display first 10 rows to verify
print(df[['prompt', 'response', 'length', 'prompt_response_sim', 'repeated_words']].head(10))
import pandas as pd
import matplotlib.pyplot as plt

# Load the updated dataset
df = pd.read_csv('prompt_data_updated.csv')

# Calculate mean of features for each quality_label
mean_features = df.groupby('quality_label')[['length', 'prompt_response_sim', 'repeated_words']].mean()
print("Mean of features by quality_label:")
print(mean_features)

# Plot bar chart for mean features
mean_features.plot(kind='bar', figsize=(10, 6))
plt.title('Mean Feature Values by Quality Label')
plt.xlabel('Quality Label (0=Bad, 1=Average, 2=Good)')
plt.ylabel('Mean Value')
plt.xticks(rotation=0)
plt.legend(['Length', 'Prompt-Response Similarity', 'Repeated Words'])
plt.tight_layout()
plt.show()
from sklearn.model_selection import train_test_split

X = df[['length', 'prompt_response_sim', 'repeated_words']]
y = df['quality_label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)


from sklearn.metrics import classification_report

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))


import matplotlib.pyplot as plt

feature_importances = model.feature_importances_
plt.bar(X.columns, feature_importances)
plt.title("Feature Importances")
plt.show()
import matplotlib.pyplot as plt

feature_importances = model.feature_importances_
plt.bar(X.columns, feature_importances)
plt.title("Feature Importances")
plt.show()


from sklearn.metrics import confusion_matrix
import pandas as pd

# فرض بر اینکه y_test و y_pred آماده هستن
cm = confusion_matrix(y_test, y_pred)

# تبدیل به DataFrame برای دیدن راحت‌تر
labels = [0, 1, 2]  # لیبل‌های کیفیت: بد، متوسط، خوب
cm_df = pd.DataFrame(cm, index=[f'True {i}' for i in labels],
                        columns=[f'Pred {i}' for i in labels])

print(cm_df)




from sklearn.metrics import confusion_matrix
import pandas as pd

# فرض بر اینکه y_test و y_pred آماده هستن
cm = confusion_matrix(y_test, y_pred)

# تبدیل به DataFrame برای دیدن راحت‌تر
labels = [0, 1, 2]  # لیبل‌های کیفیت: بد، متوسط، خوب
cm_df = pd.DataFrame(cm, index=[f'True {i}' for i in labels],
                        columns=[f'Pred {i}' for i in labels])

print(cm_df)


from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# فرض: df دیتا فریم نهایی با ویژگی‌ها و لیبل‌هاست

X = df[['length', 'prompt_response_sim', 'repeated_words']]
y = df['quality_label']

# تقسیم‌بندی
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ساخت مدل SVM با کرنل RBF (غیرفقط خطی)
model = SVC(kernel='rbf', C=1, gamma='scale')  # kernel='linear' هم میشه تست کرد

# آموزش مدل
model.fit(X_train, y_train)

# پیش‌بینی و ارزیابی
y_pred = model.predict(X_test)

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# فرض: df دیتا فریم نهایی با ویژگی‌ها و لیبل‌هاست

X = df[['length', 'prompt_response_sim', 'repeated_words']]
y = df['quality_label']

# تقسیم‌بندی
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ساخت مدل SVM با کرنل RBF (غیرفقط خطی)
model = SVC(kernel='rbf', C=1, gamma='scale')  # kernel='linear' هم میشه تست کرد

# آموزش مدل
model.fit(X_train, y_train)

# پیش‌بینی و ارزیابی
y_pred = model.predict(X_test)

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# ساخت کانفیوژن
cm = confusion_matrix(y_test, y_pred)
labels = [0, 1, 2]

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - SVM')
plt.show()
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# مدل رگرسیون
reg = LinearRegression()
reg.fit(X_train, y_train)

# پیش‌بینی
y_pred_reg = reg.predict(X_test)

# نتایج
print("MSE:", mean_squared_error(y_test, y_pred_reg))
print("R^2 Score:", r2_score(y_test, y_pred_reg))



import matplotlib.pyplot as plt
import seaborn as sns

# تنظیمات ساده نمودار
plt.figure(figsize=(8, 4))

# نمودار واقعی در برابر پیش‌بینی
sns.lineplot(x=range(len(y_test)), y=y_test, label="Actual", color='blue')
sns.lineplot(x=range(len(y_pred_reg)), y=y_pred_reg, label="Predicted", color='red')

plt.title("Actual vs Predicted (Linear Regression)")
plt.xlabel("Sample")
plt.ylabel("Target")
plt.legend()
plt.show()


import numpy as np

# فرض: y_test و y_pred_reg اعداد اعشاری هستند
# می‌خوایم اونا رو گِرد کنیم و به دسته‌های گسسته تبدیل کنیم
y_test_discrete = np.round(y_test).astype(int)
y_pred_discrete = np.round(y_pred_reg).astype(int)

# حالا می‌تونی کانفیوژن ماتریس بزنی
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test_discrete, y_pred_discrete)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix (Discretized Regression)")
plt.show()

