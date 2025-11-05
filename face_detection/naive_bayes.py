import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import LabelEncoder

# Khởi tạo dữ liệu
data = {
    'age': [
        '<=30', '<=30', '31...40', '>40', '>40', '>40', '31...40',
        '<=30', '<=30', '>40', '<=30', '31...40', '31...40', '>40'
    ],
    'income': [
        'high', 'high', 'high', 'medium', 'low', 'low', 'low',
        'medium', 'low', 'medium', 'medium', 'medium', 'high', 'medium'
    ],
    'student': [
        'no', 'no', 'no', 'no', 'yes', 'yes', 'yes',
        'no', 'yes', 'yes', 'yes', 'yes', 'no', 'no'
    ],
    'credit_rating': [
        'fair', 'excellent', 'fair', 'fair', 'fair', 'excellent', 'excellent',
        'fair', 'fair', 'fair', 'excellent', 'excellent', 'fair', 'excellent'
    ],
    'buys_computer': [
        'no', 'no', 'yes', 'yes', 'yes', 'no', 'yes',
        'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'no'
    ]
}
df = pd.DataFrame(data)

# Mã hóa dữ liệu
encoders = {col: LabelEncoder() for col in df.columns}

for col in df.columns:
    df[f"{col}_enc"] = encoders[col].fit_transform(df[col])

X = df[[col for col in df.columns if col.endswith('_enc') and col != 'buys_computer_enc']]
y = df['buys_computer_enc']

# Huấn luyện mô hình Naive Bayes
model = CategoricalNB()
model.fit(X, y)

# Dự đoán với mẫu mới
sample = pd.DataFrame({
    'age': ['<=30'],
    'income': ['medium'],
    'student': ['yes'],
    'credit_rating': ['fair']
})

# Mã hóa đúng cột, tránh bị "_enc_enc"
cols_to_encode = sample.columns.values.tolist()
for col in cols_to_encode:
    sample[f"{col}_enc"] = encoders[col].transform(sample[col])

X_sample = sample[[f"{col}_enc" for col in cols_to_encode]]
y_sample_pred = model.predict(X_sample)
predicted_label = encoders['buys_computer'].inverse_transform(y_sample_pred)

print("🎯 Dự đoán cho mẫu mới:", predicted_label[0])

y_pred = model.predict(X)

# Hiển thị Confusion Matrix
cm = confusion_matrix(y, y_pred, labels=model.classes_)
labels = encoders['buys_computer'].inverse_transform(model.classes_)

fig, ax = plt.subplots(figsize=(6, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap=plt.cm.Blues, ax=ax)
plt.title("Confusion Matrix with Labels")
plt.tight_layout()
plt.show()
