# Email Spam Detection

## Description
This project implements a machine learning-based **email spam detection system** using various classification models, including **Naive Bayes, Logistic Regression, Random Forest, and Support Vector Machine (SVM)**.  
The dataset contains emails labeled as **"spam" or "ham" (non-spam)**, and the system aims to classify them accordingly.

---

## Prerequisites
Before running the code, ensure you have the following installed:

### **1. Python 3.x**
### **2. Required Python Libraries:**
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `wordcloud`
- `scikit-learn`
- `scipy`

You can install the necessary libraries using:
```bash
pip install pandas numpy matplotlib seaborn wordcloud scikit-learn scipy
```

---

## Dataset
The dataset is in **CSV format** with the name `spam_ham_dataset3.csv`.  
It contains the following columns:

- `Unnamed: 0` → An index column (removed during preprocessing).
- `label` → The label for the email (**spam** or **ham**).
- `text` → The content of the email.
- `label_num` → The numerical encoding (**spam: 1, ham: 0**).

---

## Steps to Run the Code

### **1. Load the Dataset**
```python
import pandas as pd

data = pd.read_csv("spam_ham_dataset3.csv")
```

### **2. Data Preprocessing**
- Remove unwanted columns and handle missing values.
- Clean text by removing non-alphabetic characters and converting all text to lowercase.
- Create new features like `email_length` (length of each email).

```python
import re
data['text'] = data['text'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x.lower()))  # Text cleaning
```

---

## **3. Exploratory Data Analysis (EDA)**
The following visualizations are created:

- **Spam vs Ham Distribution** using `countplot`
- **Email Length Analysis** using `histplot`
- **Word Clouds** for spam and ham emails

```python
import seaborn as sns

sns.countplot(x='label', data=data)
sns.histplot(data=data, x="email_length", hue="label", multiple="stack")
```

---

## **4. Feature Extraction and Scaling**
- Convert text into numerical features using **TfidfVectorizer**.
- Scale `label_num` using **MinMaxScaler**.
- Combine both features using **scipy.sparse.hstack**.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import hstack

vectorizer = TfidfVectorizer()
X_text = vectorizer.fit_transform(data['text'])

scaler = MinMaxScaler()
X_numeric = scaler.fit_transform(data[['label_num']])

X_combined = hstack((X_text, X_numeric))
```

---

## **5. Train-Test Split**
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_combined, data['label_num'], test_size=0.2, random_state=42)
```

---

## **6. Model Training and Evaluation**
### **Trained Models:**
- **Naive Bayes** → `MultinomialNB`
- **Logistic Regression** → `LogisticRegression`
- **Random Forest** → `RandomForestClassifier`
- **Support Vector Machine (SVM)** → `SVC`

### **Hyperparameter Tuning & Evaluation**
```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV

nb_model = MultinomialNB()
nb_grid = GridSearchCV(nb_model, param_grid={'alpha': [0.1, 1, 10]}, scoring='f1_weighted', cv=5, n_jobs=-1)
```

---

## **7. Model Performance Metrics**
After training and prediction, the following evaluations are performed:

```python
from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test, y_pred_nb))
print(confusion_matrix(y_test, y_pred_nb))
```

### **Performance Metrics:**
- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score** (Primary Metric)

---

## **8. Data Visualization**
### **The project includes the following visualizations:**
- **Spam vs Ham Distribution** → A bar plot showing spam vs ham proportions.
- **Email Length Distribution** → A histogram comparing email lengths.
- **Word Clouds** → Visualizing most frequent words in spam vs ham emails.

---

## **Next Steps**
- **Address Class Imbalance:** Use techniques like oversampling/undersampling.
- **Deploy the Model:** Consider deploying for real-time spam detection.

---

## **Output**
After running the code, you should see:

1. **Spam vs Ham Distribution** (count plot).
2. **Email Length Analysis** (histogram).
3. **Word Clouds** (spam vs ham word frequency).
4. **Model Evaluation** (classification reports & confusion matrices).

---

## 📩 **Contact**
For any queries or improvements, feel free to reach out!

---
