#!/usr/bin/env python
# coding: utf-8

# In[12]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

from sklearn.model_selection import train_test_split


# In[3]:


# ccdf = Credit_card_Data_frame
ccdf = pd.read_csv('creditcard_2023.csv')
ccdf.head(10)


# In[4]:


ccdf.info()


# In[5]:


ccdf.isnull().sum()


# In[6]:


sns.countplot(ccdf['Class'])
plt.title("Class Distribution")


# In[7]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
ccdf['Amount'] = scaler.fit_transform(ccdf['Amount'].values.reshape(-1, 1))

ccdf.head(5)


# In[8]:


X = ccdf.drop(['Class','id'], axis=1)
y = ccdf['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[9]:


random_forest_model = RandomForestClassifier()

random_forest_model.fit(X_train, y_train)


# In[11]:


# Make predictions
y_pred = random_forest_model.predict(X_test)

# Print classification report
print(classification_report(y_test, y_pred))

# Compute ROC curve and AUC
y_prob = random_forest_model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = roc_auc_score(y_test, y_prob)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


# In[ ]:




