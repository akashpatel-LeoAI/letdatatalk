#!/usr/bin/env python
# coding: utf-8

# In[24]:


#canada_per_capita_income.csv

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model


# In[25]:


df = pd.read_csv("canada_per_capita_income.csv")
df = df.rename(columns = {'per capita income (US$)' : 'per_capita_income'})
df.head()


# In[26]:


df.dtypes


# In[27]:


get_ipython().run_line_magic('matplotlib', 'inline')

plt.xlabel('Year')
plt.ylabel('Per Capita Income (US$)')
plt.scatter(df.year, df.per_capita_income, color='red', marker= "*")


# In[37]:


model_reg = linear_model.LinearRegression()
model_reg.fit(df[['year']], df.per_capita_income)
income_2025_prediction = reg.predict([[2025]])

print(income_2025_prediction)

