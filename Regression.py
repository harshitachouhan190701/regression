#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


companies = pd.read_csv("C:\\Users\\harsh\\Documents\\Python stuff\\1000_Companies.csv")
companies


# In[7]:


X = companies.iloc[:, :-1].values
y = companies.iloc[:, 4].values


# In[8]:


X


# In[9]:


y


# In[10]:


companies.head()


# In[11]:


companies.info()


# In[12]:


companies.isnull().sum()


# In[13]:


sns.heatmap(companies.corr())


# In[14]:


companies.head()


# In[15]:


# Encoding categorical data


# In[17]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Assuming X is your input data

# Use LabelEncoder for encoding categorical variable
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])

# Define the column transformer with OneHotEncoder for column 3
preprocessor = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(), [3])  # Apply OneHotEncoder to column 3
    ],
    remainder='passthrough'  # Keep the other columns as they are
)

# Apply the transformations
X = preprocessor.fit_transform(X)


# In[24]:


print(X)


# In[27]:


print(y)


# In[20]:


X = X[:, 1:]


# In[21]:


X


# In[33]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[34]:


from sklearn.linear_model import LinearRegression


# In[37]:


regressor = LinearRegression()
regressor.fit(X_train, y_train)
model_fit = LinearRegression()
model_fit.fit(X_train, y_train)


# In[39]:


y_pred = regressor.predict(X_test)
print(y_pred)


# In[41]:


print(regressor.coef_)


# In[42]:


print(regressor.intercept_)


# In[43]:


from sklearn.metrics import r2_score


# In[44]:


r2_score(y_test, y_pred)


# In[ ]:




