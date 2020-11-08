#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from operator import itemgetter
from tpot import TPOTClassifier
from sklearn.metrics import accuracy_score , roc_auc_score ###Importing the required libraries


# In[2]:


data = pd.read_csv('transfusion.data') ##reading the dataset with the help of pandas library using pd.read_csv!!!


# In[3]:


data.head(10)   ###Viewing the first 10 lines of the dataset


# In[4]:


print(data.shape)
data.isnull().sum()   ###dataset doesnot contains any null vlues so we can move to further process


# In[5]:


data.info()   ##Viewing the info of the dataset


# In[6]:


data.describe()


# In[7]:


data.rename(columns = {'whether he/she donated blood in March 2007':'Target'} ,inplace= True)
data.head(2)


# In[8]:


data['Target'].value_counts(normalize=True)*100  ###The Percentage Of the the Target variable


# In[9]:


plt.figure(figsize =(10,5))
sns.barplot(data['Target'] ,data['Frequency (times)'], data =data )


# In[10]:


plt.figure(figsize =(10,5))
sns.barplot(data['Target'] ,data['Time (months)'], data =data )


# In[11]:


X  = data.drop('Target' ,axis =1)  ####Taking X as a independant Variable 
y  = data['Target']    #### Taking y as a dependant variable


# In[12]:


X.head()


# In[13]:


y.head()


# # Dividing the Data Into Train-Test-Split and Using TPOTClassifier for Getting the best Pipeline.

# In[14]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)


# In[15]:


tpot = TPOTClassifier(generations=5,population_size=20,verbosity=2,scoring='roc_auc',
                      random_state=42,disable_update_check=True,config_dict='TPOT light')
tpot.fit(X_train, y_train)                                                                  # AUC score for tpot model

tpot_auc_score = roc_auc_score(y_test, tpot.predict_proba(X_test)[:, 1])
print(f'\nAUC score: {tpot_auc_score:.4f}')

                                                                                            # Print best pipeline steps
print('\nBest pipeline steps:', end='\n')
for idx, (name, transform) in enumerate(tpot.fitted_pipeline_.steps, start=1):
    print(f'{idx}. {transform}')


# In[16]:


X_train_norm , X_test_norm = X_train.copy(), X_test.copy()


# In[17]:


normal_col = X_train_norm['Monetary (c.c. blood)']

for df_ in [X_train_norm, X_test_norm]:
    df_['monetary_log'] = np.log(df_['Monetary (c.c. blood)'])
    df_.drop(columns ='Monetary (c.c. blood)', inplace=True)


# In[18]:


df_.head()


# # Training Our Model With The Logistic Regression

# In[19]:


log =  LogisticRegression(solver='liblinear',random_state=42)

log.fit(X_train_norm, y_train)
print('Training is Succesfully Completed')


# In[20]:


log_auc_score = roc_auc_score(y_test, log.predict_proba(X_test_norm)[:, 1])
print(f'\nAUC score: {log_auc_score:.4f}')


# # Training Our Model With Decision Tree Classifier

# In[21]:


Tree =DecisionTreeClassifier(max_depth=8,min_samples_leaf=13,min_samples_split=10,random_state=42)
Tree.fit(X_train ,y_train)
print("Training is completed with the Decision Tree")


# In[22]:


tree_auc_score = roc_auc_score(y_test, Tree.predict_proba(X_test_norm)[:, 1])
print(f'\nAUC score: {tree_auc_score:.4f}')


# # Conclusion Of the Model

# In[23]:


sorted(
    [('tpot', tpot_auc_score), ('log', log_auc_score)],
    key=itemgetter(1), 
    reverse=False)


# ### The Accuracy of the decision tree is less compared to Logistic Regression
