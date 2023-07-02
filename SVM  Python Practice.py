#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
from sklearn.svm import SVC


# In[8]:


df = load_breast_cancer()
type(df)


# In[13]:


df['data']


# In[14]:


len(df['data'])


# In[15]:


X = df.data[:, :2]
y = df.target


# In[16]:


plt.scatter(X[:, 0], X[:, 1],
            c=y,
            s=20, edgecolors="k")
plt.show()


# In[17]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)


# In[18]:


svm = SVC(kernel="rbf", gamma=0.5, C=1.0)
svm.fit(X_train, y_train)
 


# In[19]:


y_pred= svm.predict(X_test)


# In[20]:


from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_pred)


# In[ ]:




