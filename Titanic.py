
# coding: utf-8

# In[49]:


import pandas as pd


# In[50]:


import numpy as np


# In[51]:


import os  #在Python中调用os模块


# In[52]:


os.getcwd()  #在Python中使用os.getcwd()方法获取当前工作目录


# In[53]:


data=pd.read_csv('E:/MP/Titanic/data/train.csv') #在绝对路径下搜索文件并导入,注意绝对路径的格式，E:\\MP\是错误的格式


# In[54]:


data.info()


# In[55]:


data['Sex']=data['Sex'].apply(lambda s:1 if s=='male' else 0)


# In[56]:


data=data.fillna(0)


# In[57]:


dataset_X=data[['Sex','Age','Pclass','SibSp','Parch','Fare']]


# In[58]:


dataset_X=dataset_X.as_matrix() #Method .as_matrix will be removed in a future version. Use .values instead.


# In[59]:


data['Deceased']=data['Survived'].apply(lambda s: int(not s))


# In[60]:


dataset_Y=data[['Deceased','Survived']]


# In[61]:


dataset_Y=dataset_Y.as_matrix()


# In[62]:


from sklearn.model_selection import train_test_split


# In[63]:


X_train,X_test,y_train,y_test=train_test_split(dataset_X,dataset_Y,test_size=0.2,random_state=42)


# In[64]:


import tensorflow as tf


# In[65]:


X=tf.placeholder(tf.float32,shape=[None,6])
y=tf.placeholder(tf.float32,shape=[None,2])
W=tf.Variable(tf.random_normal([6,2]),name='weights')
b=tf.Variable(tf.zeros([2]),name='bias')


# In[67]:


y_pred=tf.nn.softmax(tf.matmul(X,W)+b)

