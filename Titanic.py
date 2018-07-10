
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


import numpy as np


# In[3]:


import os


# In[ ]:


#在Python中调用os模块


# In[4]:


os.getcwd()


# In[ ]:


#在Python中使用os.getcwd()方法获取当前工作目录


# In[5]:


data=pd.read_csv('E:/MP/Titanic/data/train.csv')


# In[ ]:


#在绝对路径下搜索文件并导入,注意绝对路径的格式，E:\\MP\是错误的格式


# In[6]:


data.info()


# In[ ]:


#通过DataFrame.info()的方法查看数据的概况


# In[ ]:


#以下是将样本的标签转换成独热编码(one-hot encoding)


# In[7]:


data['Sex']=data['Sex'].apply(lambda s:1 if s=='male' else 0)


# In[ ]:


#将特征值正规化


# In[8]:


data=data.fillna(0)


# In[ ]:


#将缺失的字段补零


# In[9]:


dataset_X=data[['Sex','Age','Pclass','SibSp','Parch','Fare']]


# In[ ]:


#确定逻辑回归：y'=softmax(xW+b)的x的特征
#x输入向量，是大小为d×1的列向量，d是特征数


# In[10]:


dataset_X=dataset_X.as_matrix() #Method .as_matrix will be removed in a future version. Use .values instead.


# In[ ]:


#存疑，窃以为：这是将确定的特征矩阵化，使之能进行计算图运算


# In[11]:


data['Deceased']=data['Survived'].apply(lambda s: int(not s))


# In[ ]:


#取'Survived'为非


# In[12]:


dataset_Y=data[['Deceased','Survived']]


# In[ ]:


#新增'Deceased'字段表示第二种分类的标签


# In[13]:


dataset_Y=dataset_Y.as_matrix()


# In[ ]:


#存疑，窃以为：这是将确定的特征矩阵化，使之能进行计算图运算


# In[ ]:


#以下是随机打乱数据后按比例拆分数据集


# In[14]:


from sklearn.model_selection import train_test_split


# In[15]:


X_train,X_test,y_train,y_test=train_test_split(dataset_X,dataset_Y,test_size=0.2,random_state=42)


# In[ ]:


#使用sklearn的train_test_split函数将标记数据切分为“训练数据集和验证数据集”


# In[ ]:


#将标记数据切分后，验证数据占20%，由test_size=0.2告诉程序


# In[ ]:


#以下是构建计算图的过程


# In[16]:


import tensorflow as tf


# In[17]:


X=tf.placeholder(tf.float32,shape=[None,6])
y=tf.placeholder(tf.float32,shape=[None,2])
W=tf.Variable(tf.random_normal([6,2]),name='weights')
b=tf.Variable(tf.zeros([2]),name='bias')


# In[ ]:


#placeholder声明占位符


# In[ ]:


#TensorFlow的Feed机制：程序不会直接交互执行，而是在声明过程中只做计算图的构建


# In[ ]:


#占位符三参数：元素类型(dtype)、维度形状(shape)、占位符名称标识(name)


# In[ ]:


#以下内容是构建前向传播计算图


# In[18]:


y_pred=tf.nn.softmax(tf.matmul(X,W)+b)

