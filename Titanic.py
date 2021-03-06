
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import tensorflow as tf #构建计算图中用到简写tf
import os #在Python中调用os模块


# In[2]:


####################################################################################################
##################                        数据读入及预处理                     ########################
####################################################################################################
#C++的注释，通过//或/* */
#MATLAB的注释，通过%
#Verilog的注释，通过//
#python的注释，通过#


# In[3]:


os.getcwd() #在Python中使用os.getcwd()方法获取当前工作目录
data=pd.read_csv('E:/MP/Titanic/data/train.csv') #在绝对路径下搜索文件并导入,注意绝对路径的格式，E:\\MP\是错误的格式


# In[4]:


data.info() #通过DataFrame.info()的方法查看数据的概况


# In[5]:


#------------------------------将样本的标签转换成独热编码(one-hot encoding)--------------------------------
data['Sex']=data['Sex'].apply(lambda s:1 if s=='male' else 0) #将特征值正规化
data=data.fillna(0)#将缺失的字段补零
dataset_X=data[['Sex','Age','Pclass','SibSp','Parch','Fare']] #确定逻辑回归：y'=softmax(xW+b)的x的特征
                                                              #x输入向量，是大小为d×1的列向量，d是特征数
dataset_X=dataset_X.as_matrix() #存疑，窃以为：这是将确定的特征矩阵化，使之能进行计算图运算
data['Deceased']=data['Survived'].apply(lambda s: int(not s)) #取'Survived'为非
dataset_Y=data[['Deceased','Survived']] #新增'Deceased'字段表示第二种分类的标签
dataset_Y=dataset_Y.as_matrix() #存疑，窃以为：这是将确定的特征矩阵化，使之能进行计算图运算


# In[6]:


#-----------------------------------随机打乱数据后按比例拆分数据集-----------------------------------------
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(dataset_X,dataset_Y,test_size=0.2,random_state=42)
#使用sklearn的train_test_split函数将标记数据切分为“训练数据集和验证数据集”
#将标记数据切分后，验证数据占20%，由test_size=0.2告诉程序


# In[7]:


#######################################################################################################
##################                      以下是构建计算图的过程                     ########################
#######################################################################################################


# In[8]:


X=tf.placeholder(tf.float32,shape=[None,6]) #placeholder声明占位符
y=tf.placeholder(tf.float32,shape=[None,2]) #placeholder声明占位符
W=tf.Variable(tf.random_normal([6,2]),name='weights')
b=tf.Variable(tf.zeros([2]),name='bias')


# In[9]:


#TensorFlow的Feed机制：程序不会直接交互执行，而是在声明过程中只做计算图的构建
#placeholder声明一个数据占据的位置，真正运算时，用数据替换placeholder，占位符可看成一个算子
#占位符三参数：元素类型(dtype)、维度形状(shape)、占位符名称标识(name)


# In[10]:


#-----------------------------------------构建前向传播计算图--------------------------------------------
y_pred=tf.nn.softmax(tf.matmul(X,W)+b) 
#y'=softmax(xW+b)
#b偏置列向量，x输入列向量，W权重矩阵（tensor可看成边的权值参数）
#x：d×1，d是特征数
#W：c×d，c是分类类别数目
#b：c×1


# In[11]:


cross_entropy=-tf.reduce_sum(y*tf.log(y_pred+1e-10),reduction_indices=1) 
                                                    #声明代价函数
                                                    #使用交叉熵作为代价函数：交叉熵就是用来判定实际的输出与期望的输出的接近程度！
cost=tf.reduce_mean(cross_entropy)  #批量样本的代价值，是所有样本交叉熵的平均值
train_op=tf.train.GradientDescentOptimizer(0.001).minimize(cost)
#加入优化算法：使用随机梯度下降法优化器最小化代价，系统自动构建反向传播部分的计算图


# In[12]:


#####################################################################################################
##################                        构建训练过程并执行                    ########################
#####################################################################################################


# In[16]:


with tf.Session() as sess:
    tf.global_variables_initializer().run()
    #初始化变量
    #-----------------------------------以下是训练迭代，迭代10轮----------------------------------------
    for epoch in range(10):
        total_loss=0.
        for i in range(len(X_train)):
            feed={X:[X_train[i]],y:[y_train[i]]}
            _,loss=sess.run([train_op,cost],feed_dict=feed)#通过session.run接口触发执行
            total_loss+=loss
        print('Epoch:%04d,total loss=%.9f'%(epoch+1,total_loss))
    print('Training complete!') #python2.X中的打印是：print 'Training complete!'
    #------------------------------------------------------------------------------------------------
    #-----------------------------------评估校验数据集上的准确率-----------------------------------------
    pred=sess.run(y_pred,feed_dict={X:X_test})
    correct=np.equal(np.argmax(pred,1),np.argmax(y_test,1))
    accuracy=np.mean(correct.astype(np.float32))
    print("Accuracy on Validation set:%.9f"%accuracy)
    #注：这个模块不能分开，否则，with将无法将Session作为上下文管理器(context manager)！
    #   分开则出现"Attempted to use a closed Session."错误
    #   同时，也通过with以后语句的缩进，控制作用域。退出作用域，资源释放


# In[17]:


####################################################################################################
##################                     存储和加载模型参数(略)                  ########################
####################################################################################################


# In[18]:


####################################################################################################
##################                       预测测试数据结果                      ########################
####################################################################################################


# In[21]:


#-------------------------------------读入测试数据集并完成预处理----------------------------------------
testdata=pd.read_csv('E:/MP/Titanic/data/test.csv')
testdata=testdata.fillna(0)
testdata['Sex']=testdata['Sex'].apply(lambda s:1 if s=='male' else 0)
X_test=testdata[['Sex','Age','Pclass','SibSp','Parch','Parch','Fare']]


# In[23]:


#----------------------------------------开启session进行预测------------------------------------------
with tf.Session() as sess:
    saver.restore(sess,'model.ckpt') #加载模型存档
predictions=np.argmax(sess.run(y_pred,feed_dicgt-{X:X_test}),1)

