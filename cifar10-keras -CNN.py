#!/usr/bin/env python
# coding: utf-8

# In[1]:


import keras.datasets


# In[2]:


import keras.datasets.cifar10 as ci
import numpy as np


# In[3]:


(x_train, y_train), (x_test, y_test)=ci.load_data()


# In[4]:


import matplotlib.pyplot as plt


# In[5]:


plt.imshow(x_train[2543])


# In[6]:


y_train


# In[7]:


import pandas as pd


# In[8]:


labels=pd.read_csv("cifar10_labels.csv")


# In[9]:


a=labels.columns
a=pd.DataFrame(data=a)
labels=a


# In[10]:


#converting pandas dataframe into list objects
names=[]
for i in labels.values:
    names.append(list(i))
categories=[]
for i in range(len(names)):
    categories.append(names[i][0])


# In[11]:


import keras as ke
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder


# In[12]:


x_train[0].max()


# In[13]:


x_train=x_train/255.0
x_test=x_test/255.0


# In[14]:


#one hot encoding the labels
#oe=OneHotEncoder()
#y_train=oe.fit_transform(y_train).toarray()
#y_test=oe.fit_transform(y_test).toarray()


# In[15]:


x_train.shape


# In[16]:


from sklearn.metrics import accuracy_score


# In[17]:


#importing models

from keras.models import Sequential
from keras.layers import  Dense,Activation,Flatten,Conv2D,MaxPooling2D,Dropout


# In[ ]:


#Building the moodel

model=Sequential()
model.add(Conv2D(64, (3,3) ,input_shape=(32, 32, 3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Activation("relu"))

model.add(Conv2D(filters=64,kernel_size=(3,3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Activation("relu"))



model.add(Flatten())
model.add(Dense(32))
model.add(Activation("relu"))

model.add(Dense(32))
model.add(Activation("relu"))



model.add(Dense(10))
model.add(Activation("sigmoid"))

model.compile(optimizer="adam",
             loss="sparse_categorical_crossentropy",
             metrics=["accuracy"])
model.fit(x_train,y_train,epochs=10,batch_size=60)


# In[22]:


eva=model.evaluate (x_test, y_test)


# In[23]:


eva


# In[37]:


y_predict=model.predict(x_test)
pred=[]
for i in y_predict:
    pred.append(i.argmax())
pred=np.array(pred)


# In[61]:


predictions=[]
for i in range(len(pred)):
    predictions.append(categories[pred[i]])


# In[68]:


i=38
print(predictions[i])
plt.imshow(x_test[i])


# In[ ]:




