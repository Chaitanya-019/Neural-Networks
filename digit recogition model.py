#!/usr/bin/env python
# coding: utf-8

# In[60]:


import tensorflow as tf

mnist=tf.keras.datasets.mnist #28*28 datatsets


# In[63]:


(x_train,y_train),(x_test,y_test)=mnist.load_data()


# In[78]:


import matplotlib.pyplot as plt

plt.imshow(x_train[1],cmap=plt.cm.binary)


# In[74]:


x_train[0].max()


# In[79]:


x_train=tf.keras.utils.normalize(x_train, axis=1)
x_test=tf.keras.utils.normalize(x_test, axis=1)


# In[81]:


x_train.max()


# In[93]:


model=tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer="adam",
             loss="sparse_categorical_crossentropy",
             metrics=["accuracy"])
model.fit(x_train,y_train,epochs=150)


# In[94]:


val_loss,val_acc=model.evaluate(x_test,y_test)
val_loss,val_acc


# In[95]:


model.save("digit_master.model")


# In[ ]:





# In[ ]:




