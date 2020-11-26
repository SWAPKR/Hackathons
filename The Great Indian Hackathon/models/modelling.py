#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, Reshape, Embedding,Concatenate


import pandas as pd 
from keras.models import Model
from keras.layers import Input, Dense, Concatenate, Reshape, Dropout
from keras.layers.embeddings import Embedding
import numpy as np
from sklearn.model_selection import StratifiedKFold
from  sklearn.utils import shuffle 
from sklearn.model_selection import train_test_split
from keras.models import model_from_json

from keras.models import load_model
from keras.callbacks import ModelCheckpoint
import keras.objectives
from keras.callbacks import CSVLogger


# In[ ]:


import pandas as pd
from sklearn import preprocessing
# read training data
train = pd.read_csv("../input/train_upd.csv")
test = pd.read_csv("../input/test_upd.csv")


# In[29]:


train.head()


# In[30]:


# concatenate both training and test data
data = pd.concat([train, test]).reset_index(drop=True)
cat_col=[feat for feat in train.columns if feat not in ['UnitPrice']]


# In[31]:


# for feat in cat_col:
#     if feat !='Quantity':
#         data.loc[data[feat].map(data[feat].value_counts())<5,feat]='rare'


# In[32]:


# for feat in cat_col:
#     print(data[feat].value_counts())


# In[33]:


for feat in cat_col:
    lbl_enc = preprocessing.LabelEncoder()
    temp_col = data[feat].fillna("NONE").astype(str).values
    data.loc[:, feat] = lbl_enc.fit_transform(temp_col)
train2= data[data.UnitPrice != -100].reset_index(drop=True)
test2 = data[data.UnitPrice == -100].reset_index(drop=True)


# In[34]:


# from sklearn.model_selection import KFold
# train2['kfold']=-1

# kf=KFold(n_splits=10,shuffle=True,random_state=42)
# for fold,(trn,vld) in enumerate(kf.split(train2)):
#     train2.loc[vld,'kfold']=fold


# In[35]:


train2.head()


# In[36]:


# def kfold(fold):
#     train_f=train2[train2['kfold']!=fold]
#     valid_f=train2[train2['kfold']==fold]
#     x_train=train_f.drop('UnitPrice',axis=1)
#     y_train=train_f['UnitPrice']
#     x_valid=valid_f.drop('UnitPrice',axis=1)
#     y_valid=valid_f['UnitPrice']
#     x_train.drop('kfold',axis=1,inplace=True)
#     x_valid.drop('kfold',axis=1,inplace=True)
#     return x_train,y_train,x_valid,y_valid
# # x_train,y_train,x_valid,y_valid=kfold(0)


# In[37]:


# x_train,y_train,x_valid,y_valid=kfold(0)
# x_train.head()


# In[38]:


data.Quantity.nunique()


# In[39]:


x_train=train2.drop('UnitPrice',axis=1)
y_train=train2['UnitPrice']


# In[40]:


# fold=0
# x_train,y_train,x_valid,y_valid=kfold(fold)

x_trn=[]
for feat in cat_col:
    x_trn.append(x_train[feat].values)
    
# x_vld=[]
# for feat in cat_col:
#     x_vld.append(x_valid[feat].values)
    
x_tst=[]
for feat in cat_col:
    x_tst.append(test2[feat].values)  
    
inp_size={}
emb_size={}
for feat in cat_col:
    inp_size[feat]=data[feat].nunique()
    
for feat in cat_col:
    emb_size[feat]=min(50,data[feat].nunique()//2)


# In[41]:


emb_size


# In[42]:


tensorboard_cb = keras.callbacks.TensorBoard(
    log_dir='tensorboard!',
    histogram_freq=1,
    write_graph=True,
    write_images=True
)
csv_logger = CSVLogger('history/loss!.csv', append=True, separator=',')


# In[44]:


inputs = []
embeddings = []


for feat in cat_col:
    input1 = Input(shape=(1,))
    embedding = Embedding(inp_size[feat], emb_size[feat], input_length=1)(input1)
    embedding = Reshape(target_shape=(emb_size[feat],))(embedding)
    inputs.append(input1)
    embeddings.append(embedding)


x = Concatenate()(embeddings)
# x = Dense(1024, activation='relu')(x)
# x = Dropout(.35)(x)
x = Dense(512, activation='relu')(x)
x = Dropout(.35)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(.15)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(.15)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(.15)(x)
x = Dense(64, activation='relu')(x)
x = Dropout(.15)(x)
x = Dense(16, activation='relu')(x)
x = Dropout(.15)(x)
output = Dense(1, activation='linear')(x)
model = Model(inputs, output)
# model = Model(inputs, output)


from keras import backend as K

# def root_mean_squared_error(y_true, y_pred):
#         return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 

def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))




def lr_scheduler(epoch, lr):
        if (epoch+1)%5==0:
                return lr * 0.4
        return lr


callback = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)



opt=keras.optimizers.Adam(lr=1e-3)


# opt=keras.optimizers.RMSprop(lr=4e-3,decay=0.95,momentum=0.9, epsilon=1e-8, name="RMSprop")

model.compile(loss = root_mean_squared_error,optimizer=opt , metrics=['mean_absolute_error'])
# model.compile(loss ='mean_absolute_error' ,optimizer='adam', metrics=['mean_absolute_error',root_mean_squared_error])
print(model.summary())



cp1= ModelCheckpoint(filepath="finalswap/save_best.h5", monitor='loss',save_best_only=True,verbose=1,mode='min',save_weights_only=True)
cp2= ModelCheckpoint(filepath='finalswap/save_all.h5', monitor='loss',save_best_only=False ,verbose=2,save_weights_only=True)

callbacks_list = [callback,cp1,cp2,csv_logger]






# model.fit(X_train,y_train, batch_size =1024, epochs = 1000, validation_split = 0.2,validation_data=(xval, yval))
# model.fit(x,y_train.values, batch_size =1024, epochs = 10,validation_data=(x_v,y_valid.values),shuffle=True,callbacks=[callbacks_list])
# model.fit(x_trn,y_train.values, batch_size =2048, epochs = 25 ,validation_data=(x_vld,y_valid.values))
model.fit(x_trn,y_train.values, batch_size =2048, epochs = 15,callbacks=callbacks_list)
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")


# In[45]:


# sub=model.predict(x_tst)
model.predict(x_tst)


# In[151]:


sub=pd.Series(sub.reshape(1,-1)[0],name='UnitPrice')


# In[ ]:


sub.var()


# In[153]:


sub.to_csv('submission.csv',index=False)


# In[130]:


json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)


# In[132]:


loaded_model.load_weights('model.h5')
# loaded_model.load_weights('finalswap/save_best.h5')


# In[133]:


loaded_model.predict(x_tst)


# In[ ]:





# In[ ]:




