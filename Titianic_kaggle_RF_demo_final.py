#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np


# In[2]:


dir_data = 'C:/Users/Big data/Downloads/titanic'
f_train = os.path.join(dir_data,'train.csv')
f_test =os.path.join(dir_data,'test.csv')


# In[3]:


type(f_train)


# In[4]:


Train = pd.read_csv(f_train)
Test = pd.read_csv(f_test)


# In[5]:


Train.info()


# In[6]:


Train.columns


# In[7]:


#'Survived'為y 另存Train_mod為X的欄位 ,Train['Survived']為Y欄位
Train_mod = Train.drop(labels='Survived' ,axis=1)


# In[8]:


Train.head()


# In[9]:


'''from sklearn.preprocessing import LabelEncoder
LE.fit(Train['Sex'])
Train['Sex']=LE.transform(Train['Sex'])'''


# In[10]:


from sklearn.preprocessing import LabelEncoder

LE = LabelEncoder()
for col in Train_mod:
    #print(Train[col])
    if Train_mod[col].dtype =='object':
        #print(Train[col])
        #print(len(list((Train[col]).unique())))
        #if len(Train[col].unique())<=2:
        if len(list(Train_mod[col].unique())) <=2:
            LE.fit(Train_mod[col])
            Train_mod[col] = LE.transform(Train[col])
            
            

        


# In[11]:


Train_mod.head(10)


# In[12]:


'''from sklearn.preprocessing import OneHotEncoder
OE = OneHotEncoder()
OE.fit(Train_mod['Embarked'])
Train_mod['Embarked'] = OE.transform(Train_mod['Embarked'])'''


# In[13]:


Embarked=pd.get_dummies(Train_mod['Embarked'],prefix='Embarked')
Embarked.head()


# In[14]:


Train_mod1 = pd.concat([Train_mod,Embarked['Embarked_C'],Embarked['Embarked_Q'],Embarked['Embarked_S']],axis=1)
Train_mod1.head()


# In[15]:


Train_mod2 = Train_mod1.drop(labels='Cabin',axis=1)


# In[16]:


Train_mod2.head(10)


# In[17]:


Train_mod2 = Train_mod2.drop(labels='Embarked',axis=1)


# In[18]:


Train_mod2 = Train_mod2.drop(labels='PassengerId',axis=1)


# In[19]:


Train_mod2.head(10)


# In[20]:


Train_describle = Train_mod2.describe()
Train_describle


# In[21]:


mdeian_age = Train_mod2['Age'].median()
mdeian_age


# In[22]:


mode_age = Train_mod2['Age'].mode()
mode_age


# In[20]:


from sklearn.preprocessing import Imputer
#imputer = Imputer('mode')


# In[23]:


#補中位數
Train_mod2['Age'] = Train_mod2['Age'].fillna(28)


# In[24]:


Train_mod2.head(10)


# In[25]:


Train_mod2['Family']=Train_mod2['SibSp']+Train_mod2['Parch']


# In[26]:


Train_mod2.head(10)


# In[27]:


Train_mod2 = Train_mod2.drop(labels='SibSp',axis=1)


# In[28]:


Train_mod2 = Train_mod2.drop(labels='Parch',axis=1)


# In[29]:


Train_mod2 = Train_mod2.drop(labels='Name',axis=1)


# In[30]:


Train_final = Train_mod2.drop(labels='Ticket',axis=1)


# In[31]:


Train_final.info()


# In[32]:


Train.info()


# In[33]:


from sklearn.ensemble import RandomForestClassifier
RF =RandomForestClassifier()


# In[34]:


train_x = Train_final


# In[36]:


train_y = Train['Survived']
len(train_y)


# In[37]:


RF.fit(train_x,train_y)


# In[38]:


Test.info()


# In[39]:


Test_x_predict = Test.drop(labels=['PassengerId','Name','Cabin','Ticket'],axis=1)


# In[42]:


Embarked_Test =pd.get_dummies(Test_x_predict['Embarked'],prefix='Embarked')
Embarked_Test.head()


# In[44]:


Test_x_predict = Test.drop(labels=['Embarked'],axis=1)
Test_x_predict.head()


# In[45]:


Train_mod1 = pd.concat([Train_mod,Embarked['Embarked_C'],Embarked['Embarked_Q'],Embarked['Embarked_S']],axis=1)
Test_x_predict01 = pd.concat([Test_x_predict,Embarked_Test['Embarked_C'],Embarked_Test['Embarked_Q'],Embarked_Test['Embarked_S']],axis=1)


# In[46]:


Test_x_predict01.head()


# In[48]:


Test_x = Test_x_predict01.drop(labels=['PassengerId'],axis=1)
Test_x.head()


# In[91]:


'''Test_x = Test_x.drop(labels=['Name','Cabin'],axis=1)'''


# In[52]:


Test_x = Test_x.drop(labels=['Ticket'],axis=1)


# In[53]:


Test_x.head()


# In[54]:


from sklearn.preprocessing import LabelEncoder

LE = LabelEncoder()
for col in Test_x:
    #print(Train[col])
    if Test_x[col].dtype =='object':
        #print(Train[col])
        #print(len(list((Train[col]).unique())))
        #if len(Train[col].unique())<=2:
        if len(list(Test_x[col].unique())) <=2:
            LE.fit(Test_x[col])
            Test_x[col] = LE.transform(Test_x[col])


# In[55]:


Test_x.head()


# In[63]:


Test_x_median = Test_x.median()
Test_x_median


# In[60]:


Test_x['Age']=Test_x['Age'].fillna(27)


# In[62]:


Test_x['Fare']=Test_x['Fare'].fillna(14.4542)


# In[66]:


Test_x['Family'] = Test_x['SibSp']+Test_x['Parch']


# In[69]:


Test_x=Test_x.drop(labels='SibSp',axis=1)


# In[70]:


Test_x=Test_x.drop(labels='Parch',axis=1)


# In[72]:


Test_y_predict = RF.predict(Test_x)


# In[80]:


df_test_y = pd.DataFrame(data = Test_y_predict,columns=['Survived'])


# In[82]:


df_test_y.info()


# In[84]:


len(Test_x_predict['PassengerId'])


# In[85]:


df_test_y['PassengerId'] = Test_x_predict['PassengerId']


# In[87]:


df_test_y = df_test_y.reindex(columns=['PassengerId','Survived'])


# In[88]:


df_test_y.head()


# In[89]:


import csv
df_test_y.to_csv('Test_Y',columns=['PassengerId','Survived'],index=0)


# In[90]:


import os
os.getcwd()

