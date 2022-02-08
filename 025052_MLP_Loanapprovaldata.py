#!/usr/bin/env python
# coding: utf-8

# In[46]:


#importing packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

pd.options.display.max_columns =None
pd.options.display.max_rows =None


# In[9]:


df=pd.read_csv("E:\FSM\TERMS\TERM 3\MLP\loanapprovaldataset.csv")


# In[10]:


df.head()


# In[11]:


df.tail()


# In[12]:


df.info()


# In[13]:


df.columns


# In[14]:


df.shape


# In[15]:


df.dtypes


# In[16]:


df.nunique()


# In[17]:


df.describe()


# In[18]:


df.sample(n=10)


# In[19]:


df.isnull().sum()


# In[20]:


df.Property_Area.unique()


# In[21]:


num_atr=['ApplicantIncome' , 'CoapplicantIncome' , 'LoanAmount' , 'Loan_Amount_Term']

cat_atr=['Gender' , 'Married' , 'Dependents' , 'Education' , 'Self_Employed' , 'Credit_History' , 'Property_Area']


# In[24]:


df[num_atr].hist(bins=40, figsize=(20,15)) 
plt.show()


# In[25]:


df[df.duplicated(keep = 'last')] 


# In[26]:


df.head()


# In[27]:


from sklearn.model_selection import train_test_split
X= df.drop(['Loan_Status'], axis=1)
y= df['Loan_Status']


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)


# In[28]:


atr=X_train.select_dtypes('number').columns


# In[29]:


atr


# In[30]:


from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error,r2_score 
from sklearn.preprocessing import OneHotEncoder as onehot


# In[31]:


from sklearn.preprocessing import LabelEncoder


# In[32]:


le = LabelEncoder()
X_train[cat_atr] = X_train[cat_atr].apply(le.fit_transform)


# In[33]:


X_train[cat_atr].head()


# In[34]:


df.head()


# In[35]:


df.info()


# In[47]:


df.loc[df['Loan_Status'] =='Y', 'Loan_approval'] = 1
df.loc[df['Loan_Status'] =='N', 'Loan_approval'] = 0
df.loc[df['Education'] =='Graduate', 'Edu'] = 1
df.loc[df['Education'] =='Not Graduate', 'Edu'] = 0
df.loc[df['Self_Employed'] =='No', 'Self_Emp'] = 1
df.loc[df['Self_Employed'] =='Yes', 'Self_Emp'] = 0
df.head()
df.tail()


# In[48]:


#Making a new column based on summation of salaries of applicants

df['Total Income'] = (df['ApplicantIncome'] + df['CoapplicantIncome'])
df.head()

##Handling the missing data with 0 value
df = df.fillna(0)


# In[49]:


x= df.iloc[:, [14,15,16]].values  
y= df.iloc[:, 13].values 


# In[50]:


x_train, x_test, y_train, y_test= train_test_split(x, y, random_state=0)  


# In[51]:


#feature Scaling  
st_x= StandardScaler()    
x_train= st_x.fit_transform(x_train)    
x_test= st_x.transform(x_test)  
y_test


# In[52]:


knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn.fit(x_train, y_train)


# In[53]:


y_pred = knn.predict(x_test)


# In[54]:


confusion_matrix(y_test, y_pred)


# In[55]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)


# In[56]:


from sklearn.metrics import precision_recall_fscore_support
precision_recall_fscore_support(y_test, y_pred)


# In[57]:


from sklearn.metrics import precision_score
precision_score(y_test, y_pred)


# In[58]:


from sklearn.metrics import recall_score
recall_score(y_test, y_pred)


# In[59]:


from sklearn.metrics import f1_score
f1_score(y_test, y_pred)


# In[60]:


error_rate = []
for i in range(1,40):
 knn = KNeighborsClassifier(n_neighbors=i)
 knn.fit(x_train,y_train)
 pred_i = knn.predict(x_test)
 #print (pred_i)
 #print (1-accuracy_score(y_test, pred_i))
 error_rate.append(1-accuracy_score(y_test, pred_i))

plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', 
         marker='o',markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
plt.show()
print("Minimum error:-",min(error_rate),"at K =",error_rate.index(min(error_rate))+1)


# In[61]:


knn = KNeighborsClassifier(n_neighbors=17, metric='euclidean')
knn.fit(x_train, y_train)


# In[62]:


y_pred = knn.predict(x_test)
accuracy_score(y_test, y_pred)


# In[ ]:




