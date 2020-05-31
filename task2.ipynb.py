#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
get_ipython().run_line_magic('matplotlib', 'inline')

#reading the data into pandas
student_math = pd.read_csv('C:/Users/rl893/Documents/personal info/priya_assgn/task2/student-math.csv',sep = ';',quotechar = '"')


# In[2]:


#printing head
student_math.head()


# In[3]:


#printing table information (variables,datatypes)
student_math.info()


# In[4]:


#storing catagorical columns
cat_cols = student_math.select_dtypes(include=['object']).columns


# In[5]:


cat_cols


# In[6]:



#onehotencoding of categorical columns
encoder=OneHotEncoder(sparse=False)
df_encoded = pd.DataFrame()
for i in cat_cols:
    df=pd.DataFrame()
    df = pd.DataFrame (encoder.fit_transform(student_math[[i]]))
    df.columns = encoder.get_feature_names([i])
    df_encoded = pd.concat([df_encoded, df], axis=1)


# In[7]:


#concatenating original data with onehotencoded data
student_math_oneHot = pd.concat([student_math, df_encoded], axis=1)


# In[8]:


#Deleting categorical columns
for i in cat_cols:
    del student_math_oneHot[i]


# In[9]:


#printing the statistics of the columns
student_math_oneHot.describe()


# In[10]:


student_math_oneHot.info()


# In[11]:


#creating target column
student_math_oneHot["final_grade"] = student_math_oneHot.apply(lambda x: x.G1 + x.G2 + x.G3, axis=1)


# In[12]:


#preparing exogenous and endogenous values
X = student_math_oneHot.drop(["final_grade","G3"],axis=1).values
y =student_math_oneHot["final_grade"].values


# In[13]:


#Splitting data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[14]:


#fitting linear regression model
regressor = LinearRegression()  
regressor.fit(X_train, y_train)


# In[15]:


#Accuracy of the model
print('Accuracy of linear regression on test set: {:.2f}'.format(regressor.score(X_test, y_test)))


# In[16]:


#predicting target values
y_pred = regressor.predict(X_test)


# In[17]:


#Model evaluation metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[18]:


#dataframe of actual value & predicted value
df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
df


# In[19]:


#bar chart of actual with predicted

df1 = df.head(25)
df1.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()


# In[20]:


#scatter plot of actual with predicted
df.plot(x='Actual', y='Predicted', style='o')  
plt.title('Actual vs Predicted')  
plt.xlabel('Actual')  
plt.ylabel('Predicted')  
plt.show()


# In[21]:


# building the optimal model using backward elimination
# SL = 0.05 and eliminating those features which have p-value > SL
import statsmodels.regression.linear_model as sm 
X = np.append(arr = np.ones((395,1)).astype(int), values = X, axis = 1)


# In[22]:


sigLevel = 0.05
X_opt = X[:,np.arange(0,X.shape[1]).tolist()]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
pVals = regressor_OLS.pvalues
regressor_OLS.summary()


# In[23]:


while pVals[np.argmax(pVals)] > sigLevel:
     X_opt = np.delete(X_opt, np.argmax(pVals), axis = 1)    
     print("pval of dim removed: " + str(np.argmax(pVals)))
     print(str(X_opt.shape[1]) + " dimensions remaining...")
     regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
     pVals = regressor_OLS.pvalues


# In[24]:


regressor_OLS.summary()


# In[25]:


# Optimized training and test sets
y_opt = y
X_train, X_test, y_train, y_test = train_test_split(X_opt, y_opt, test_size=0.2, random_state=0)


# In[26]:


#fitting linear regression model
regressor = LinearRegression()  
regressor.fit(X_train, y_train)


# In[27]:


#Accuracy of the model
print('Accuracy of linear regression on test set: {:.2f}'.format(regressor.score(X_test, y_test)))


# In[28]:


#predicting target values
y_pred = regressor.predict(X_test)


# In[29]:


#Model evaluation metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[30]:


#dataframe of actual value & predicted value
df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
df


# In[31]:


#barplot of actual with predcited
df1 = df.head(25)
df1.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()


# In[32]:


#scatter plot of actual with predicted
df.plot(x='Actual', y='Predicted', style='o')  
plt.title('Actual vs Predicted')  
plt.xlabel('Actual')  
plt.ylabel('Predicted')  
plt.show()


# In[ ]:




