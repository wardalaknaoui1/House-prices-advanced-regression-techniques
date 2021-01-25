#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression


# # Train data cleaning and prep

# In[2]:


# Importing the train data.

housing_train=pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')


# In[3]:


# Displaying data head.

housing_train.head()


# In[4]:


# Displaying data shape.

housing_train.shape


# In[5]:


# Describe my numeric data (Statistics)

numerical_columns = [col for col in housing_train.columns if (housing_train[col].dtype=='int64' or housing_train[col].dtype=='float64')]
housing_train[numerical_columns].describe().loc[['mean', 'std', 'min', '25%', '50%', '75%', 'max'], :]


# In[6]:


# Display correlation matrix to check if multicolinearity exist (correlation between idependent
# and dependent variables.)

plt.figure(figsize=(30,20))
corrMatrix = housing_train.corr()
sns.heatmap(corrMatrix, annot=True, cmap='coolwarm')
### There are some variables with high correlation(Multicolinearity issue), We will deal with this issue later


# #### Avoidng multicolinearity
# * "Garage area" variable is highly correlated with GarageCars.That makes sense; as the area of the garage gets bigger, the number of cars that fit in gets bigger.
# we will drop GarageArea(It has less correlation with the target variable.)
# *  "Garage built" and "year built" have high correlation. We will drop "GarageYrBlt."
# * "1stFlrSF" has a high correlation with "TotalBsmtSF", and both have equal correlation with 
# "Price". Therefore, dropping either one would be fine. I decided to drop  "1stFlrSF".
# * "GarageYrBlt" and "YearBuilt" are highly correlated, We will drop "GarageYrBlt" since it has less correlattion with the target variable.

# In[7]:


# Visualizing the missing data (Heat map)

sns.heatmap(housing_train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[8]:


# Train data variables, values and type.

housing_train.info()


# ### Dropping few variables
# * Id: is not relevant.
# * "Alley", "PoolQC", "Fence", "Miscfeature" have more than 50% missing data, I decided to drop them.

# In[9]:


housing_train.drop(['Id'],axis=1,inplace=True)
housing_train.drop(['Alley'],axis=1,inplace=True)
housing_train.drop(['PoolQC','Fence','MiscFeature', 'GarageArea'],axis=1,inplace=True)
housing_train.drop(['GarageYrBlt'],axis=1,inplace=True)
housing_train.drop(['1stFlrSF'],axis=1,inplace=True)


# #### Imputing missing data: numeric using mean and categorical using mode.

# In[10]:


## Fill Missing Values for numeric variables.
housing_train['LotFrontage']=housing_train['LotFrontage'].fillna(housing_train['LotFrontage'].mean())


# In[11]:


# Fill Missing Values for categorical variables.
housing_train['BsmtCond']=housing_train['BsmtCond'].fillna(housing_train['BsmtCond'].mode()[0])
housing_train['BsmtQual']=housing_train['BsmtQual'].fillna(housing_train['BsmtQual'].mode()[0])
housing_train['FireplaceQu']=housing_train['FireplaceQu'].fillna(housing_train['FireplaceQu'].mode()[0])
housing_train['GarageType']=housing_train['GarageType'].fillna(housing_train['GarageType'].mode()[0])
housing_train['GarageFinish']=housing_train['GarageFinish'].fillna(housing_train['GarageFinish'].mode()[0])
housing_train['GarageQual']=housing_train['GarageQual'].fillna(housing_train['GarageQual'].mode()[0])
housing_train['GarageCond']=housing_train['GarageCond'].fillna(housing_train['GarageCond'].mode()[0])
housing_train['MasVnrType']=housing_train['MasVnrType'].fillna(housing_train['MasVnrType'].mode()[0])
housing_train['MasVnrArea']=housing_train['MasVnrArea'].fillna(housing_train['MasVnrArea'].mode()[0])
housing_train['BsmtFinType2']=housing_train['BsmtFinType2'].fillna(housing_train['BsmtFinType2'].mode()[0])
housing_train['BsmtExposure']=housing_train['BsmtExposure'].fillna(housing_train['BsmtExposure'].mode()[0])


# #### We still have 38 missing values

# In[12]:


# Total missing values

housing_train.isnull().sum().sum()


# In[13]:


# Only 38 missing variables remained, I will drop them.

housing_train.dropna(inplace=True)
housing_train.isna().sum().sum()


# In[14]:


## Making sure there is no missing data (heatmap.)

sns.heatmap(housing_train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# # Test data: cleaning and prep
# 

# In[15]:


# Importing my test data

housing_test= pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')


# In[16]:


# Saving the Id variable for later.

Id= housing_test['Id']


# In[17]:


# Displaying first 5 rows of the test data.

housing_test.head(5)


# In[18]:


# Test data shape.

housing_test.shape


# In[19]:


# Displaying test data information ( variables, value counts, variable type)

housing_test.info()


# In[20]:


# Displaying the total missing data

housing_test.isna().sum().sum()


# In[21]:


# Visualizing the missing data

sns.heatmap(housing_test.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[22]:


#These variables have more than 50% missing values that is why I have decided to drop them.

housing_test.drop(['Id', 'Alley', 'MiscFeature', 'Fence', 'PoolQC', 'GarageYrBlt', 'GarageArea','1stFlrSF'], axis=1, inplace=True)


# In[23]:


# Imputing the missing data in numeric variables( mean).

housing_test.LotFrontage.fillna(housing_test.LotFrontage.mean(), inplace=True)


# In[24]:


# Impute the missing data with the mode.

housing_test.MSZoning .fillna(housing_test.MSZoning.mode()[0], inplace= True)
housing_test.Utilities.fillna(housing_test.Utilities.mode()[0], inplace= True)
housing_test.MasVnrType.fillna(housing_test.MasVnrType.mode()[0], inplace= True)
housing_test.BsmtQual.fillna(housing_test.BsmtQual.mode()[0], inplace= True)
housing_test.BsmtCond.fillna(housing_test.BsmtCond.mode()[0], inplace= True)
housing_test.BsmtExposure.fillna(housing_test.BsmtExposure.mode()[0], inplace= True)
housing_test.BsmtFinType1.fillna(housing_test.BsmtFinType1.mode()[0], inplace= True)
housing_test.BsmtFinType2.fillna(housing_test.BsmtFinType2.mode()[0], inplace= True)
housing_test.FireplaceQu.fillna(housing_test.FireplaceQu.mode()[0], inplace= True)
housing_test.GarageType.fillna(housing_test.GarageType.mode()[0], inplace= True)
housing_test.GarageFinish.fillna(housing_test.GarageFinish.mode()[0], inplace= True) 
housing_test.GarageQual.fillna(housing_test.GarageQual.mode()[0], inplace= True)
housing_test.GarageCond.fillna(housing_test.GarageCond.mode()[0], inplace= True)
housing_test['GarageCars'].fillna(housing_test['GarageCars'].mean())
housing_test['BsmtUnfSF'].fillna(housing_test['BsmtUnfSF'].mean())
housing_test['BsmtFinSF1'].fillna(housing_test['BsmtFinSF1'].mean())
housing_test['BsmtFinSF2'].fillna(housing_test['BsmtFinSF2'].mean())


# In[25]:


# Test data shape.

housing_test.shape


# In[26]:


# Exporting the clean test data into CSV file.

housing_test.to_csv('housing_testclean.csv', index=False)


# # Converting categorical variables into dummies and # Concatinating train and test data

# In[27]:


# These are all the categorical variables.

columns=['MSZoning','Street','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood',
         'Condition2','BldgType','Condition1','HouseStyle','SaleType',
        'SaleCondition','ExterCond',
         'ExterQual','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',
        'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','Heating','HeatingQC',
         'CentralAir',
         'Electrical','KitchenQual','Functional',
         'FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive']


# In[28]:


# Defining a function that will be used to convert categorical data into dummies.
def category_onehot_multcols(multcolumns):
    df_final=final_df
    i=0
    for fields in multcolumns:
        
        print(fields)
        df1=pd.get_dummies(final_df[fields],drop_first=True)
        
        final_df.drop([fields],axis=1,inplace=True)
        if i==0:
            df_final=df1.copy()
        else:
            
            df_final=pd.concat([df_final,df1],axis=1)
        i=i+1
       
        
    df_final=pd.concat([final_df,df_final],axis=1)
        
    return df_final


# In[29]:


# Importing the clean test data.

test_df=pd.read_csv('housing_testclean.csv')


# In[30]:


# "Clean" Test data shape.

test_df.shape


# In[31]:


# Displaying first 5 rows from the test data

test_df.head()


# In[32]:


# Concating test and train data

final_df=pd.concat([housing_train,test_df],axis=0, sort=False)


# In[33]:


# Displaying the shape of the final data

final_df.shape


# In[34]:


# Using the function to convert categorical data into dummies

final_df=category_onehot_multcols(columns)


# In[35]:


# Final data shape

final_df.shape


# In[36]:


# Eliminating the duplicates.

final_df =final_df.loc[:,~final_df.columns.duplicated()]


# In[37]:


# Data shape after eliminating duplicates.

final_df.shape


# In[38]:


final_df.head()


# # Splitting the data into train and test data

# In[39]:


# Train, test data split.

df_Train=final_df.iloc[:1422,:]
df_Test=final_df.iloc[1422:,:]


# In[40]:


# Dependent variable (Test data)

y_test= df_Test['SalePrice']


# In[41]:


df_Test.drop(['SalePrice'],axis=1,inplace=True)
# You can ignore this warning. SettingWithCopyWarning happens when indexing a DataFrame returns a reference to the initial DataFrame.


# In[42]:


X_train=df_Train.drop(['SalePrice'],axis=1)
y_train=df_Train['SalePrice']


# In[43]:


X_train=df_Train.drop(['SalePrice'],axis=1)
y_train=df_Train['SalePrice']


# In[44]:


# X_train, y_train, test shape.
print(X_train.shape)
print(y_train.shape)
print(df_Test.shape)


# # Model building

# In[45]:


# Fill NAs with 0.

df_Test.fillna(0, inplace= True)


# ## Fitting test and train data into a linear model

# In[46]:


# RidgeCV
from sklearn.linear_model import RidgeCV
ridge_model = RidgeCV(alphas=(0.01, 0.05, 0.1, 0.3, 1, 3, 5, 10))
ridge_model.fit(X_train, y_train)
ridge_model_preds = ridge_model.predict(df_Test)
ridge_model_preds


# ## Fitting the data into xgboost model.

# In[47]:


#xgboost model
import xgboost as xgb
xgb_model = xgb.XGBRegressor(n_estimators=340, max_depth=2, learning_rate=0.2)
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(df_Test)
predictions = ( ridge_model_preds + xgb_preds )/2
predictions


# # Submission :)

# In[48]:


# Final submission
submission = {'Id': Id.values,'SalePrice': predictions}
final_submission = pd.DataFrame(submission)
# Exporting the final prediction into a csv
final_submission.to_csv('submission_house_price.csv', index=False)
final_submission.head()


# In[49]:


# Please if you find this notebook helpful up.vote. I appreciate it.

