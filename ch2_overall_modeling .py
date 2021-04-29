#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os
import numpy as np


# In[2]:


os.chdir("C:/Users/samsung/Desktop/대학교/4학년 2학기/파이썬공부/data")
housing = pd.read_csv('housing.csv')


# #### 1. 기본 데이터셋 정보 확인

# In[7]:


housing.head()


# In[8]:


housing.info()


# In[9]:


housing["ocean_proximity"].value_counts()


# In[10]:


housing.describe()


# In[9]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
housing.hist(bins = 50, figsize = (10,10))
plt.show()


# #### 2. train/test split

# #### a.일반 나누기

# In[17]:


from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(housing, test_size = 0.2, random_state = 42)


# #### b. 계층적 샘플링 위한 파생변수 만들고 계층적 샘플링

# In[4]:


housing['income_cat'] = pd.cut(housing['median_income'], bins= [0,1.5,3,4.5,6, np.inf],
                              labels = [1,2,3,4,5])


# In[5]:


from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 42)
for train_index, test_index in split.split(housing, housing['income_cat']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]


# In[6]:


strat_train_set.drop('income_cat', axis = 1, inplace = True)
strat_test_set.drop('income_cat', axis = 1, inplace = True)


# #### 시각화

# In[7]:


housing = strat_train_set.copy()


# In[10]:


housing.plot(kind = "scatter", x = "longitude", y = "latitude", alpha = 0.4, 
            s= housing['population']/100, label = 'population', figsize = (10,7),
            c = "median_house_value", cmap = plt.get_cmap("jet"), colorbar = True, sharex = False)
plt.legend()


# #### 상관관계 분석

# In[11]:


corr_matrix = housing.corr()
corr_matrix


# In[12]:


from pandas.plotting import scatter_matrix

attributes = ['median_house_value', 'median_income', 'total_rooms', 'housing_median_age']
scatter_matrix(housing[attributes], figsize = (12,8))


# #### 3. 전처리

# In[13]:


#이후 파이프라인으로 한번에 처리
housing['rooms_per_household'] = housing['total_rooms']/housing['households']
housing['bedrooms_per_household'] = housing['total_bedrooms']/housing['households']
housing['population_per_household'] = housing['population']/housing['households']


# In[14]:


housing = strat_train_set.drop("median_house_value", axis = 1)
housing_labels = strat_train_set['median_house_value'].copy()


# #### imputation

# In[15]:


from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy = 'median')
housing_num = housing.drop('ocean_proximity', axis = 1)
imputer.fit(housing_num)
X = imputer.transform(housing_num)


# In[17]:


housing_tr = pd.DataFrame(X, columns = housing_num.columns, index = housing_num.index)


# #### 범주형

# In[19]:


housing_cat = housing[['ocean_proximity']]


# In[20]:


from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)


# In[22]:


from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)


# In[23]:


housing_cat_1hot.toarray()


# #### 파이프라인(Train data 전처리 파이프라인)

# In[24]:


from sklearn.base import BaseEstimator, TransformerMixin

rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self  
    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


# In[25]:


attr_adder = CombinedAttributesAdder(add_bedrooms_per_room = False)
housing_extra_attribs = attr_adder.transform(housing.values)


# In[28]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([('imputer', SimpleImputer(strategy = 'median')),
                        ('attribs_adder', CombinedAttributesAdder()),
                        ('std_scaler', StandardScaler())])
housing_num_tr = num_pipeline.fit_transform(housing_num)


# In[29]:


from sklearn.compose import ColumnTransformer

num_attrbs = list(housing_num)
cat_attrbs = ['ocean_proximity']

full_pipeline = ColumnTransformer([('num', num_pipeline, num_attrbs),
                                  ('cat',OneHotEncoder(), cat_attrbs)])

housing_prepared = full_pipeline.fit_transform(housing)


# #### 4. modeling

# #### a. 교차검증

# In[37]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
tree_reg = DecisionTreeRegressor()
scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                         scoring = "neg_mean_squared_error", cv = 10)


# In[35]:


tree_rmse_scores = np.sqrt(-scores)
tree_rmse_scores.mean()


# #### b. 그리드서치

# In[38]:


from sklearn.model_selection import GridSearchCV

param_grid = [
    {'n_estimators' : [3,10, 30],
    'max_features' : [2,4,6,8]},
    {'bootstrap' : [False],
    'n_estimators' : [3,10],
    'max_features' : [2,3,4]},
]


# In[43]:


from sklearn.ensemble import RandomForestRegressor
forest = RandomForestRegressor()


# In[44]:


gs = GridSearchCV(forest, param_grid, cv = 5,
                 scoring = 'neg_mean_squared_error',
                 return_train_score = True)
gs.fit(housing_prepared, housing_labels)


# In[45]:


gs.best_params_


# In[47]:


gs.best_estimator_


# In[80]:


results = {}
for ms, params in zip(gs.cv_results_['mean_test_score'], gs.cv_results_['params']):
    results[np.sqrt(-ms)] = params


# #### c. 테스트

# In[82]:


final = gs.best_estimator_


# In[83]:


strat_test_set.head()


# In[88]:


x_test = full_pipeline.transform(strat_test_set.drop('median_house_value',axis =1))
y_test = strat_test_set['median_house_value'].copy()


# In[98]:


a = final.predict(x_test)


# In[100]:


final


# In[99]:


from sklearn.metrics import mean_squared_error
np.sqrt(mean_squared_error(a, y_test))

