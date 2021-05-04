#!/usr/bin/env python
# coding: utf-8

# #### 1. Ridge Regression

# In[5]:


import numpy as np

#rand : 0-1 균일분포
#randn : 표준정규분포
X = 2*np.random.rand(100,1)
y = 4 + 3 * X + np.random.randn(100,1)


# In[17]:


from sklearn.linear_model import Ridge,SGDRegressor

ridge = Ridge(alpha = 1, solver = 'cholesky')
ridge.fit(X,y)


# In[15]:


sgd_ridge = SGDRegressor(penalty = 'l2')
sgd_ridge.fit(X,y)


# #### 2. Lasso Regression

# In[18]:


from sklearn.linear_model import Lasso, ElasticNet
lasso = Lasso(alpha = 0.1)
lasso.fit(X,y)

elastic = ElasticNet(alpha = 0.1, l1_ratio  = 0.5) #l1_ration : 혼합비율
elastic.fit(X,y)


# #### 3. Early Stopping

# In[23]:


from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
X_train,X_val, y_train, y_val = train_test_split(X,y, test_size = 0.2)
sgd_reg = SGDRegressor(max_iter = 1, tol = -np.infty, warm_start = True,
                      penalty = None, learning_rate = 'constant', eta0 = 0.0005)

minimum_val_error = float('inf')
best_epoch = None
best_model = None
for epoch in range(1000):
    sgd_reg.fit(X_train,y_train)
    y_pred = sgd_reg.predict(X_val)
    val_error = mean_squared_error(y_val, y_pred)
    
    if val_error < minimum_val_error : 
        minimum_val_error = val_error
        best_epoch = epoch
        best_model = clone(sgd_reg)
    


# #### 4. Logistic / Softmax(Multinomial Logistic)

# In[25]:


from sklearn import datasets
iris = datasets.load_iris()

X = iris['data'][:, 3:]
y = (iris['target'] == 2).astype(np.int)


# In[30]:


X.shape


# In[35]:


from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(X,y)
logreg.predict([[1.7]]) #2차원이어야 함
logreg.predict_proba([[1.7]]) #뒤가 y일 확률


# In[39]:


X = iris['data'][:, (2,3)]
y = iris['target']

multi_reg = LogisticRegression(multi_class = 'multinomial', solver = 'lbfgs', C = 10) #C:l2
multi_reg.fit(X,y)
multi_reg.predict_proba([[5,2]]) 
multi_reg.predict([[5,2]]) 

