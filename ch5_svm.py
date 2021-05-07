#!/usr/bin/env python
# coding: utf-8

# #### 1. 소프트 마진분류

# In[2]:


import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


# In[3]:


iris = datasets.load_iris()
X = iris['data'][:,(2,3)]
y = (iris['target'] == 2).astype(np.float64)


# In[7]:


svm_clf = Pipeline([
    ('scaler', StandardScaler()),
    ('linear_svc', LinearSVC(C = 1, loss = 'hinge')), #hinge loss : maximum margin
])

svm_clf.fit(X,y)


# In[10]:


svm_clf.predict([[5.5, 1.7]])


# #### 2. 비선형 SVM

# In[25]:


from sklearn.datasets import make_moons
from sklearn.preprocessing import PolynomialFeatures

X, y = make_moons(n_samples = 100, noise = 0.15)


# In[31]:


polynomial_svm_clf = Pipeline([
    ('poly_features', PolynomialFeatures(degree = 3)),
    ('scaler', StandardScaler()),
    ('svm_clf', LinearSVC(C = 10, loss = 'hinge'))
])
polynomial_svm_clf.fit(X,y)


# #### a. 다항식 커널

# In[34]:


from sklearn.svm import SVC
polynomial_svm_clf = Pipeline([
    ('scaler', StandardScaler()),
    ('svm_clf', SVC(kernel = 'poly', degree = 3,
                    coef0 = 1, C = 5))
])
polynomial_svm_clf.fit(X,y)


# #### b. 가우시안 RBF 커널

# In[35]:


polynomial_svm_clf = Pipeline([
    ('scaler', StandardScaler()),
    ('svm_clf', SVC(kernel = 'rbf', gamma = 5, C = 0.0001)) #gamma, C 튜닝, gamma 규제역할함
])
polynomial_svm_clf.fit(X,y)


# #### 3. SVM 회귀

# In[36]:


from sklearn.svm import LinearSVR

svm_reg = LinearSVR(epsilon = 1.5)
svm_reg.fit(X,y)

svm_poly_reg = LinearSVR(kernal = 'poly', degree = 2, C = 100, epsilon = 1.5)
svm_poly_reg.fit(X,y)

