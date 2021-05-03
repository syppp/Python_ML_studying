#!/usr/bin/env python
# coding: utf-8

# In[40]:


import pandas as pd
import os
import numpy as np

os.chdir("C:/Users/samsung/Desktop/대학교/4학년 2학기/파이썬공부/data")


# ### 1. 데이터 준비 및 살펴보기

# In[41]:


from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', version=1, cache=True)


# In[55]:


X, y = mnist['data'],mnist['target']
y = y.astype(np.uint8)


# In[62]:


X_train, X_test, y_train, y_test = X[:60000],X[60000:], y[:60000],y[60000:]


# In[53]:


import matplotlib as mpl
import matplotlib.pyplot as plt

digit = np.array(X.iloc[0])
digit_image = digit.reshape(28,28)

plt.imshow(digit_image, cmap = 'binary')
plt.axis('off')
plt.show()


# ### 2. 이진분류기

# In[57]:


y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)


# In[61]:


X_train


# In[63]:


from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state = 42)
sgd_clf.fit(X_train, y_train_5)


# In[80]:


sgd_clf.predict(X_test) #2차원 형태로 만들어줘야 함!


# #### a. 교차검증

# In[102]:


from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
results = []
skfolds = StratifiedKFold(n_splits =3, random_state = 42, shuffle = True)
for train_index, test_index in skfolds.split(X_train, y_train_5):
    clone_clf = clone(sgd_clf)
    X_train_folds = X_train.iloc[train_index]
    y_train_folds = y_train_5.iloc[train_index]
    X_test_folds = X_train.iloc[test_index]
    y_test_folds = y_train_5.iloc[test_index]
    
    clone_clf.fit(X_train_folds,y_train_folds)
    y_pred = clone_clf.predict(X_test_folds)
    accuracy = sum(y_pred == y_test_folds)/len(y_test_folds)
    print(accuracy)
    results.append(accuracy)


# In[103]:


np.mean(results)


# In[101]:


from sklearn.model_selection import cross_val_score

cross_val_score(sgd_clf, X_train, y_train_5, cv = 3, scoring = 'accuracy')


# In[106]:


from sklearn.model_selection import cross_val_predict
y_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv = 3)


# In[108]:


len(y_pred) #cv 전체 predict 값


# #### b. 정확도, 재현율, f1_score

# In[109]:


from sklearn.metrics import confusion_matrix

confusion_matrix(y_train_5, y_pred)


# In[111]:


from sklearn.metrics import precision_score, recall_score, f1_score

precision_score(y_train_5, y_pred)
recall_score(y_train_5, y_pred)
f1_score(y_train_5, y_pred)


# In[113]:


from sklearn.metrics import precision_recall_curve
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv = 3, method= 'decision_function')
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)


# In[117]:


#정밀도, 재현율 trade-off
plt.plot(thresholds, precisions[:-1], "b--", label = "정밀도")
plt.plot(thresholds, recalls[:-1], "g--", label = "재현율")
plt.show()


# In[128]:


#정밀도가 0.9가 되도록 임계값 정하기
precision_score(y_train_5, (y_scores> thresholds[np.argmax(precisions >= 0.9)])) 


# In[129]:


#ROC 곡선
from sklearn.metrics import roc_curve
fpr, tpr, threshols = roc_curve(y_train_5, y_scores)


# In[132]:


plt.plot(fpr, tpr, linewidth = 2)
plt.plot([0,1],[0,1], 'k--')
plt.show()


# #### c. RF와 SGD ROC Curve로 분류기 선택

# In[135]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()

rf_scores = cross_val_predict(rf, X_train, y_train_5, cv = 3, method= 'predict_proba')


# In[142]:


fpr, tpr, threshols = roc_curve(y_train_5, rf_scores[:,1])
plt.plot(fpr, tpr, `linewidth = 2)
plt.plot([0,1],[0,1], 'k--')
plt.show()


# ### 3. 다중레이블 분류

# In[144]:


from sklearn.neighbors import KNeighborsClassifier

y_train_large = (y_train >= 7)
y_train_odd = (y_train % 2 == 1)
y_multilabel = np.c_[y_train_large, y_train_odd]


# In[148]:


knn = KNeighborsClassifier()
knn.fit(X_train, y_multilabel)


# In[152]:


y_pred = cross_val_predict(knn, X_train, y_multilabel, cv = 3)
f1_score(y_multilabel, y_pred, average = 'macro')


# ### 4. 다중 출력 분류

# In[157]:


noise = np.random.randint(0, 100, (len(X_train), 784))
X_train_mode = X_train + noise
noise = np.random.randint(0, 100, (len(X_test), 784))
X_test_mode = X_test + noise


# In[158]:


y_train_mode = X_train
y_test_mode = X_test


# In[159]:


knn.fit(X_train_mode, y_train_mode)


# In[165]:


a = knn.predict([X_test_mode.iloc[1]])


# In[166]:


digit_image =a.reshape(28,28)

plt.imshow(digit_image, cmap = 'binary')
plt.axis('off')
plt.show()

