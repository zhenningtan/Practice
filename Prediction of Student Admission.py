
# coding: utf-8

# ## This is a practice to use machine learning to predict admission using a dataset (http://www.ats.ucla.edu/stat/data/binary.csv)

# In[41]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# In[42]:

cd "C:\Users\Zhenning\Documents\Code\practice"


# In[43]:

df = pd.read_csv("binary.csv")  # this dataset can be feteched from "http://www.ats.ucla.edu/stat/data/binary.csv"


# In[44]:

df.head()


# In[45]:

df.info()


# In[46]:

df.describe()


# In[47]:

df.shape


# In[48]:

# rename "rank" column to avoid a confliction with a dataframe method rank
col_names = ["admit", "gre", "gpa", "reputation"]
df.columns = col_names
df.head()


# ## Dataset summary and look at data

# In[150]:

# plot each feature 
df.hist(figsize = (10, 10))


# ### The admission column, which is our label for data point, has uneven number of two groups. For machine learning, it may be worhtwile to create stratified splits for cross validation instead of simple train and test set split. 

# In[133]:

# calculate the majority class prediction accuracy
admit_group = df.groupby( "admit")


# In[134]:

admit.describe()


# In[136]:

majority_class_pred = 127 /273.0
print "majority class prediction", majority_class_pred


# In[152]:

df.groupby("admit").hist(figsize = (10,10))


# ## Feature engineering: dummy variables

# In[50]:

dummy_rep = pd.get_dummies(df["reputation"], prefix = "reputation")


# In[51]:

dummy_rep.head()


# In[52]:

df = df.drop('reputation', axis = 1)  # drop the original reputation column
df = df.join(dummy_rep)
df.head()


# ## Split dataset for training model

# In[53]:

from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.cross_validation import train_test_split


# In[54]:

X = df[["gre", "gpa", "reputation_1", "reputation_2", "reputation_3", "reputation_4"]]
y = df["admit"]


# In[55]:

sss = StratifiedShuffleSplit(y, n_iter = 10, test_size = 0.1, random_state = 0)


# In[58]:

# another way is to create simple train, test set split 
X_train, X_test = train_test_split(X, test_size = 0.3, random_state = 0)
y_train, y_test = train_test_split(y, test_size = 0.3, random_state =0)


# ## Test a few machine learning algorithms

# In[113]:

from sklearn.preprocessing import MinMaxScaler

from sklearn.cross_validation import cross_val_score

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier


# In[68]:

# Scale data between 0 and 1
mms = MinMaxScaler()
mms.fit(X)
X = mms.transform(X)


# ### 1. Logistic regression

# In[117]:

# evaluate logistic regression using stratified split
log_clf= LogisticRegression()

log_sss_scores = cross_val_score(log_clf, X, y,  cv = sss)
print "Score for sss:", log_sss_scores.mean()


# In[112]:

log_clf.fit(X_train, y_train)
print'Coefficients: \n', lr_clf.coef_
print "Score:", log_clf.score(X_test, y_test)


# ### 2. Naive Bayes 

# In[116]:

nb_clf = GaussianNB()
nb_clf.fit(X_train, y_train)
print "Score:", nb_clf.score(X_test, y_test)


# In[115]:

nb_sss_score = cross_val_score(nb_clf, X, y,  cv = sss)
print "Score for sss:", nb_sss_score.mean()


# ### 3. Decision tree

# In[91]:

dt_clf = DecisionTreeClassifier()
dt_clf.fit(X_train, y_train)
print "Score:", dt_clf.score(X_test, y_test)


# In[118]:

dt_sss_score = cross_val_score(dt_clf, X, y,  cv = sss)
print "Score for sss:", dt_sss_score.mean()


# ### 4. Adaboost

# In[119]:

# Ensemble method, Adaboost
n_estimators = [1, 5, 10, 20, 50]
for i in n_estimators:
    clf = AdaBoostClassifier(n_estimators = i)
    clf.fit(X_train, y_train)
    print "Score for {} estimators: {}".format(i, clf.score(X_test, y_test))


# In[120]:

adaboost_clf = AdaBoostClassifier(n_estimators = 5)  # use n_estimator = 5 for best performance
adaboost_sss_score = cross_val_score(adaboost_clf, X, y,  cv = sss)
print "Score for sss:", adaboost_sss_score.mean()

