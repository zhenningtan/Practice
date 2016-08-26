
# coding: utf-8

# ## This is a practice to use machine learning to predict admission using a dataset (http://www.ats.ucla.edu/stat/data/binary.csv)

# In[144]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# In[145]:

cd "C:\Users\Zhenning\Documents\Code\practice"


# In[146]:

df = pd.read_csv("binary.csv")  # this dataset can be feteched from "http://www.ats.ucla.edu/stat/data/binary.csv"


# In[147]:

df.head()


# In[148]:

df.info()


# In[149]:

df.describe()


# In[150]:

df.shape


# In[151]:

# rename "rank" column to avoid a confliction with a dataframe method rank
col_names = ["admit", "gre", "gpa", "reputation"]
df.columns = col_names
df.head()


# ## Dataset summary and look at data

# In[152]:

# plot each feature 
df.hist(figsize = (10, 10))


# ### The admission column, which is our label for data point, has uneven number of two groups. For machine learning, it may be worhtwile to create stratified splits for cross validation instead of simple train and test set split. 

# In[153]:

# calculate the majority class prediction accuracy
admit_group = df.groupby( "admit")


# In[154]:

admit_group.describe()


# In[155]:

majority_class_pred = 127 /273.0
print "majority class prediction", majority_class_pred


# In[156]:

df.groupby("admit").hist(figsize = (10,10))


# ## Feature engineering: dummy variables

# In[157]:

dummy_rep = pd.get_dummies(df["reputation"], prefix = "reputation")


# In[158]:

dummy_rep.head()


# In[159]:

df = df.drop('reputation', axis = 1)  # drop the original reputation column
df = df.join(dummy_rep)
df.head()


# ### Create a new variable "gre_gpa", which is to multiply gre with gpa. Because both factors are important in determining the admission, it's possible that the multiplication of two will give better prediction on admission.

# In[160]:

df["gre_gpa"] = df['gre'] * df ['gpa']


# In[161]:

df.head()


# ## Split dataset for training model

# In[162]:

from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.cross_validation import train_test_split

from sklearn.preprocessing import MinMaxScaler


# In[163]:

X = df[["gre", "gpa", "gre_gpa", "reputation_1", "reputation_2", "reputation_3", "reputation_4"]]
y = df["admit"]


# In[164]:

sss = StratifiedShuffleSplit(y, n_iter = 10, test_size = 0.1, random_state = 0)


# In[165]:

# another way is to create simple train, test set split. However, this split may have issue due to unbalanced labels in two groups
# I will create a simple train, test set for now and compare training results with sss later
X_train, X_test = train_test_split(X, test_size = 0.3, random_state = 0)
y_train, y_test = train_test_split(y, test_size = 0.3, random_state =0)


# ### Scale data between 0 and 1

# In[166]:

mms = MinMaxScaler()
mms.fit(X)
X = mms.transform(X)


# ### Create a helper function to store algorithm performance scores 

# In[167]:

result_table = []
def predict_score(clf, feature, true_label):
    
    #calculate predicted label and scores
    y_pred = clf.predict(feature)
    precision = metrics.precision_score(true_label, y_pred)
    recall = metrics.recall_score(true_label, y_pred)
    accuracy = metrics.accuracy_score(true_label, y_pred)
    
    #store the scores in a list and convert it to a dataframe
    result = [str(clf), precision, recall, accuracy]
    if result not in result_table:
        result_table.append(result)
    
    result_df = pd.DataFrame(result_table, columns =["Classifier", "Precision", "Recall", "Accuracy"])
    return result_df


# ## Test a few machine learning algorithms

# In[168]:

from sklearn.cross_validation import cross_val_score

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.grid_search import GridSearchCV
from sklearn import metrics


# ### 1. Logistic regression

# In[169]:

params = { "C":  [0.1, 1, 10, 15, 20, 30, 35, 40, 50, 60, 100]}
log_clf = LogisticRegression()

gs_clf = GridSearchCV(log_clf, param_grid = params, scoring = "f1", cv = sss)
gs_clf.fit(X, y)

log_clf = gs_clf.best_estimator_
print "Best logistic regression:", log_clf


# In[170]:

predict_score(log_clf, X, y)


# In[171]:

print "coefficients:", log_clf.coef_ # it is strange that gre_gpa varialbe has negative effect on admission


# ### 2. Naive Bayes 

# In[172]:

nb_clf = GaussianNB()
nb_clf.fit(X_train, y_train)

predict_score(nb_clf, X_test, y_test)


# In[173]:

nb_sss_score = cross_val_score(nb_clf, X, y, scoring = "f1",  cv = sss)
print "Score for sss:", nb_sss_score.mean()


# ### 3. Decision tree

# In[174]:

params = { "min_samples_split":  [2, 4, 6, 10, 20, 25, 30, 40],
           "max_depth" : [5,8, 10,12]}

tree_clf = DecisionTreeClassifier(random_state = 0)

gs_clf = GridSearchCV(tree_clf, param_grid = params, scoring = "f1", cv = sss)
gs_clf.fit(X, y)

tree_clf = gs_clf.best_estimator_
print "Best found classifier:", tree_clf


# In[175]:

predict_score(tree_clf, X, y)


# ### 4. Adaboost

# In[176]:

# Ensemble method, Adaboost
params = { "n_estimators": [100, 200, 250, 300],
           "learning_rate": [0.5, 1, 5]}

adaboost_clf = AdaBoostClassifier(random_state = 0)

gs_clf = GridSearchCV(adaboost_clf, param_grid = params, scoring = "f1", cv = sss)
gs_clf.fit(X, y)

adaboost_clf = gs_clf.best_estimator_
print "Best found classifier:", adaboost_clf


# In[177]:

predict_score(adaboost_clf, X, y)


# ### 5. Random Forest

# In[178]:

#  Random forest: fits a number of decision tree classifiers on various sub-samples of the dataset and 
# use averaging to improve the predictive accuracy and control over-fitting

params = {"n_estimators": [10, 50, 100, 200],
          "min_samples_split": [ 2, 5, 10, 20, 40]}

rf_clf = RandomForestClassifier(random_state = 0)

gs_clf = GridSearchCV(rf_clf, param_grid = params, scoring = "f1", cv =sss)
gs_clf.fit(X, y)

rf_clf = gs_clf.best_estimator_
print "Best found classifier:", rf_clf


# In[179]:

predict_score(rf_clf, X, y)


# ### 6. K Nearest Neighbors

# In[180]:

params = {"n_neighbors": [3, 5, 10, 15, 20],
          "weights": ["uniform", "distance"]}

KNN_clf = KNeighborsClassifier()

gs_clf = GridSearchCV(KNN_clf, param_grid = params, scoring = "f1", cv =sss) 

gs_clf.fit(X, y)

KNN_clf = gs_clf.best_estimator_
print "Best found classifier:", KNN_clf


# In[181]:

predict_score(KNN_clf, X, y)


# ## Summary
# ### In this practice, I tried a few different classifiers in sklearn and found that K Nearest Neighbors (k=10) gives the best prediction on this dataset. 

