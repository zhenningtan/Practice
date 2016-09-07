
# coding: utf-8

# ## 8/26/2016 This is a practice to use machine learning to predict admission using a dataset (http://www.ats.ucla.edu/stat/data/binary.csv)

# In[97]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
import time

pd.set_option('display.max_colwidth', -1)


# In[98]:

cd "C:\Users\Zhenning\Documents\Code\practice"


# In[99]:

df = pd.read_csv("binary.csv")  # this dataset can be feteched from "http://www.ats.ucla.edu/stat/data/binary.csv"


# In[100]:

df.head()


# In[101]:

df.info()


# In[102]:

df.describe()


# In[103]:

df.shape


# In[104]:

# rename "rank" column to avoid a confliction with a dataframe method rank
col_names = ["admit", "gre", "gpa", "reputation"]
df.columns = col_names
df.head()


# ## Dataset summary and look at data

# In[105]:

# plot each feature 
df.hist(figsize = (10, 10))


# ### The admission column, which is our label for data point, has uneven number of two groups. For machine learning, it may be worhtwile to create stratified splits for cross validation instead of simple train and test set split. 

# In[106]:

# calculate the majority class prediction accuracy
admit_group = df.groupby( "admit")


# In[107]:

admit_group.describe()


# In[108]:

majority_class_pred = 127 /273.0
print "majority class prediction", majority_class_pred


# In[109]:

df.groupby("admit").hist(figsize = (10,10))


# ## Feature engineering: dummy variables

# In[110]:

dummy_rep = pd.get_dummies(df["reputation"], prefix = "reputation")


# In[111]:

dummy_rep.head()


# In[112]:

df = df.drop('reputation', axis = 1)  # drop the original reputation column
df = df.join(dummy_rep)
df.head()


# ### Create a new variable "gre_gpa", which is to multiply gre with gpa. Because both factors are important in determining the admission, it's possible that the multiplication of two will give better prediction on admission.

# In[113]:

df["gre_gpa"] = df['gre'] * df ['gpa']


# In[114]:

df.head()


# ## Split dataset for training model

# In[115]:

from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.cross_validation import train_test_split

from sklearn.preprocessing import MinMaxScaler


# In[116]:

features = ["gre", "gpa", "gre_gpa", "reputation_1", "reputation_2", "reputation_3", "reputation_4"]
X = df[features]
y = df["admit"]


# In[117]:

sss = StratifiedShuffleSplit(y, n_iter = 100, test_size = 0.2, random_state = 0)


# In[118]:

# another way is to create simple train, test set split. However, this split may have issue due to unbalanced labels in two groups
# I will create a simple train, test set for now and compare training results with sss later
X_train, X_test = train_test_split(X, test_size = 0.3, random_state = 0)
y_train, y_test = train_test_split(y, test_size = 0.3, random_state =0)


# ### Scale data between 0 and 1

# In[119]:

mms = MinMaxScaler()
mms.fit(X)
X = mms.transform(X)


# ### Create a helper function to store algorithm performance 

# In[120]:

from sklearn import metrics


# In[121]:

result_table = []
def predict_score(clf, features, labels, folds = 1000):
    
    cv = StratifiedShuffleSplit(labels, folds, random_state = 42)
    
    true_negatives = 0
    false_negatives = 0
    true_positives = 0
    false_positives = 0
    for train_idx, test_idx in cv: 
        features_train = []
        features_test  = []
        labels_train   = []
        labels_test    = []
        for ii in train_idx:
            features_train.append( features[ii] )
            labels_train.append( labels[ii] )
        for jj in test_idx:
            features_test.append( features[jj] )
            labels_test.append( labels[jj] )
        
        ### fit the classifier using training set, and test on test set
        clf.fit(features_train, labels_train)
        predictions = clf.predict(features_test)
        for prediction, truth in zip(predictions, labels_test):
            if prediction == 0 and truth == 0:
                true_negatives += 1
            elif prediction == 0 and truth == 1:
                false_negatives += 1
            elif prediction == 1 and truth == 0:
                false_positives += 1
            elif prediction == 1 and truth == 1:
                true_positives += 1
            else:
                print "Warning: Found a predicted label not == 0 or 1."
                print "All predictions should take value 0 or 1."
                print "Evaluating performance for processed predictions:"
                break
    try:
        total_predictions = true_negatives + false_negatives + false_positives + true_positives
        accuracy = 1.0*(true_positives + true_negatives)/total_predictions
        precision = 1.0*true_positives/(true_positives+false_positives)
        recall = 1.0*true_positives/(true_positives+false_negatives)
        f1 = 2.0 * true_positives/(2*true_positives + false_positives+false_negatives)
        f2 = (1+2.0*2.0) * precision*recall/(4*precision + recall)
    
        #store the scores in a list and convert it to a dataframe
        result = [str(clf), "{0:.4f}".format(precision), "{0:.4f}".format(recall), "{0:.4f}".format(accuracy),
              "{0:.4f}".format(f1),  "{0:.4f}".format(f2)]
        if result not in result_table:
            result_table.append(result)
            
        result_df = pd.DataFrame(result_table, columns =["Classifier", "Precision", "Recall", "Accuracy", "F1", "F2"])
        return result_df
    
    except:
        print "Got a divide by zero when trying out:", clf
        print "Precision or recall may be undefined due to a lack of true positive predicitons."


# ## Test a few machine learning algorithms

# In[122]:

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.grid_search import GridSearchCV


# ### 1. Logistic regression

# In[123]:

params = { "C":  [0.1, 1, 10, 50, 100],
           "tol": [ 1e-1, 1e-2, 1e-3, 1e-4],
           "class_weight": [None, "auto"]
           }
log_clf = LogisticRegression()

t1 = time.time()
gs_clf = GridSearchCV(log_clf, param_grid = params, scoring = "f1", cv = sss)
gs_clf.fit(X, y)

t2 = time.time()
print "time:", (t2-t1), "second\n"
log_clf = gs_clf.best_estimator_
print "Best logistic regression:", log_clf


# In[124]:

predict_score(log_clf, X, y)


# In[125]:

coeffs = log_clf.coef_[0]
for ind, i in enumerate(coeffs):
    print features[ind], ":", i


# ### Based on the coefficients, I find that GRE, GPA have positive impact on the admission rate, as well as the school reputation in the first tier. School reputation in the second to fourth tier has a negative impact on admission rate. 

# ### 2. Naive Bayes 

# In[126]:

nb_clf = GaussianNB()

nb_clf.fit(X_train, y_train)

predict_score(nb_clf, X, y)


# ### 3. Decision tree

# In[127]:

params = { "min_samples_split":  [2, 5, 10, 20, 30],
           "max_depth" : [5, 7, 10, 15, 20, 30,40],
           "max_features": [1,2,3,4,5,6,7],
           "class_weight": [None, "auto"]}

tree_clf = DecisionTreeClassifier(random_state = 0)

t1 = time.time()
gs_clf = GridSearchCV(tree_clf, param_grid = params, scoring = "f1", cv = sss)
gs_clf.fit(X, y)

t2 = time.time()
print "time:", t2-t1, "second\n"

tree_clf = gs_clf.best_estimator_
print "Best found classifier:", tree_clf


# In[128]:

print tree_clf.n_features_
print tree_clf.feature_importances_


# In[129]:

predict_score(tree_clf, X, y)


# ### 4. Adaboost

# In[137]:

# Ensemble method, Adaboost
t1= time.time()
params = { "n_estimators": [5, 10, 50, 100],
           "learning_rate": [0.1, 1, 10, 50]}

adaboost_clf = AdaBoostClassifier(base_estimator= DecisionTreeClassifier(class_weight= "auto"), random_state = 0)

gs_clf = GridSearchCV(adaboost_clf, param_grid = params, scoring = "f1", cv = sss)
gs_clf.fit(X, y)

t2= time.time()
print "time", t2-t1, "seconds\n"

adaboost_clf = gs_clf.best_estimator_
print "Best found classifier:", adaboost_clf


# In[138]:

predict_score(adaboost_clf, X, y)


# ### 5. Random Forest

# In[132]:

#  Random forest: fits a number of decision tree classifiers on various sub-samples of the dataset and 
# use averaging to improve the predictive accuracy and control over-fitting
t1= time.time()
params = {"n_estimators": [ 25, 50, 75, 100],
          "min_samples_split": [1, 2, 5, 10, 20, 30],
          "max_depth": [10, 50, 75, 100]}

rf_clf = RandomForestClassifier(random_state = 0, class_weight = "auto")

gs_clf = GridSearchCV(rf_clf, param_grid = params, scoring = "f1", cv =sss)
gs_clf.fit(X, y)

t2= time.time()
print "time", t2-t1, "second\n"

rf_clf = gs_clf.best_estimator_
print "Best found classifier:", rf_clf


# In[133]:

predict_score(rf_clf, X, y)


# ### 6. K Nearest Neighbors

# In[139]:

t1= time.time()
params = {"n_neighbors": [3,4, 5,6,7, 10, 15, 20],
          "weights": ["uniform", "distance"]}

KNN_clf = KNeighborsClassifier()

gs_clf = GridSearchCV(KNN_clf, param_grid = params, scoring = "f1", cv =sss) 

gs_clf.fit(X, y)

t2= time.time()
print "time", t2-t1, "second\n"

KNN_clf = gs_clf.best_estimator_
print "Best found classifier:", KNN_clf


# In[140]:

predict_score(KNN_clf, X, y)


# ## Summary
# ### I tried a few different classifiers in sklearn and found that logistic regression (C = 1, tol = 0.01, class weight = auto) gives the best prediction on this dataset with F1 score of 0.5246. GRE, GPA and tier 1 school ranking have a positive contibution to admission, while tier 2, 3 and 4 schools have negative effect on admission.

# In[136]:

#!ipython nbconvert --to python Prediction_Student_Admission.ipynb

