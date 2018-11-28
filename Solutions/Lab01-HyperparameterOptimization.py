import pandas as pd
from time import time
from scipy.stats import randint as sp_randint

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from hpsklearn import HypeEstimator
from hyperopt import tpe

data = pd.read_csv('./data/numerai_training_data.csv', index_col='id')
data.sample(3000)

#Focus on target_elizabeth
data['era'] = data['era'].apply(lambda x: float(x[3:]))
columns = ["feature{}".format(i) for i in range(1,51)]
columns += ['era', 'target_elizabeth']

data = data.loc[:,columns]

data.head()


X = data.iloc[:,:-1]
y = data['target_elizabeth']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.05)

#Task 1: Manual Tuning
#Manually fit a Support Vector Classifier to our dataset.
rfc = RandomForestClassifier()

start = time()
rfc_man = rfc.fit(X_train,y_train)

y_pred = rfc_man.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("accuracy: {}".format(acc))

#Task 2: GridSearchCV
parms = {
    'n_estimators': [5,10,50],
    'max_depth': [10,20,50],
    'min_samples_split': [5,10,20],
    'min_samples_leaf': [5,10],
    'n_jobs': [-1]
}

rfc_grid = GridSearchCV(rfc, parms, cv=5)
rfc_grid.fit(X_train, y_train)

y_pred = rfc_grid.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("accuracy: {}".format(acc))

#Task 2: RandomizedSearchCV
n_iter_search = 20
parms = {
    'n_estimators': sp_randint(2,1000),
    'criterion': ['gini'],
    'max_depth': sp_randint(5,100),
    'min_samples_split': sp_randint(1,50),
    'min_samples_leaf': sp_randint(1,35),
    'bootstrap': ['True'],
    'n_jobs': [-1]
}

random_search = RandomizedSearchCV(rfc, param_distributions=parms,
                                    n_iter = n_iter_search, cv=5)

y_pred = rfc_grid.predict(X_test)
acc - accuracy_score(y_test,y _pred)
print("accuracy: {}".format(acc))


#Task 3: Bayesian Optimization
