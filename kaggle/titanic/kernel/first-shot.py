# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

# Any results you write to the current directory are saved as output.
import csv
import math
from sklearn.linear_model import LogisticRegression

def categorize_age(age) :
    if(np.isnan(age)) :
        return 17
    return math.ceil(age / 5)

pd.set_option('display.max_columns', 20)
train = pd.read_csv("../input/train.csv")
train["Sex"] = train["Sex"].apply(lambda x: {"male": 0, "female": 1}[x])
train["Age"] = train["Age"].apply(categorize_age)
X = train[["Pclass", "Sex", "Age", "SibSp", "Parch"]].values
y = train[["Survived"]].values
lg = LogisticRegression()
lg.fit(X, y)

test = pd.read_csv("../input/test.csv")
test["Sex"] = test["Sex"].apply(lambda x: {"male": 0, "female": 1}[x])
test["Age"] = test["Age"].apply(categorize_age)
X_test = test[["Pclass", "Sex", "Age", "SibSp", "Parch"]].values
y_test = lg.predict(X_test)

writer = csv.writer(open('predict.csv', 'w'))
writer.writerow(["PassengerId", "Survived"])
for i in range(0, test.shape[0]):
    writer.writerow([test.iloc[i].PassengerId, y_test[i]])