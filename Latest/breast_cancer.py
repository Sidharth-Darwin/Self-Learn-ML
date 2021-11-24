import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

full_data = load_breast_cancer()
X = full_data.data
y = full_data.target
# print(y.shape)
# print(X.shape)
X = pd.DataFrame(data=X,columns=full_data.feature_names)
y = pd.Series(data=y)

classes= ['malignant', 'benign']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

model = SVC(kernel='linear',C=1,random_state=0)
model.fit(X_train,y_train)
predictions = model.predict(X_test)

print("SVC:",accuracy_score(y_test,predictions))


model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train,y_train)
predictions = model.predict(X_test)

print("KNN:",accuracy_score(y_test,predictions))


model = RandomForestClassifier(n_estimators=100,random_state=0)
model.fit(X_train,y_train)
predictions = model.predict(X_test)

print("RFC:",accuracy_score(y_test,predictions))


model = LogisticRegression(random_state=0,solver='lbfgs', max_iter=100)
model.fit(X_train,y_train)
predictions = model.predict(X_test)

print("LogR:",accuracy_score(y_test,predictions))


# for i in range(len(list(predictions))):
#     print(classes[list(predictions)[i]],":",classes[list(y_test)[i]])
