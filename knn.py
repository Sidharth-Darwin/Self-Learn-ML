import numpy as np
import pandas as pd
from sklearn import neighbors, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

data=pd.read_csv('C:\\Users\\SIDDU\\OneDrive\\Desktop\\work\\selflearnml\\car.data')
# print(data.head(30))

#extracting only necessasary features and labels
X=data[[
    "buying",
    "maint",
    "safety"
]].values
y=data[["class"]]
# print(X,y)

#converting the data from strings to numbers for algorithm
#x
Le=LabelEncoder()
for i in range(len(X[0])):
    X[:,i]=Le.fit_transform(X[:,i])
# print(X)
#y
label_mapping={
    "unacc":0,
    "acc":1,
    "good":2,
    "vgood":3,
}
y["class"]=y["class"].map(label_mapping)
y=np.array(y)
# print(y)

#start creating a knn model
knn=neighbors.KNeighborsClassifier(n_neighbors=25,weights="uniform")

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25)

#train the model
knn.fit(X_train,y_train)

predictions= knn.predict(X_test)
accuracy=metrics.accuracy_score(y_test,predictions)

# print("predictions: ",predictions)
# print("accuracy: ",accuracy)

a=1700
print("exact value: ",y[a])
print("prediction: ",knn.predict(X)[a])
