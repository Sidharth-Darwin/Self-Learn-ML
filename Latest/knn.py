from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv('selflearnml\\car.data')

LE = LabelEncoder()
# buying,maint,doors,persons,lug_boot,safety,class
buying = LE.fit_transform(list(data['buying']))
maint = LE.fit_transform(list(data['maint']))
doors = LE.fit_transform(list(data['doors']))
persons = LE.fit_transform(list(data['persons']))
lug_boot = LE.fit_transform(list(data['lug_boot']))
safety = LE.fit_transform(list(data['safety']))
cls = LE.fit_transform(list(data['class']))

X = pd.DataFrame(data={"buying":buying,"maint":maint,"doors":doors,"persons":persons,"lug_boot":lug_boot,"safety":safety})
y = pd.Series(cls)

X_train,X_test,y_train,y_test= train_test_split(X,y,train_size=0.8)
model = KNeighborsClassifier(n_neighbors=7)
model.fit(X_train,y_train)
predictions= model.predict(X_test)

print(accuracy_score(y_test,predictions))
