import pandas as pd
import numpy as np

data = pd.read_csv("C:\\Users\\SIDDU\\OneDrive\\Desktop\\work\\selflearnml\\practice\\adult.data")
features = ["age","workclass","fnlwgt","education","education-num","marital-status",
    "occupation","relationship","race","sex","capital-gain","capital-loss","hours-per-week",
    "native-country"]
label = "Salary"
X = data.drop([' <=50K'], axis=1)
X.columns = features
y = data[' <=50K']
y.name = label

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X.drop(['education-num'],axis=1), y, test_size=0.25, random_state=0, stratify=y)


# LabelEncoder for labels
from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
y_train_encoded = pd.Series(lb.fit_transform(y_train))
y_test_encoded = pd.Series(lb.transform(y_test))
y_test_encoded.name = "salary"
# print(X_test.merge(y_test_encoded,left_index=True,right_index=True))

# OneHotEncoding for non heirarchical features
OHEfeatures = ["workclass","marital-status","occupation","relationship","race","sex","native-country"]
from sklearn.preprocessing import OneHotEncoder
oh = OneHotEncoder(sparse=False,handle_unknown='ignore')
X_train_OHEncoded = pd.DataFrame(oh.fit_transform(X_train[OHEfeatures]))
X_test_OHEncoded = pd.DataFrame(oh.transform(X_test[OHEfeatures]))


# OrdinalEncoder for heirarchical features
OEfeatures = ["education"]
edu = [' Preschool',' 1st-4th',' 5th-6th',' 7th-8th',' 9th',' 10th',' 11th',' 12th',' HS-grad' ,' Prof-school',' Some-college',' Assoc-acdm',' Assoc-voc',' Bachelors',' Masters',' Doctorate']
from sklearn.preprocessing import OrdinalEncoder
oe = OrdinalEncoder(categories=[edu])
X_train_OEncoded = pd.DataFrame(oe.fit_transform(X_train[OEfeatures]))
X_test_OEncoded = pd.DataFrame(oe.fit_transform(X_test[OEfeatures]))


X_train_encoded = X_train[["age","fnlwgt","capital-gain","capital-loss","hours-per-week"]].merge(X_train_OHEncoded,left_index=True,right_index=True,how='left').merge(X_train_OEncoded,left_index=True,right_index=True,how='left')
X_test_encoded = X_test[["age","fnlwgt","capital-gain","capital-loss","hours-per-week"]].merge(X_test_OHEncoded,left_index=True,right_index=True,how='left').merge(X_test_OEncoded,left_index=True,right_index=True,how='left')
