from sklearn import datasets
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

boston= datasets.load_boston()

#features and labels
X=boston.data
y=boston.target

# print(X)
# print(X.shape)
# print(y)
# print(y.shape)

#algorithm
lin_reg=linear_model.LinearRegression()


# # print(X.T)#for taking transpose of matrix X
# plt.scatter(X.T[5],y) #taking the 6th feature rm which gives appropriate linear graph
# plt.show()
# # plotting was done beforehand to check which feature is dependent on independent value y

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

model=lin_reg.fit(X_train,y_train)
prediction=model.predict(X_test)

#linear regression is represented as Y=mX+C
print("predictions: ",prediction)
print("r2 value: ",model.score(X,y)) #accuracy is checked by r2 value, more r2 value more the point is closer to regression line
print("coeff: ",lin_reg.coef_) #value of coefficient or slope m (slope m=dy/dx)
print("intercept: ",lin_reg.intercept_) #value of intercept Y
