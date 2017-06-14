# Linear Regression with sklearn

# Imports 


import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import sklearn

# Dataset is included in sklearn

from sklearn.datasets import load_boston
boston = load_boston()

# boston is a dictionary object
boston.keys()

boston.data.shape

print(boston.feature_names)

print(boston.DESCR)

# check the data (without column name)
bos = pd.DataFrame(boston.data)
bos.head()

# add column name
bos.columns = boston.feature_names
bos.head()

# check the first 10 house price
boston.target[:10]

# add the field PRICE to bos
bos['PRICE'] = boston.target
bos.head()

from sklearn.linear_model import LinearRegression

# construct X by dropping target field
X = bos.drop('PRICE', axis=1)
X.head()

# Create a LR object
lm = LinearRegression()

# Fit the model
lm.fit(X,bos.PRICE)

# print the result of model fitting
print('Estimated intercept coefficient: ', lm.intercept_)
print('Number of coefficients:', len(lm.coef_))

# show the estimated coefficients of model fitting
pd.DataFrame(list(zip(X.columns,lm.coef_)),columns=['features','estimatedCoefficients'])

# plot the relationship between number of rooms and price
plt.scatter(bos.RM,bos.PRICE)

plt.xlabel("number of rooms")
plt.ylabel("House Price")
plt.title("Relationship between number of rooms and Price")
plt.show()

lm.predict(X)[:10]

# plot the relationship between actual price and predicted price
plt.scatter(bos.PRICE, lm.predict(X))
plt.xlabel("Price: $Y_i$")
plt.ylabel("Predicted Price: $\hat{Y_i}$")
plt.show()

# Clac Mean Square Error
MSE = np.mean((bos.PRICE-lm.predict(X))**2)
print(MSE)

# fit another model with only one feature RM, the MSE increases
lm1 = LinearRegression()
lm1.fit(X[['RM']],bos.PRICE)
mseRM = np.mean((bos.PRICE-lm1.predict(X[['RM']]))**2)
print(mseRM)





# Plot the fitted line

X_new = pd.DataFrame({'RM':[X.RM.min(),X.RM.max()]})
preds = lm1.predict(X_new)

plt.scatter(bos.RM,bos.PRICE)
plt.plot(X_new,preds,c='red', linewidth=2)
plt.xlabel("number of rooms")
plt.ylabel("House Price")
plt.title("Relationship between number of rooms and Price")
plt.show()


# data split for the model testing
X_train, X_test, Y_train, Y_test = sklearn.cross_validation.train_test_split(X, bos.PRICE, test_size=0.33, random_state=5)
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)