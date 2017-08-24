import csv
import pandas
import numpy
from sklearn import linear_model, preprocessing
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cross_validation import KFold, cross_val_score

wine = pandas.read_csv('wine.csv', index_col = 0)
wine.loc[:, wine.columns != 'test'] = wine.loc[:, wine.columns != 'test'].apply(preprocessing.scale)

wine_train = wine[wine['test'] == False].drop('test', axis = 1)
wine_train_x = wine_train.drop('quality', axis = 1)
wine_train_y = wine_train[['quality']]

wine_test = wine[wine['test'] == True].drop('test', axis = 1)
wine_test_x = wine_test.drop('quality', axis = 1)
wine_test_y = wine_test[['quality']]

linearRegression = LinearRegression(fit_intercept = False)
linearRegression.fit(wine_train_x, wine_train_y)
linear_predictions = linearRegression.predict(wine_test_x)

# Stats
linearRegression.coef_
mean_squared_error(wine_test_y, linear_predictions)
r2_score(wine_test_y, linear_predictions)

ridgeRegression = RidgeCV(fit_intercept = False, cv = 10)
ridgeRegression.fit(wine_train_x, wine_train_y)
ridge_predictions = ridgeRegression.predict(wine_test_x)
mean_squared_error(wine_test_y, ridge_predictions)
r2_score(wine_test_y, ridge_predictions)

lassoRegression = LassoCV(fit_intercept = False, cv = 10)
lassoRegression.fit(wine_train_x, wine_train_y)
lasso_predictions = lassoRegression.predict(wine_test_x)
mean_squared_error(wine_test_y, lasso_predictions)
r2_score(wine_test_y, lasso_predictions)
