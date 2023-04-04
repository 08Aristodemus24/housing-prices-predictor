import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# Pipeline is akin to a Sequential class in tf where architecture of model is defined
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd

data = pd.read_csv('./CaliforniaHousing/cal_housing.data', sep=',', header=None)
print(data)

X, Y = data.loc[:, 0:7].to_numpy(), data.loc[:, 8].to_numpy()
print(f"{X[:5]}\n{Y[:5]}")

X_trains, X_tests, Y_trains, Y_tests = train_test_split(X, Y, test_size=0.3, random_state=0)
print('X_trains: {} \n'.format(X_trains))
print('Y_trains: {} \n'.format(Y_trains))
print('X_tests: {} \n'.format(X_tests))
print('Y_tests: {} \n'.format(Y_tests))

def plot_data(X_trains, X_tests):
    fig = plt.figure(figsize=(15, 10))
    axis = fig.add_subplot()

    axis.scatter(X_trains[:, 0], X_trains[:, 1], alpha=0.25, c='#4248f5', marker='p', label='training data')
    axis.scatter(X_tests[:, 0], X_tests[:, 1], alpha=0.25, c='#f542a1', marker='.', label='test data')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend()
    plt.show()

def analyze(X_trains, X_tests, n_row_plots, m_col_plots):
    # sees the range where each feature lies
    fig, axes = plt.subplots(n_row_plots, m_col_plots, figsize=(15, 10))
    fig.tight_layout(pad=1)

    # no. of instances and features
    num_instances = X_trains.shape[0]
    num_features = X_trains.shape[1]

    # feature names
    feature_names = ["median income", "median house age", "avg no. of rooms/household", "avg no. of bedrooms/household", 
    "block group population", "avg no of household members", "block group latitude", "block group longitude"]
    
    zeros = np.zeros((num_instances,))
    
    # how do I keep the title without it being removed after plt.show()
    for feature_col_i, axis in enumerate(axes.flat):
        # print(feature_col_i)
        curr_feature = X_trains[:, feature_col_i].reshape(-1)
        # print(curr_feature)
        # print(curr_feature.shape)
        axis.scatter(curr_feature, zeros, alpha=0.25, marker='p', c='#036bfc')
        # print(feature_names[feature_col_i])
        
        # if feature_col_i % 8 == 0:
        #     axis.set_title(feature_names[feature_col_i])
        axis.set_title(f"feature [{feature_col_i}]")
        
    plt.show()

plot_data(X_trains, X_tests)
analyze(X_trains, X_tests, 2, 4)


# Linear Regression Model
# standardize data or normalize both training and test data
# note that self.fit_transform() should only be used absolutely
# for the training dataset only
scaler = StandardScaler()
X_trains_normed = scaler.fit_transform(X_trains)

# note that self.transform() is used specifically 
# for both test and development data sets only
X_tests_normed = scaler.transform(X_tests)

plot_data(X_trains_normed, X_tests_normed)
analyze(X_trains_normed, X_tests_normed, 4, 2)



# train model
model = LinearRegression()
model.fit(X_trains_normed, Y_trains)

# get results of training data
Y_preds = model.predict(X_trains_normed)
print(f"mean squared error for training data: {mean_squared_error(Y_trains, Y_preds)}")
print(f"root mean squared error for training data: {math.sqrt(mean_squared_error(Y_trains, Y_preds))}")

# get results of training data
Y_preds = model.predict(X_tests_normed)
print(f"mean squared error for test data: {mean_squared_error(Y_tests, Y_preds)}")
print(f"root mean squared error for test data: {math.sqrt(mean_squared_error(Y_tests, Y_preds))}")

# train a linear regression model with L2
# regularization to see if error goes down
model = Ridge(alpha=1.0)
model.fit(X_trains_normed, Y_trains)

# get results of training data
Y_preds = model.predict(X_trains_normed)
print(f"mean squared error for training data: {mean_squared_error(Y_trains, Y_preds)}")
print(f"root mean squared error for training data: {math.sqrt(mean_squared_error(Y_trains, Y_preds))}")

# get results of training data
Y_preds = model.predict(X_tests_normed)
print(f"mean squared error for test data: {mean_squared_error(Y_tests, Y_preds)}")
print(f"root mean squared error for test data: {math.sqrt(mean_squared_error(Y_tests, Y_preds))}")



# Polynomial Regression
# use a function now that engineers new features out of the dataset
# such that it mimics a polynomial equation with a degree in this case of 2
# since data is parabolic
poly = PolynomialFeatures(degree=2, include_bias=False)
X_trains_enged = poly.fit_transform(X_trains)
X_tests_enged = poly.transform(X_tests)

# view and see the shape of teh dataset which as observed 
# now contains 44 features
print(X_trains_enged, X_trains_enged.shape)
print(X_tests_enged, X_tests_enged.shape)

# standardize/normalize again the data both training and test data
scaler = StandardScaler()
X_trains_normed = scaler.fit_transform(X_trains_enged)
X_tests_normed = scaler.transform(X_tests_enged)

plot_data(X_trains_normed, X_tests_normed)
analyze(X_trains_normed, X_tests_normed, 11, 4)

# instantiate linear regression model again
# and fit a line to the data
model = LinearRegression()
model.fit(X_trains_normed, Y_trains)

# get results of training data
Y_preds = model.predict(X_trains_normed)
print(f"mean squared error for training data: {mean_squared_error(Y_trains, Y_preds) / 2}")
print(f"root mean squared error for training data: {math.sqrt(mean_squared_error(Y_trains, Y_preds))}")

# get results of training data
Y_preds = model.predict(X_tests_normed)
print(f"mean squared error for test data: {mean_squared_error(Y_tests, Y_preds) / 2}")
print(f"root mean squared error for test data: {math.sqrt(mean_squared_error(Y_tests, Y_preds))}")

# instead of instantiating PolynomialFeatures, StandardScaler, and
# LinearRegression and getting each their outputs to pass to the next class
# we can use Pipeline to reduce redundancy 

# this block of code is akin or equivalent to the aforementioned code
# that implemented a polynomial regression model
poly = PolynomialFeatures(degree=2, include_bias=False)
scaler = StandardScaler()
model = Ridge(alpha=1.0)

poly_model = Pipeline([
    ("engineered features", poly),
    ("input normalizer|standardizer", scaler),
    ("linear regression model", model),
])

poly_model.fit(X_trains, Y_trains)



# instead of LinearRegression we use Ridge to train a linear regression 
# model with L2 regularization to see if error goes down
model = Ridge(alpha=1.0)
model.fit(X_trains_normed, Y_trains)

# get results of training data
Y_preds = model.predict(X_trains_normed)
print(f"mean squared error for training data: {mean_squared_error(Y_trains, Y_preds)}")
print(f"root mean squared error for training data: {math.sqrt(mean_squared_error(Y_trains, Y_preds))}")

# get results of training data
Y_preds = model.predict(X_tests_normed)
print(f"mean squared error for test data: {mean_squared_error(Y_tests, Y_preds)}")
print(f"root mean squared error for test data: {math.sqrt(mean_squared_error(Y_tests, Y_preds))}")
# In conclusion what can be deduced from the aforementioned code is that when engineering new features or turning the data in such that it mimics a polynomial equation rather than a linear equation, the process is always first and foremost, and then comes only normalizing the newly engineered features and then training the model
# 


