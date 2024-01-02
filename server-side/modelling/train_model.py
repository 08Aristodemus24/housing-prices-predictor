import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from argparse import ArgumentParser

class MultivariateLinearRegression:
    def __init__(self, X=None, Y=None, epochs=10000, learning_rate=0.003, lambda_=0):
        if (X is not None) and (Y is not None):
            self.X_trains, self.X_cross, self.Y_trains, self.Y_cross = train_test_split(X, Y, test_size=0.3, random_state=0, shuffle=True)

            self.num_features = self.X_trains.shape[1]
            self.train_num_instances = self.X_trains.shape[0]
            self.cross_num_instances = self.X_cross.shape[0]

            
            self.epochs = epochs
            self.learning_rate = learning_rate
            self.lambda_ = lambda_
            
            self.history = {
                'history': {
                    'mean_squared_error': [],
                    'val_mean_squared_error': [],
                    'root_mean_squared_error': [],
                    'val_root_mean_squared_error': []
                }
            }

        self._curr_theta = np.zeros((None))
        self._curr_beta = 0.0
        self._mean = np.zeros((None))
        self._std_dev = np.zeros((None))

    @property
    def theta(self):
        return self._curr_theta
    
    @property
    def beta(self):
        return self._curr_beta
    
    @property
    def mean(self):
        return self._mean

    @property
    def std_dev(self):
        return self._std_dev

    @theta.setter
    def theta(self, new_theta):
        self._curr_theta = new_theta

    @beta.setter
    def beta(self, new_beta):
        self._curr_beta = new_beta

    @mean.setter
    def mean(self, new_mean):
        self._mean = new_mean

    @std_dev.setter
    def std_dev(self, new_std):
        self._std_dev = new_std

    def view_data(self):
        print('X_trains: {} \n'.format(self.X_trains))
        print('Y_trains: {} \n'.format(self.Y_trains))
        print('X_cross: {} \n'.format(self.X_cross))
        print('Y_cross: {} \n'.format(self.Y_cross))
        print('number of features: {} \n'.format(self.num_features))
        print('number of training instances: {} \n'.format(self.train_num_instances))
        print('number of cross validation instances: {} \n'.format(self.cross_num_instances))

    def analyze(self):
        # see where each feature lies
        # sees the range where each feature lies
        fig, axes = plt.subplots(4, 2, figsize=(15, 10))
        fig.tight_layout(pad=1)

        # no. of instances and features
        num_instances = self.X_trains.shape[0]
        num_features = self.X_trains.shape[1]

        # feature names
        feature_names = ["longitude", "latitude", "avg. housing age", "total no. of rooms", 
        "total no. of bedrooms", "population", "households", "avg. income"]
        
        zeros = np.zeros((num_instances,))
        
        # how do I keep the title without it being removed after plt.show()
        for feature_col_i, axis in enumerate(axes.flat):
            # print(feature_col_i)
            curr_feature = self.X_trains[:, feature_col_i].reshape(-1)
            # print(curr_feature)
            # print(curr_feature.shape)
            
            axis.scatter(curr_feature, zeros, alpha=0.25, marker='p', c='#036bfc')
            # print(feature_names[feature_col_i])

            axis.set_title(feature_names[feature_col_i])
            # axis.set_title(f"feature [{feature_col_i}]")
            
        plt.show()

    def plot_all_data(self):
        pass

    def plot_data(self):
        fig = plt.figure(figsize=(15, 10))
        axis = fig.add_subplot()

        axis.scatter(self.X_trains[:, 0], self.X_trains[:, 1], alpha=0.25, c='#4248f5', marker='p', label='training data')
        axis.scatter(self.X_cross[:, 0], self.X_cross[:, 1], alpha=0.25, c='#f542a1', marker='.', label='test data')
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.legend()
        plt.show()
    
    def mean_normalize(self):
        # for feature_col_i in range(self.num_features):
        #     # we ought to normalize once each data split is established to prevent data leakage
        #     self.X_trains[:, feature_col_i] = (self.X_trains[:, feature_col_i] - np.average(self.X_trains[:, feature_col_i])) / np.std(self.X_trains[:, feature_col_i])
        #     self.X_cross[:, feature_col_i] = (self.X_cross[:, feature_col_i] - np.average(self.X_trains[:, feature_col_i])) / np.std(self.X_trains[:, feature_col_i])

        # instantiate a standard scaler object
        scaler = StandardScaler()

        # "train" the standard scaler on the training data
        self.X_trains = scaler.fit_transform(self.X_trains)

        # normalize the cross validation data on the 
        # training datas mean and standard deviation
        self.X_cross = scaler.transform(self.X_cross)

        # save also the std deviation and mean for later testing
        self.mean = scaler.mean_
        self.std_dev = scaler.scale_

    def init_params(self):
        self.theta = np.random.rand(self.num_features,)

    def fit(self, show_logs=True):
        # initialize coefficients
        self.init_params()

        # show mean and std
        print(f"mean and std: {self.mean} {self.std_dev}")

        # run algorithm for n epochs
        for epoch in range(self.epochs):
            # calculate cost per epoch for training and testing
            train_mse = self.J_MSE(self.X_trains, self.Y_trains)
            cross_mse = self.J_MSE(self.X_cross, self.Y_cross)
            train_rmse = self.J_RMSE(train_mse)
            cross_rmse = self.J_RMSE(cross_mse)

            if epoch % 1000 == 0:
                print('current theta: {} \n'.format(self.theta))
                print('current beta: {} \n'.format(self.beta))
                print('current train MSE: {} \n'.format(train_mse))
                print('current train RMSE: {} \n'.format(train_rmse))
                print('current cross MSE: {} \n'.format(cross_mse))
                print('current cross RMSE: {} \n'.format(cross_mse))

            # save each data splits cost
            self.history['history']['mean_squared_error'].append(train_mse)
            self.history['history']['val_mean_squared_error'].append(cross_mse)
            self.history['history']['root_mean_squared_error'].append(train_rmse)
            self.history['history']['val_root_mean_squared_error'].append(cross_rmse)

            # calculate gradients and then update coefficients
            self.optimize(show_logs=show_logs)
        
        print('DONE')

    def optimize(self, show_logs):
        params = self.J_prime()
        new_theta = self.theta - (self.learning_rate * params['dw'])
        new_beta = self.beta - (self.learning_rate * params['db'])
        self.theta = new_theta
        self.beta = new_beta

        if show_logs == True:
            error = params['error']
            print(f'error: {error}')

    def linear(self, X):
        return np.dot(X, self.theta) + self.beta

    def J_MSE(self, X, Y):
        num_instances = X.shape[0]

        error = self.linear(X) - Y
        return (np.dot(error.T, error) / (2 * num_instances)) + ((self.lambda_ * np.dot(self.theta, self.theta.T)) / (2 * num_instances))

    def J_RMSE(self, mse):
        return np.sqrt(mse)

    def J_prime(self):
        error = self.linear(self.X_trains) - self.Y_trains
        dw =  (np.dot(self.X_trains.T, error) / self.train_num_instances) + ((self.lambda_ * self.theta) / self.train_num_instances)
        db = np.sum(error, keepdims=True) / self.train_num_instances

        return {
            'error': error,
            'dw': dw,
            'db': db
        }
    
    def validate(self, is_training_set: bool=True):
        # predicts both training set and test set. If RMSE or root mean squared 
        # error is 0 then prediction for new data and train data is 0
        return (self.linear(self.X_trains), self.Y_trains) if is_training_set is True else (self.linear(self.X_cross), self.Y_cross)
    
    def compare(self):
        Y_preds = self.linear(self.X_trains)
        for i in range(self.train_num_instances):
            print(Y_preds[i], self.Y_trains[i])

    def predict(self, X):
        # normalize on training mean and standard dev first
        X = (X - self.mean) / self.std_dev

        # predict then return prediction value
        return self.linear(X)

    def save_weights(self):
        meta_data = {
            'non-bias': self.theta.tolist(),
            'bias': self.beta.tolist()[0],
            'mean': self.mean.tolist(),
            'std_dev': self.std_dev.tolist()
        }

        # if directory weights does not already exist create 
        # directory and save weights there
        if os.path.exists('./weights') != True:
            os.mkdir('./weights')
        
        with open('./weights/meta_data.json', 'w') as out_file:
            json.dump(meta_data, out_file)
            out_file.close()
        
    def load_weights(self, file_path: str):
        with open(file_path, "r") as in_file:
            meta_data = json.load(in_file)
            in_file.close()

        self.theta = np.array(meta_data['non-bias'])
        self.beta = meta_data['bias']
        self.mean = np.array(meta_data['mean'])
        self.std_dev = np.array(meta_data['std_dev'])




def plot_train_cross_costs(model):
    metrics_to_use = list(model.history['history'].items())
    epochs = model.epochs

    styles = [
        ('p:', '#f54949'), 
        ('h-', '#f59a45'), 
        ('o--', '#afb809'), 
        ('x:','#51ad00'),]
    
    
    for index in range(0, len(metrics_to_use) - 1, 2):
        plt.figure(figsize=(15, 10))
        metric_indeces = (index, index + 1)

        for j, (key, value) in enumerate(metrics_to_use[metric_indeces[0]: metric_indeces[1] + 1]):
            plt.plot(np.arange(epochs) + 1, value, styles[metric_indeces[j]][0], color=styles[metric_indeces[j]][1], alpha=0.1, label=key)

        plt.title(f'cost per epoch for training and cross validation data splits')
        plt.ylabel('metric value')
        plt.xlabel('epochs')
        plt.legend()

        # save figure
        plt.savefig(f'./figures & images/{metrics_to_use[index][0]} per epoch for training and cross validation data splits.png')
        plt.show()
        plt.figure().clear()



def view_model_metric_values(Y_tests, Y_preds):
    mse = mean_squared_error(Y_tests, Y_preds)
    rmse = math.sqrt(mse)
    print('mse: {:.2%}'.format(mse))
    print('rmse: {:.2%}'.format(rmse))



if __name__ == "__main__":
    # optional arguments
    parser = ArgumentParser()
    parser.add_argument('--show_logs', type=str, default=True, help='flag whether to show coeffcieints every 1000 epochs during training')

    # reading data
    data = pd.read_csv('./CaliforniaHousing/cal_housing.data', sep=',', header=None)
    print(data)

    # preprocessing X and Y data
    X, Y = data.loc[:, 0:7].to_numpy(), data.loc[:, 8].to_numpy()
    print(f"{X[:5]}\n{Y[:5]}")

    # instantiating model
    model = MultivariateLinearRegression(X, Y)

    # viewing and analyzing features of data through visualization
    model.view_data()
    model.analyze()
    model.plot_data()
    
    # normalizing data for faster and more accurate training
    model.mean_normalize()

    # viewing and analyzing features of data 
    # through visualization after normalization
    model.view_data()
    model.analyze()
    model.plot_data()

    # training model
    model.fit(show_logs=False)

    # validating model
    Y_preds, Y_tests = model.validate(is_training_set=True)
    
    # viewing metric values on training and cross validation data
    view_model_metric_values(Y_tests, Y_preds)
    plot_train_cross_costs(model)

    # saving weights for deployment and testing
    model.save_weights()

    