import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

class MultivariateLinearRegression:
    def __init__(self, X, Y, epochs=10000, learning_rate=0.003, lambda_=0):
        self.X_trains, self.X_cross, self.Y_trains, self.Y_cross = train_test_split(X, Y, test_size=0.3, random_state=0)

        self.num_features = self.X_trains.shape[1]
        self.train_num_instances = self.X_trains.shape[0]
        self.cross_num_instances = self.X_cross.shape[0]

        self._curr_theta = np.zeros((self.num_features))
        self._curr_beta = 0
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

    @property
    def theta(self):
        return self._curr_theta
    
    @property
    def beta(self):
        return self._curr_beta
    
    @theta.setter
    def theta(self, new_theta):
        self._curr_theta = new_theta

    @beta.setter
    def beta(self, new_beta):
        self._curr_beta = new_beta

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
        feature_names = ["median income", "median house age", "avg no. of rooms/household", "avg no. of bedrooms/household", 
        "block group population", "avg no of household members", "block group latitude", "block group longitude"]
        
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
        for feature_col_i in range(self.num_features):
            # we ought to normalize once each data split is established to prevent data leakage
            self.X_trains[:, feature_col_i] = (self.X_trains[:, feature_col_i] - np.average(self.X_trains[:, feature_col_i])) / np.std(self.X_trains[:, feature_col_i])
            self.X_cross[:, feature_col_i] = (self.X_cross[:, feature_col_i] - np.average(self.X_cross[:, feature_col_i])) / np.std(self.X_cross[:, feature_col_i])

    def fit(self):
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
            self.optimize()
        
        print('DONE')

    def optimize(self):
        params = self.J_prime()
        # print('dw:', params['dw'])
        new_theta = self.theta - (self.learning_rate * params['dw'])
        new_beta = self.beta - (self.learning_rate * params['db'])
        self.theta = new_theta
        self.beta = new_beta

    def linear(self, X):
        return np.dot(X, self.theta) + self.beta

    def J_MSE(self, X, Y):
        num_instances = X.shape[0]

        loss = self.linear(X) - Y
        return (np.dot(loss.T, loss) / (2 * num_instances)) + ((self.lambda_ * np.dot(self.theta, self.theta.T)) / (2 * num_instances))

    def J_RMSE(self, mse):
        return np.sqrt(mse)

    def J_prime(self):
        error = self.linear(self.X_trains) - self.Y_trains
        dw =  (np.dot(self.X_trains.T, error) / self.train_num_instances) + ((self.lambda_ * self.theta) / self.train_num_instances)
        db = np.sum(error, keepdims=True) / self.train_num_instances
        
        return {
            'dw': dw,
            'db': db
        }
    
    def predict(self, is_training_set=True):
        # predicts both training set and test set. If RMSE or root mean squared 
        # error is 0 then prediction for new data and train data is 0
        return (self.linear(self.X_trains), self.Y_trains) if is_training_set is True else (self.linear(self.X_cross), self.Y_cross)
    
    def compare(self):
        Y_preds = self.predict(self.X_trains)
        for i in range(self.train_num_instances):
            print(Y_preds[i], self.Y_trains[i])



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



def view_model_metric_values(model):
    Y_preds, Y_tests = model.predict(is_training_set=True)
    mse = mean_squared_error(Y_tests, Y_preds)
    rmse = math.sqrt(mse)
    print('mse: {:.2%}'.format(mse))
    print('rmse: {:.2%}'.format(rmse))



if __name__ == "__main__":
    data = pd.read_csv('./CaliforniaHousing/cal_housing.data', sep=',', header=None)
    print(data)

    X, Y = data.loc[:, 0:7].to_numpy(), data.loc[:, 8].to_numpy()
    print(f"{X[:5]}\n{Y[:5]}")

    model = MultivariateLinearRegression(X, Y)

    model.view_data()
    model.analyze()
    model.plot_data()
    
    model.mean_normalize()
    model.analyze()
    model.plot_data()

    model.fit()
    
    
    view_model_metric_values(model)
    plot_train_cross_costs(model)