import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
import pandas as pd



class MultivariateLinearRegression:
    def __init__(self, dataset):
        # convert numpy array to data frame
        data_frame = pd.DataFrame(
            data = dataset['data'], 
            columns = dataset['feature_names'])

        # initialize the ones feature
        ones_feature = pd.DataFrame(np.ones((data_frame.shape[0], 1)))

        # concatenate feature matrix to matrix of ones
        # then convert to numpy data frame
        frames = [ones_feature, data_frame]
        new_dataset = pd.concat(frames, axis=1).to_numpy()

        self.input = new_dataset
        self.output = dataset['target']
        self.num_features = new_dataset.shape[1]
        self.num_instances = new_dataset.shape[0]
        self.alpha = 0.001
        self.cltd_costs = np.empty(0, dtype=float)
        
    def viewPreparedData(self):
        print('input: ')
        print(self.input)
        print('\n')

        print('output: ')
        print(self.output)

        print('output length: ')
        print(self.output.shape[0])
        print('\n')

        print('number of features: ')
        print(self.num_features)
        print('\n')

        print('number of instances: ')
        print(self.num_instances)
        print('\n')

        # initialize theta values to zero based on length
        # of new feature matrix
        theta_values = np.zeros(self.num_features)
        print('theta values: ')
        print(theta_values)


    
    def cost(self, theta_values):
        loss = 0
        for i, instance in enumerate(self.input):
            # reset temp_out to 0 to calculate new summation of
            # thetas and features in the next instance
            temp_out = 0
            for j, feature in enumerate(instance):
                temp_out += (theta_values[j] * feature)

            # calculate error of each instance (y^ - y)
            # calculate loss of each instance (y^ - y) ** 2
            # use the current instance to access output attribute
            # array element in this insance
            loss += (temp_out - self.output[i]) ** 2

        # calculate cost once all losses are summed and divided by 2 * m
        cost = loss / 2 * self.num_instances
        self.cltd_costs = np.append(self.cltd_costs, cost)
        return cost

    def gradientDescent(self, coeffs):
        # calculate new coefficients
        # append old coefficients to record change overtime
        # append as well the errors overtime
        # return these new coefficients
        # set sum values initially to 0
        # pass previous coefficients in this method
        temp_coeffs = np.copy(coeffs)
        new_coeffs = np.zeros(self.num_features)
        
        # # update each coefficient by looping through each coefficient
        for index, curr_coeff in enumerate(temp_coeffs):
            loss = 0
            for i, instance in enumerate(self.input):

                # calculates summation of the product of features and
                # coefficients in an instance/row
                temp_out = 0
                for j, feature in enumerate(instance):
                    temp_out += (temp_coeffs[j] * feature)

                # calculates the summation of the error multiplied by the
                # feature at a current instance with the same index as its pair coeff
                loss += (temp_out - self.output[i]) * instance[index]

            # # calculates the new value for each coefficient
            new_coeffs[index] = curr_coeff - (self.alpha * (loss / self.num_instances))

        return new_coeffs
        

    def fit(self):
        # initial coefficients will have 14 elements
        # all set to zero
        coeffs = np.zeros(self.num_features)

        # initial cost will be appended
        cost = self.cost(coeffs)

        steps = 50
        for _ in range(steps + 1):
            if _ % 10 == 0:
                print('theta values: ', coeffs)
                print('iteration: ', _)
                print('cost: ', cost)

            coeffs = self.gradientDescent(coeffs)
            cost = self.cost(coeffs)

            

    def asJApproachesZero(self):
        print(self.cltd_costs)
        
        fig = plt.figure()
        ax = fig.add_subplot(111)

        # for x and y values respectively
        ax.scatter([0, 1, 2, 3, 4, 5, 6, 7], [20, 5, 1, 0.9, 0.5, 0.25, 0.125, 0.0625], c='orange', alpha=1)
        ax.plot([0, 1, 2, 3, 4, 5, 6, 7], [20, 5, 1, 0.9, 0.5, 0.25, 0.125, 0.0625], c='blue', alpha=0.25)
        # use the number of iterations to represent the x axis
        # use the costs per iteration to represent the y axis
        ax.set_xlabel('number of iterations')
        ax.set_ylabel('cost or J(θ)')

        plt.show()


if __name__ == "__main__":
    boston_dataset = load_boston()
    mlr = MultivariateLinearRegression(boston_dataset)
    # mlr.viewPreparedData()
    mlr.fit()
    mlr.asJApproachesZero()