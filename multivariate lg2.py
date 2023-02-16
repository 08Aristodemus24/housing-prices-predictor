import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston



class MultivariateLinearRegression:
    def __init__(self, dataset):
        # concatenate feature matrix to matrix of ones
        new_dataset = np.concatenate(
            (np.ones((dataset['data'].shape[0], 1)), dataset['data']), axis=1
        )

        self.features = new_dataset
        self.output = dataset['target']
        self.num_features = new_dataset.shape[1]
        self.num_instances = new_dataset.shape[0]
        self.alpha = 0.001
        
    def gradientDescent(self):
        print(self.features)
        print(self.output)
        print(self.num_features)
        print(self.num_instances)

        # initialize theta values to zero based on length
        # of new feature matrix
        theta_values = np.zeros(self.num_features)


        for _ in range(1000):
            if _ % 250 == 0:
                print(self.cost(theta_values))
            # save new theta values in temporary array
            temp_theta = np.zeros(self.num_features)
            for i, theta in enumerate(theta_values):
                temp_cost = 0

                # loop through all instances 
                for j, instance in enumerate(self.features):
                    summation = np.dot(instance, theta_values)
                    if _ % 250 == 0:
                        print(summation)
                    error = summation - self.output[j]

                    result = error * instance[i]

                    temp_cost += result

                    # if _ % 100 == 0:
                    #     print('summation: ', summation)
                    #     print('error: ', error)
                    #     print('result: ', result)

                temp_cost /= self.num_instances
                new_theta = theta - (self.alpha * temp_cost)

                # fill temporary list with new theta values
                temp_theta[i] = new_theta

            for i in range(len(temp_theta)):
                theta_values[i] = temp_theta[i]

    def cost(self, theta_values):
        cost = 0
        for i, instance in enumerate(self.features):
            error = (np.dot(instance, theta_values) - self.output[i])
            loss = error ** 2
            cost += loss
        return cost


if __name__ == "__main__":
    boston_dataset = load_boston()
    mlr = MultivariateLinearRegression(boston_dataset)
    mlr.gradientDescent()