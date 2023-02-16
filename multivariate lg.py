import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston



class MultivariateLinearRegression:
    def __init__(self, dataset):
        self.features = dataset['data']
        self.output = dataset['target']
        self.num_features = dataset['data'].shape[1]
        self.num_instances = dataset['data'].shape[0]
        self.alpha = 0.001
        
    def gradientDescent(self):
        print(self.features)
        print(self.output)
        print(self.num_features)
        print(self.num_instances)

        # remove the extra theta value of index 0 for now
        theta_values = [0 for _ in range(self.num_features)]
        
        # every feature must be at most greater than -1
        # and less than 1
        last_index = len(theta_values)

        for iteration in range(1000):

            saved_theta = []
            if iteration == 0 or (iteration % 200 == 0):
                print(theta_values)
                print(self.cost(theta_values))

            for index, theta in enumerate(theta_values):
                if index == last_index:
                    # summation of loss values
                    temp_i = 0
                    for i, instance in enumerate(self.features):
                        temp_f = 0
                        
                        # append 1 to every instance, to cover the feature 1
                        new_instance = list(instance)
                        new_instance.append(1)

                        # summation of all features multiplied by theta values
                        for j, feature in enumerate(new_instance):
                            # start from index 0 of theta values
                            temp_f += feature * theta_values[j]
                        
                        # current feature with the same index as the theta value
                        temp_i += temp_f - self.output[i] * 1 
                    
                    temp_i /= self.num_instances
                    saved_theta.append(theta - (self.alpha * temp_i))
                else:
                    # summation of loss values
                    temp_i = 0
                    for i, instance in enumerate(self.features):
                        temp_f = 0
                        
                        # append 1 to every instance, to cover the feature 1
                        new_instance = list(instance)
                        new_instance.append(1)

                        # summation of all features multiplied by theta values
                        for j, feature in enumerate(new_instance):
                            # start from index 0 of theta values
                            temp_f += feature * theta_values[j]

                        # multiply the feature with the same index as the current
                        # theta value, in the current instance
                        temp_i += temp_f - self.output[i] * instance[index]
                    
                    temp_i /= self.num_instances
                    saved_theta.append(theta - (self.alpha * temp_i))

            # modify the newly created elements
            for i in range(len(theta_values)):
                theta_values[i] = saved_theta[i]

            

    def cost(self, theta_values):
        temp_i = 0
        for index, instance in enumerate(self.features):
            temp_f = 0
            new_instance = list(instance)
            new_instance.append(1)
            for i, features in enumerate(new_instance):
                temp_f += features * theta_values[i]

            temp_i += (temp_f - self.output[index]) ** 2

        temp_i /= (2 * self.num_instances)
        return temp_i



if __name__ == "__main__":
    boston_dataset = load_boston()
    mlr = MultivariateLinearRegression(boston_dataset)
    mlr.gradientDescent()

    # problem:

    # idea:

    # method:

    # samples/cases:

    # to figure out:
    # since theta 0 is at the end
    # we have to append feature X_0 to the end also

    # we don't include theta_0 * 1 + summation of all other
    # theta values * current feature
