import pandas as pd

from train_model import MultivariateLinearRegression, view_model_metric_values



if __name__ == "__main__":
    data = pd.read_csv('./CaliforniaHousing/cal_housing.data', sep=',', header=None)
    print(data)

    X, Y = data.loc[:, 0:7].to_numpy(), data.loc[:, 8].to_numpy()
    print(f"{X[:5]}\n{Y[:5]}")

    model = MultivariateLinearRegression()
    model.load_weights('./weights/coefficients.json')

    Y_preds = model.predict(X)
    view_model_metric_values(Y, Y_preds)