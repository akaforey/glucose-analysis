# from LSTM import LSTM
# import torch
# import pandas as pd
import pandas as pd
import numpy as np
import torch
# import torch.nn as nn
# import torch.optim as optim
# import math
# from sklearn.metrics import mean_squared_error
# from matplotlib import pyplot as plt
from LSTM import LSTM
from sklearn.preprocessing import MinMaxScaler


def main():
    input_dim = 1
    hidden_dim = 32
    num_layers = 2
    output_dim = 1
    num_epochs = 100
    model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)

    model.load_state_dict(torch.load(f'model_{num_epochs}e'), strict=False)

    min_glucose = pd.read_csv('min_glucose.csv')
    scaler = MinMaxScaler(feature_range=(-1, 1))
    min_glucose['Min Glucose'] = scaler.fit_transform(min_glucose['Min Glucose'].values.reshape(-1, 1))

    def load_data(stock, look_back):
        data_raw = stock.values  # convert to numpy array
        data = []

        # create all possible sequences of length look_back
        for index in range(len(data_raw) - look_back):
            data.append(data_raw[index: index + look_back])

        data = np.array(data)
        test_set_size = int(np.round(0.1 * data.shape[0]))
        train_set_size = data.shape[0] - test_set_size

        x_train = data[:train_set_size, :-1, :]
        y_train = data[:train_set_size, -1, :]

        x_test = data[train_set_size:, :-1]
        y_test = data[train_set_size:, -1, :]

        return [x_train, y_train, x_test, y_test]

    look_back = 60  # choose sequence length
    x_train, y_train, x_test, y_test = load_data(min_glucose, look_back)
    model.eval()
    print(model.forward(x_test))


if __name__ == "__main__":
    main()
