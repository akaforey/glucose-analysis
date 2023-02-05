import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
from LSTM import LSTM
from sklearn.preprocessing import MinMaxScaler


def main():
    # Load the data into a pandas DataFrame
    df = pd.read_csv('../data/DanielleFerreira_glucose_2-3-2023.csv', header=1, low_memory=False)

    pd.set_option('display.max_columns', None)

    # Preprocess the data
    # ... (fill in missing values, normalize, etc.)
    low_threshold = 60
    high_threshold = 200

    glucose_readings = df.query('`Record Type` in [0, 1]')
    glucose_readings['Glucose'] = glucose_readings['Historic Glucose mg/dL'].where(glucose_readings['Record Type'] == 0,
                                                                                   other=glucose_readings[
                                                                                       'Scan Glucose mg/dL'], inplace=False)
    glucose_readings = glucose_readings[['Device Timestamp', 'Glucose']]
    glucose_readings.reset_index(drop=True, inplace=True)

    glucose_readings['Device Timestamp'] = pd.to_datetime(glucose_readings['Device Timestamp'], )
    glucose_readings = glucose_readings.set_index('Device Timestamp', inplace=False)
    hourly_glucose = glucose_readings.resample('1H').mean()
    hourly_glucose['Min Glucose'] = glucose_readings.resample('1H').min()['Glucose']
    hourly_glucose['Max Glucose'] = glucose_readings.resample('1H').max()['Glucose']
    hourly_glucose['Low Event'] = (hourly_glucose['Min Glucose'] < low_threshold).astype(float)
    # hourly_glucose['High Event'] = (hourly_glucose['Max Glucose'] > high_threshold).astype(float)
    # print(hourly_glucose.head())
    # min_val = hourly_glucose['Min Glucose'].min()
    # max_val = hourly_glucose['Max Glucose'].max()

    # hourly_glucose['Glucose'] = (hourly_glucose['Glucose']-min_val)/(max_val-min_val)
    # hourly_glucose['Min Glucose'] = (hourly_glucose['Min Glucose']-min_val)/(max_val-min_val)
    # hourly_glucose['Max Glucose'] = (hourly_glucose['Max Glucose']-min_val)/(max_val-min_val)

    print(hourly_glucose.head())
    # print(hourly_glucose['Min Glucose'].min())
    # print(hourly_glucose['Max Glucose'].min())
    # print(hourly_glucose['Glucose'].min())
    #
    # print(hourly_glucose['Min Glucose'].max())
    # print(hourly_glucose['Max Glucose'].max())
    # print(hourly_glucose['Glucose'].max())

    # LSTM Training

    scaler = MinMaxScaler(feature_range=(-1, 1))

    min_glucose = hourly_glucose[['Min Glucose']]
    min_glucose = min_glucose.fillna(method='ffill')
    min_glucose.to_csv('min_glucose.csv')
    min_glucose['Min Glucose'] = scaler.fit_transform(min_glucose['Min Glucose'].values.reshape(-1, 1))
    min_glucose.info()



    def load_data(stock, look_back):
        data_raw = stock.values  # convert to numpy array
        data = []

        # create all possible sequences of length look_back
        for index in range(len(data_raw) - look_back):
            data.append(data_raw[index: index + look_back])

        data = np.array(data)
        test_set_size = int(np.round(0.1 * data.shape[0]))
        train_set_size = data.shape[0] - (test_set_size)

        x_train = data[:train_set_size, :-1, :]
        y_train = data[:train_set_size, -1, :]

        x_test = data[train_set_size:, :-1]
        y_test = data[train_set_size:, -1, :]

        return [x_train, y_train, x_test, y_test]


    look_back = 60  # choose sequence length
    x_train, y_train, x_test, y_test = load_data(min_glucose, look_back)
    print('x_train.shape = ', x_train.shape)
    print('y_train.shape = ', y_train.shape)
    print('x_test.shape = ', x_test.shape)
    print('y_test.shape = ', y_test.shape)

    # make training and test sets in torch
    x_train = torch.from_numpy(x_train).type(torch.Tensor)
    x_test = torch.from_numpy(x_test).type(torch.Tensor)
    y_train = torch.from_numpy(y_train).type(torch.Tensor)
    y_test = torch.from_numpy(y_test).type(torch.Tensor)

    print(y_train.size(), x_train.size())

    # Build model
    #####################
    input_dim = 1
    hidden_dim = 32
    num_layers = 2
    output_dim = 1

    model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)

    loss_fn = torch.nn.MSELoss()

    optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
    print(model)
    print(len(list(model.parameters())))
    for i in range(len(list(model.parameters()))):
        print(list(model.parameters())[i].size())

    # Train model
    #####################
    num_epochs = 100
    hist = np.zeros(num_epochs)

    # Number of steps to unroll
    seq_dim = look_back - 1

    for t in range(num_epochs):
        # Initialise hidden state
        # Don't do this if you want your LSTM to be stateful
        # model.hidden = model.init_hidden()

        # Forward pass
        y_train_pred = model(x_train)

        loss = loss_fn(y_train_pred, y_train)
        if t % 10 == 0 and t != 0:
            print("Epoch ", t, "MSE: ", loss.item())
        hist[t] = loss.item()

        # Zero out gradient, else they will accumulate between epochs
        optimiser.zero_grad()

        # Backward pass
        loss.backward()

        # Update parameters
        optimiser.step()

    torch.save(model.state_dict(), f'model_{num_epochs}e')

    # make predictions
    y_test_pred = model(x_test)

    # invert predictions
    # y_train_pred = y_train_pred.detach().numpy()
    # y_train = y_train.detach().numpy()
    # y_test_pred = y_test_pred.detach().numpy()
    # y_test = y_test.detach().numpy()

    # invert predictions
    y_train_pred = scaler.inverse_transform(y_train_pred.detach().numpy())
    y_train = scaler.inverse_transform(y_train.detach().numpy())
    y_test_pred = scaler.inverse_transform(y_test_pred.detach().numpy())
    y_test = scaler.inverse_transform(y_test.detach().numpy())

    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(y_train[:, 0], y_train_pred[:, 0]))
    print('Train Score: %.2f RMSE' % trainScore)
    testScore = math.sqrt(mean_squared_error(y_test[:, 0], y_test_pred[:, 0]))
    print('Test Score: %.2f RMSE' % testScore)

    # Visualising the results
    figure, axes = plt.subplots(figsize=(15, 6))
    axes.xaxis_date()

    axes.plot(min_glucose[len(min_glucose) - len(y_test):].index, y_test, color='red', label='Real Min Glucose')
    axes.plot(min_glucose[len(min_glucose) - len(y_test):].index, y_test_pred, color='blue', label='Predicted Min Glucose')
    # axes.xticks(np.arange(0,394,50))
    plt.title('Min Glucose Prediction')
    plt.xlabel('Time')
    plt.ylabel('Glucose')
    plt.legend()
    plt.savefig(f'graphs/glucose_pred_{num_epochs}e.png')


if __name__ == "__main__":
    main()
