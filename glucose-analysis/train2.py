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
    low_threshold = 80
    high_threshold = 200

    glucose_readings = df.query('`Record Type` in [0, 1]')
    glucose_readings['Glucose'] = glucose_readings['Historic Glucose mg/dL'].where(glucose_readings['Record Type'] == 0,
                                                                                   other=glucose_readings[
                                                                                       'Scan Glucose mg/dL'], inplace=False)
    glucose_readings = glucose_readings[['Device Timestamp', 'Glucose']]
    glucose_readings.reset_index(drop=True, inplace=True)


    time_chunk = '6H'

    glucose_readings['Device Timestamp'] = pd.to_datetime(glucose_readings['Device Timestamp'], )
    glucose_readings = glucose_readings.set_index('Device Timestamp', inplace=False)
    hourly_glucose = glucose_readings.resample(time_chunk).mean()
    hourly_glucose['Min Glucose'] = glucose_readings.resample(time_chunk).min()['Glucose']
    hourly_glucose['Max Glucose'] = glucose_readings.resample(time_chunk).max()['Glucose']
    hourly_glucose['Low Event'] = (hourly_glucose['Min Glucose'] < low_threshold).astype(float)
    # hourly_glucose['High Event'] = (hourly_glucose['Max Glucose'] > high_threshold).astype(float)
    # print(hourly_glucose.head())
    # min_val = hourly_glucose['Min Glucose'].min()
    # max_val = hourly_glucose['Max Glucose'].max()

    hourly_glucose['Glucose'] = (hourly_glucose['Glucose']-low_threshold)/(high_threshold-low_threshold)
    hourly_glucose['Min Glucose'] = (hourly_glucose['Min Glucose']-low_threshold)/(high_threshold-low_threshold)
    hourly_glucose['Max Glucose'] = (hourly_glucose['Max Glucose']-low_threshold)/(high_threshold-low_threshold)

    print(hourly_glucose.head())


    # LSTM Training


    min_glucose = hourly_glucose[['Glucose', 'Min Glucose', 'Max Glucose']]
    min_glucose = min_glucose.fillna(method='ffill')
    min_glucose['Range'] = min_glucose['Max Glucose'] - min_glucose['Min Glucose']
    # min_glucose['y'] = hourly_glucose['Low Event']
    min_glucose['y'] = min_glucose['Min Glucose']
    min_glucose.to_csv('min_glucose.csv')

    # min_glucose.info()



    def load_data(stock, look_back):
        data_raw = stock.values  # convert to numpy array
        data = []

        # create all possible sequences of length look_back
        for index in range(len(data_raw) - look_back):
            data.append(data_raw[index: index + look_back])

        data = np.array(data)
        test_set_size = int(np.round(0.2 * data.shape[0]))
        train_set_size = data.shape[0] - test_set_size

        x_train = data[:train_set_size, :-1, :-1]
        y_train = data[:train_set_size, -1, -1]

        x_test = data[train_set_size:, :-1, :-1]
        y_test = data[train_set_size:, -1, -1]

        return [x_train, y_train, x_test, y_test]


    look_back = 30  # choose sequence length
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


    class MV_LSTM(torch.nn.Module):
        def __init__(self, n_features, seq_length):
            super(MV_LSTM, self).__init__()
            self.n_features = n_features
            self.seq_len = seq_length
            self.n_hidden = 5  # number of hidden states
            self.n_layers = 1  # number of LSTM layers (stacked)

            self.l_lstm = torch.nn.LSTM(input_size=n_features,
                                        hidden_size=self.n_hidden,
                                        num_layers=self.n_layers,
                                        batch_first=True)
            # according to pytorch docs LSTM output is
            # (batch_size,seq_len, num_directions * hidden_size)
            # when considering batch_first = True
            self.l_linear = torch.nn.Linear(self.n_hidden * self.seq_len, 1)

        def init_hidden(self, batch_size):
            # even with batch_first = True this remains same as docs
            hidden_state = torch.zeros(self.n_layers, batch_size, self.n_hidden)
            cell_state = torch.zeros(self.n_layers, batch_size, self.n_hidden)
            self.hidden = (hidden_state, cell_state)

        def forward(self, x):
            batch_size, seq_len, _ = x.size()

            lstm_out, self.hidden = self.l_lstm(x, self.hidden)
            # lstm_out(with batch_first = True) is
            # (batch_size,seq_len,num_directions * hidden_size)
            # for following linear layer we want to keep batch_size dimension and merge rest
            # .contiguous() -> solves tensor compatibility error
            x = lstm_out.contiguous().view(batch_size, -1)
            return self.l_linear(x)

    n_features = 4  # this is number of parallel inputs
    n_timesteps = look_back-1  # this is number of timesteps

    # convert dataset into input/output
    # X, y = split_sequences(dataset, n_timesteps)
    X, y = x_train, y_train
    print(X.shape, y.shape)

    # create NN
    mv_net = MV_LSTM(n_features, n_timesteps)
    criterion = torch.nn.MSELoss()  # reduction='sum' created huge loss value
    optimizer = torch.optim.Adam(mv_net.parameters(), lr=.005)

    train_episodes = 20
    batch_size = 16

    mv_net.train()
    for t in range(train_episodes):
        for b in range(0, len(X), batch_size):
            inpt = X[b:b + batch_size, :, :]
            target = y[b:b + batch_size]

            x_batch = torch.tensor(inpt, dtype=torch.float32)
            y_batch = torch.tensor(target, dtype=torch.float32)

            mv_net.init_hidden(x_batch.size(0))
            #    lstm_out, _ = mv_net.l_lstm(x_batch,nnet.hidden)
            #    lstm_out.contiguous().view(x_batch.size(0),-1)
            output = mv_net(x_batch)
            # print(type(output))
            loss = criterion(output.view(-1), y_batch)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print('step : ', t, 'loss : ', loss.item())

    mv_net.eval()
    y_test_pred = []
    batch_size = 1
    with torch.no_grad():
        for b in range(0, len(x_test), batch_size):
            inpt = x_test[b:b + batch_size, :, :]
            # target = y_test[b:b + batch_size]

            x_batch = torch.tensor(inpt, dtype=torch.float32)
            # y_batch = torch.tensor(target, dtype=torch.float32)

            mv_net.init_hidden(x_batch.size(0))
            #    lstm_out, _ = mv_net.l_lstm(x_batch,nnet.hidden)
            #    lstm_out.contiguous().view(x_batch.size(0),-1)
            y_test_pred.append((mv_net(x_batch)).detach().numpy())
        # y_test_pred = mv_net(x_test)
        # print(y_test_pred)
    # y_test_pred = y_test_pred.detach().numpy() * (high_threshold - low_threshold) + low_threshold
    y_test_pred = np.array([x * (high_threshold - low_threshold) + low_threshold for x in y_test_pred]).flatten()
    y_test = y_test.detach().numpy() * (high_threshold - low_threshold) + low_threshold

    print(y_test_pred.shape)
    print(y_test.shape)
    testScore = math.sqrt(mean_squared_error(y_test[:], y_test_pred[:]))
    print('Test Score: %.2f RMSE' % testScore)
    figure, axes = plt.subplots(figsize=(15, 6))
    axes.xaxis_date()

    axes.plot(min_glucose[len(min_glucose) - len(y_test):].index, y_test, color='red', label='Real Min Glucose')
    axes.plot(min_glucose[len(min_glucose) - len(y_test):].index, y_test_pred, color='blue',
              label='Predicted Min Glucose')
    # axes.xticks(np.arange(0,394,50))
    plt.title('Min Glucose Prediction')
    plt.xlabel('Time')
    plt.ylabel('Glucose')
    plt.legend()
    # plt.show()
    plt.savefig(f'graphs/glucose_pred2_{train_episodes}e.png')

    with torch.no_grad():

        inpt = x_test[-1:, :, :]
        target = y_test[-1:]

        x_batch = torch.tensor(inpt, dtype=torch.float32)
        # y_batch = torch.tensor(target, dtype=torch.float32)

        mv_net.init_hidden(x_batch.size(0))
        #    lstm_out, _ = mv_net.l_lstm(x_batch,nnet.hidden)
        #    lstm_out.contiguous().view(x_batch.size(0),-1)
        output = [x * (high_threshold - low_threshold) + low_threshold for x in ((mv_net(x_batch)).detach().numpy().flatten())]
        print(output, target)


    exit()


if __name__ == "__main__":
    main()
