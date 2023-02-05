# glucose-analysis
2023 HackBU Hackathon - Analyze Glucose readings from a diabetic

This code uses pandas to process glucose readings over ~1 year. The data is then fed into an LSTM using a couple stats like min, max, average, and range over a timeframe such as 1 hour or 1 day. The result is an image showing the predictions compared to the actual result. Typically I was trying to predict the minimum glucose in the next timeframe, because low glucose events can be very dangerous. This proved to be minorly effective. The LSTM was good at fitting to the overall trend of the data accoding to the results on the test dataset, but it is difficult to predict lows before they happen. The accuracy overall of predicting whether the next reading will be above or below the low threshold was over 90%, but that is because they are sparse. The accuracy of predicting a low glucose event given that one is about to occur is about 40-45%.

I relied on the multivariate LSTM from https://stackoverflow.com/questions/56858924/multivariate-input-lstm-in-pytorch to get started
