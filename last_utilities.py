
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import math
from keras.models import Sequential
from keras.layers import Dense, LSTM
import pandas as pd

def entrenar_modelo(df, predictors):
    # Instantiate a RandomForestClassifier model with specific parameters
    modelo = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1 )

    # Split the DataFrame into training and testing sets
    # The first 50 rows are used for training, the rest for testing
    train = df.iloc[:-50]
    train = train.dropna()
    # Train the model using the fit method
    # The training data and the target values are passed to this method
    modelo.fit(train[predictors], train['TARGET9'])

    # Return the trained model
    return modelo

def testear_modelo(modelo, df, predictors):
    # Define the predictor column names
    # These are the features that the model will use to make predictions
    #predictors = ['CIERRE', 'APERTURA', 'MAXIMO', 'MINIMO', 'RESULTADO', 'TOMORROW']
    test = df.iloc[-50:]
    test = test.dropna()
    test.reset_index(drop=True, inplace=True)
    # Use the predict method to make predictions
    # The test data is passed to this method
    predictions = modelo.predict_proba(test[predictors])

    probs_df = pd.DataFrame(predictions, columns=modelo.classes_)
    # Calculate the precision of the model
    #res = precision_score(test['TARGET'], predictions[:,1] > 0.5)
    #print('Precision: ', res)

    #resultado = pd.DataFrame({'FECHA': test['FECHA'], 'TARGET': test['TARGET9'], 'PREDICTION': probs_df[0].apply(lambda x: 1 if x < 0.5 else 0)})
    resultado = pd.DataFrame({'FECHA': test['FECHA'], 'TARGET': test['TARGET9'], 'PREDICTION': probs_df[0]})
    # Return the predictions
    return resultado

def predict(train, test, predictors, model):
    model.fit(train[predictors], train['TARGET9'])
    preds = model.predict_proba(test[predictors])
    preds = pd.Series(preds, index=test.index, name='Predictions')
    combined = pd.concat([test['TARGET9'], preds], axis=1)
    return combined

def backtest(df, predictors, model, start=1500, step=150):
    all_predictions =  []
    for i in range(start, len(df), step):
        train = df.iloc[0:i].copy()
        test = df.iloc[i:(i+step)].copy()
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)
    return pd.concat(all_predictions)



def create_dataset(df):
    # Create a new dataframe with only the 'CIERRE' column
    data = df.filter(['CIERRE'])
    dataset = data.values
    training_data_len = math.ceil(len(dataset) * .8)

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)

    # Create the training data set
    train_data = scaled_data[0:training_data_len, :]

    # Split the data into x_train and y_train data sets
    x_train = []
    y_train = []
    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])

    # Convert the x_train and y_train to numpy arrays
    x_train, y_train = np.array(x_train), np.array(y_train)

    # Reshape the data
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(x_train, y_train, batch_size=1, epochs=1)

    # Create the testing data set
    test_data = scaled_data[training_data_len - 60:, :]

    return
#%%
