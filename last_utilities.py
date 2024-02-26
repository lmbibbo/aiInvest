from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

def entrenar_modelo(df, y):
    # Instantiate a RandomForestClassifier model with specific parameters
    modelo = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)

    # Split the DataFrame into training and testing sets
    # The first 50 rows are used for training, the rest for testing
    train = df.iloc[:50]

    # Define the predictor column names
    # These are the features that the model will use to make predictions
    predictors = ['CIERRE_N', 'APERTURA_N', 'MAXIMO_N', 'MINIMO_N', 'RESULTADO_N', 'PORCENTAJE_N']

    # Train the model using the fit method
    # The training data and the target values are passed to this method
    modelo.fit(train[predictors], train['TARGET'])

    # Return the trained model
    return modelo

def testear_modelo(modelo, df):
    # Define the predictor column names
    # These are the features that the model will use to make predictions
    predictors = ['CIERRE_N', 'APERTURA_N', 'MAXIMO_N', 'MINIMO_N', 'RESULTADO_N', 'PORCENTAJE_N']
    test = df.iloc[50:]
    # Use the predict method to make predictions
    # The test data is passed to this method
    predictions = modelo.predict(test[predictors])

    # Calculate the precision of the model
    precision_score(test['TARGET'], predictions)
    # Return the predictions
    return predictions