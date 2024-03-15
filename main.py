#!/usr/bin/env python
# coding: utf-8

# In[1]:


from utilities import *
import matplotlib.pyplot as plt


# In[2]:


dias = 2455
access_token = '63ab8fe9b94c7c7a4ebc6c03d79550f93818202a'


# In[3]:


df_dmep_original = load_especie("DOLAR MEP", access_token)
df_djones_original = load_especie("DIA_US", access_token)
df_merval_original = load_especie('MERVAL', access_token)
df_doficial_original = load_especie('DOLAR OFICIAL', access_token)
df_ypfd_original = load_especie('YPFD', access_token)

df_ypfd = recortar(df_ypfd_original, dias)
df_dmep = recortar(df_dmep_original, dias)
df_doficial = recortar(df_doficial_original, dias + (int(dias/7)*2))
df_merval = recortar(df_merval_original, dias)
df_djones = recortar(df_djones_original, dias)


# In[4]:


plt.figure(figsize=(14, 8))
plt.plot(df_dmep['FECHA'], df_dmep['CIERRE_N'], color='blue')
plt.plot(df_merval['FECHA'], df_merval['CIERRE_N'], color='red')
plt.plot(df_doficial['FECHA'], df_doficial['CIERRE_N'], color='green')
plt.plot(df_djones['FECHA'], df_djones['CIERRE_N'], color='yellow')

print('Dolar Oficial: Verde')
print('Dolar MEP: Azul')
print('Merval: Rojo')
print('Down Jones: Amarillo')


# In[5]:


predictors = ['RESULTADO', 'CIERRE1', 'CIERRE3', 'CIERRE5', 'CIERRE1_dmep', 'CIERRE3_dmep', 'CIERRE5_dmep', 'RESULTADO_dmep', 'CIERRE1_merval', 'CIERRE3_merval', 'CIERRE5_merval', 'RESULTADO_merval', 'CIERRE1_djones', 'CIERRE3_djones', 'CIERRE5_djones', 'RESULTADO_djones']
#predictors = ['CIERRE', 'APERTURA', 'RESULTADO']

df_mergeado = pd.merge(df_ypfd, df_dmep, on='FECHA', how='inner', suffixes=('', '_dmep'))
df_mergeado = pd.merge(df_mergeado, df_merval, on='FECHA', how='inner', suffixes=('', '_merval'))
df_mergeado = pd.merge(df_mergeado, df_djones, on='FECHA', how='inner', suffixes=('', '_djones'))
df_mergeado.dropna()


# In[8]:


get_ipython().run_line_magic('pip', 'install tensorflow')


# In[9]:


get_ipython().run_line_magic('pip', 'install keras')


# In[10]:


from sklearn.preprocessing import MinMaxScaler
import numpy as np
import math
from keras.models import Sequential
from keras.layers import Dense, LSTM
import pandas as pd

# Create a new dataframe with only the 'CIERRE' column
data = df_ypfd.filter(['CIERRE'])
dataset = data.values
training_data_len = math.ceil(len(dataset) * .8)


# In[9]:


dataset.shape


# In[10]:


# Scale the data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

# Create the training data set
train_data = scaled_data[0:training_data_len, :]

train_data


# Acá está la clave..... cómo armo x_train e y_train para que la máquina aprenda a obtener el y.
# 
# Una cosa que se me ocurre es armar el x con varios días previos y ponerle al y un % de ganancia/perdida

# In[12]:


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



# In[13]:


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


# Llegamos hasta acá

# In[14]:


#Create de Testing data set
#Create a new array containing scaled values
test_data = scaled_data[training_data_len - 60: , :]
#Create the data sets x_test and y_test
x_test = []
y_test = dataset[training_data_len:, :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])
    


# In[15]:


# Convert the data to a numpy array
x_test = np.array(x_test)

x_test.shape


# In[16]:


# Reshape the data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Get the models predicted price values
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Get the root mean squared error (RMSE)
rmse = np.sqrt(np.mean(predictions - y_test) ** 2)
rmse


# In[18]:


#plot the data
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions
#Visualize the data
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['CIERRE'])
plt.plot(valid[['CIERRE', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()


# In[19]:


#Show the valid and predicted prices
valid


# In[21]:


#Get the quote
#ypfd_quote = load_especie('YPFD', access_token)
ypfd_quote = df_ypfd
#Create a new dataframe
new_df = ypfd_quote.filter(['CIERRE'])
#Get the last 60 day closing price values and convert the dataframe to an array
last_60_days = new_df[-60:].values
#Scale the data to be values between 0 and 1
last_60_days_scaled = scaler.transform(last_60_days)
#Create an empty list
X_test = []
#Append the past 60 days
X_test.append(last_60_days_scaled)
#Convert the X_test data set to a numpy array
X_test = np.array(X_test)
#Reshape the data
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
#Get the predicted scaled price
pred_price = model.predict(X_test)
#undo the scaling
pred_price = scaler.inverse_transform(pred_price)
print(pred_price)



# In[22]:


pred_price


# In[19]:


predictions


# In[20]:


y_test


# In[17]:


# Unir los arreglos
unidos = np.column_stack((predictions, y_test))

# Calcular la diferencia
diferencia = y_test - predictions

# Añadir la diferencia como una nueva columna
resultado = np.column_stack((unidos, diferencia))

resultado


# In[ ]:


#Main File
from importlib import reload
from utilities import *
from last_utilities import *
import matplotlib.pyplot as plt

acces_token = '2501a8d1e3bb3592457f0e03181bc6087ee16794'
df_dmep_original = load_especie("DOLAR MEP", acces_token)
df_djones_original = load_especie("DIA_US", acces_token)
df_merval_original = load_especie('MERVAL', acces_token)
df_doficial_original = load_especie('DOLAR OFICIAL', acces_token)

file = 'data/series_historicas_acciones.csv'
df = read_data(file)
df = convertir(df)
df = df.sort_values(by=['FECHA'])

dias = 350


# In[ ]:


get_ipython().system('pip install -U scikit-learn')
get_ipython().system('pip install -U keras')


# In[ ]:


import matplotlib.pyplot as plt
import pandas as pd
import sklearn as sklearn



# In[ ]:


df_dmep = recortar(df_dmep_original, dias)
df_doficial = recortar(df_doficial_original, dias + (int(dias/7)*2))
df_merval = recortar(df_merval_original, dias)
df_djones = recortar(df_djones_original, dias)

plt.figure(figsize=(14, 8))
plt.plot(df_dmep['FECHA'], df_dmep['CIERRE_N'], color='blue')
plt.plot(df_merval['FECHA'], df_merval['CIERRE_N'], color='red')
plt.plot(df_doficial['FECHA'], df_doficial['CIERRE_N'], color='green')
plt.plot(df_djones['FECHA'], df_djones['CIERRE_N'], color='yellow')

print('Dolar Oficial: Verde')
print('Dolar MEP: Azul')
print('Merval: Rojo')
print('Down Jones: Amarillo')


# In[ ]:


df_ypfd_original = load_especie('YPFD', acces_token)
df_ypfd = recortar(df_ypfd_original, dias)


# In[ ]:


reload(last_utilities)


# In[ ]:


from last_utilities import *

# Train the model using the fit method
# The training data and the target values are passed to this method
predictors = ['RESULTADO', 'CIERRE1', 'CIERRE3', 'CIERRE5', 'CIERRE1_dmep', 'CIERRE3_dmep', 'CIERRE5_dmep', 'RESULTADO_dmep', 'CIERRE1_merval', 'CIERRE3_merval', 'CIERRE5_merval', 'RESULTADO_merval', 'CIERRE1_djones', 'CIERRE3_djones', 'CIERRE5_djones', 'RESULTADO_djones']
#predictors = ['CIERRE', 'APERTURA', 'RESULTADO']

df_mergeado = pd.merge(df_ypfd, df_dmep, on='FECHA', how='inner', suffixes=('', '_dmep'))
df_mergeado = pd.merge(df_mergeado, df_merval, on='FECHA', how='inner', suffixes=('', '_merval'))
df_mergeado = pd.merge(df_mergeado, df_djones, on='FECHA', how='inner', suffixes=('', '_djones'))
df_mergeado.dropna()

model = entrenar_modelo(df_mergeado, predictors)


# In[ ]:


result = testear_modelo(model, df_mergeado, predictors)


# In[ ]:


result


# 

# In[ ]:


get_ipython().system('pip install tensorflow')


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import math
from keras.models import Sequential
from keras.layers import Dense, LSTM
import pandas as pd

data = df_ypfd.filter(['CIERRE'])
dataset = data.values
training_data_len = math.ceil(len(dataset) * .8)


# In[ ]:


# Scale the data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)


# In[ ]:


training_data_len


# In[ ]:


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



# In[ ]:


df= df_mergeado
start = 2500
step = 100
all_predictions =  []
for i in range(start, df.shape[0], step):
    train = df.iloc[0:i].copy()
    test = df.iloc[i:(i+step)].copy()
    predictions = predict(train, test, predictors, model)
    print(predictions)
    all_predictions.append(predictions)


# In[ ]:


df.shape[0]


# In[ ]:


preditions = backtest(df_mergeado, model, predictors)


# In[ ]:


horizons = [1, 3, 5, 9]
#df_ypfd = df_ypfd.apply(pd.to_numeric, errors='coerce')
for horizon in horizons:
    rolling_ypfd = df_ypfd.rolling(window=horizon)
    
    ratio_column = f'CIERRE{horizon}'
    df_ypfd[ratio_column] = df_ypfd['CIERRE'].shift(-horizon)
    


# In[ ]:


df_ypfd


# In[ ]:


grupos_por_clase = df.groupby('CLASE')
for clave_clase, grupo_clase in grupos_por_clase:
    grupos_por_simbolo = grupo_clase.groupby('SIMBOLO')
    for clave_grupo, grupo in grupos_por_simbolo:
        df1 = grupos_por_simbolo.get_group(clave_grupo)
        df1 = df1.tail(dias)
        df1 = newBullishBearish(df1)
        df1 = normalizar(df1)
        plt.figure(figsize=(14, 8))
        plt.plot(df1['FECHA'], df1['CIERRE_N'], color='orange')
        plt.plot(df_dmep['FECHA'], df_dmep['CIERRE_N'], color='blue')
        plt.plot(df_merval['FECHA'], df_merval['CIERRE_N'], color='red')
        plt.plot(df_doficial['FECHA'], df_doficial['CIERRE_N'], color='green')
        #plt.plot(df_djones['FECHA'], df_djones['CIERRE_N'], color='yellow')
        plt.xticks(rotation=45, fontsize=16)
        plt.grid()
        plt.xlabel('Fecha')
        plt.ylabel('Valor de Cierre')
        plt.title('Cierres de'+ ' ' + clave_clase + '-' + clave_grupo)
        plt.show()


# In[ ]:


df_dmep.index = df_dmep['FECHA']
df_dmep.index


# In[ ]:


grupos_por_simbolo = df.groupby('SIMBOLO')
df1 = pd.DataFrame()
for clave_grupo, grupo in grupos_por_simbolo:
    df1 = grupos_por_simbolo.get_group(clave_grupo)
    df1 = df1.tail(dias)
    df1 = newBullishBearish(df1)
    df1 = normalizar(df1)

    plt.figure(figsize=(14, 8))
    plt.plot(df1['FECHA'], df1['RESULTADO_N'], color='orange')
    plt.plot(df_dmep['FECHA'], df_dmep['RESULTADO_N'], color='blue')
    plt.plot(df_merval['FECHA'], df_merval['RESULTADO_N'], color='red')
    
    plt.xticks(rotation=45, fontsize=16)
    plt.grid()
    plt.xlabel('Fecha')
    plt.ylabel('Valor de Cierre')
    plt.title('Cierres de'+ ' ' + clave_grupo)
    plt.show()


# In[ ]:


panel_gral_file = 'data/panelgeneral.csv'
df_panel_gral = read_data(panel_gral_file)
df_panel_gral = df_panel_gral.sort_values(by=['FECHA'])
df_panel_gral = convertir(df_panel_gral)

grupos_por_simbolo = df_panel_gral.groupby('SIMBOLO')

for clave_grupo, grupo in grupos_por_simbolo:
    df1 = grupos_por_simbolo.get_group(clave_grupo)
    df1 = df1.tail(dias)
    df1 = newBullishBearish(df1)
    df1 = normalizar(df1)

    plt.figure(figsize=(14, 8))
    plt.plot(df1['FECHA'], df1['CIERRE_N'], color='orange')
    plt.plot(df_dmep['FECHA'], df_dmep['CIERRE_N'], color='blue')
    plt.plot(df_merval['FECHA'], df_merval['CIERRE_N'], color='red')

    plt.xticks(rotation=45, fontsize=16)
    plt.grid()
    plt.xlabel('Fecha')
    plt.ylabel('Valor de Cierre')
    plt.title('Cierres de' + ' ' + clave_grupo)
    plt.show()


# In[ ]:


import matplotlib.pyplot as plt
import numpy as np

# Random test data
np.random.seed(19680801)
all_data = [np.random.normal(0, std, size=100) for std in range(1, 4)]
labels = ['x1', 'x2', 'x3']

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))

# rectangular box plot
bplot1 = ax1.boxplot(all_data,
                     vert=True,  # vertical box alignment
                     patch_artist=True,  # fill with color
                     labels=labels)  # will be used to label x-ticks
ax1.set_title('Rectangular box plot')

# notch shape box plot
bplot2 = ax2.boxplot(all_data,
                     notch=True,  # notch shape
                     vert=True,  # vertical box alignment
                     patch_artist=True,  # fill with color
                     labels=labels)  # will be used to label x-ticks
ax2.set_title('Notched box plot')

# fill with colors
colors = ['pink', 'lightblue', 'lightgreen']
for bplot in (bplot1, bplot2):
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)

# adding horizontal grid lines
for ax in [ax1, ax2]:
    ax.yaxis.grid(True)
    ax.set_xlabel('Three separate samples')
    ax.set_ylabel('Observed values')

plt.show()


# In[ ]:


df1


# In[ ]:


#Cómo marcar alertas en un gráfico?
# definir que quiero hacer.... comprar o vender?: Esto se puede definir a partir de % que estoy dispuesto a perder o ganar
# Costo de la operación: comisiones, impuestos, etc
# Costo del mantenimiento mensual
# inflación acumulada
#Alerta de Venta por papel
#Entrenar un modelo para predecir el precio de cierre

comprar = 2
vender = 20
tamaño_lote = 4
grupos_por_simbolo = df.groupby('SIMBOLO')
df1 = pd.DataFrame()
for clave_grupo, grupo in grupos_por_simbolo:
    lotes = [grupo.iloc[i:i+tamaño_lote] for i in range(0, len(grupo), tamaño_lote)]
    
    for i, lote_actual in enumerate(lotes):
        lote_procesado = procesar_lote(lote_actual, comprar, vender)
        resultado_medio = lote_procesado['RESULTADO_N'].mean()
        resultado_avg = lote_procesado['RES_PESADO_N'].mean()
        print('Clave:' + clave_grupo+ ' Indice:', i, ' Resultado pesado:', resultado_medio)
    


# In[ ]:


lote_procesado


# In[ ]:




