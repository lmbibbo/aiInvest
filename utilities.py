import pandas as pd
import numpy as np

def read_data(file):
    filas = 365
    df = pd.read_csv(file, skiprows=lambda x: x != 0 and x < (filas*-1))
    return df

def read_data_xls(file):
    df = pd.read_excel(file)
    return df

def clasificar(simbolo):
    if simbolo in ['BBAR', 'BMA', 'VALO', 'SUPV', 'GGAL']:
        return 'BANCOS'
    elif simbolo in ['ALUA', 'LOMA', 'TXAR' ]:
        return 'CONSTRUCCION'
    elif simbolo in ['CEPU', 'EDN', 'PAMP', 'TGNO4', 'TGSU2', 'TRAN', 'YPFD']:
        return 'ENERGIA'
    elif simbolo in ['COME', 'MIRG', 'BYMA']:
        return 'COMERCIO'
    elif simbolo in ['AGRO', 'CRES' ]:
        return 'AGRO'
    elif simbolo in ['TECO2' ]:
        return 'TELECOMUNICACIONES'
    else:
        return 'Desconocido'
def convertir_rava(df):
    df.columns = df.columns.str.upper()
    df['FECHA'] = pd.to_datetime(df['FECHA'], format='%Y-%m-%d')
    df['CIERRE'] = df['CIERRE'].astype(float)
    df['APERTURA'] = df['APERTURA'].astype(float)
    df['RESULTADO'] = df['CIERRE'] - df['APERTURA']
    df['PORCENTAJE'] = ((df['CIERRE'] - df['APERTURA']) / df['APERTURA'])
    return df
def convertir(df):
    df['FECHA'] = pd.to_datetime(df['FECHA'], format='%Y-%m-%d')
    df['CIERRE'] = df['CIERRE'].str.replace(',', '.').astype(float)
    df['APERTURA'] = df['APERTURA'].str.replace(',', '.').astype(float)
    df['MAXIMO'] = df['MAXIMO'].str.replace(',', '.').astype(float)
    df['MINIMO'] = df['MINIMO'].str.replace(',', '.').astype(float)
    #df['PRECIO PROMEDIO'] = df['PRECIO PROMEDIO'].str.replace(',', '.').astype(float)
    #df['CANTIDAD DE OPERACIONES'] = df['CANTIDAD DE OPERACIONES'].astype(int)
    #df['VOLUMEN NOMINAL'] = df['VOLUMEN NOMINAL'].str.replace(',', '.').astype(float)
    #df['MONTO NEGOCIADO'] = df['MONTO NEGOCIADO'].str.replace(',', '.').astype(float)
    df['RESULTADO'] = df['CIERRE'] - df['APERTURA']
    df['PORCENTAJE'] = ((df['CIERRE'] - df['APERTURA']) / df['APERTURA'])
    #df['RES_PESADO'] = df['RESULTADO'] * df['MONTO NEGOCIADO']
    df['CLASE'] = df['SIMBOLO'].apply(clasificar)
    return df


def norma_key(df, clave, rango=0):
    v_min = np.min(df[clave])
    v_max = np.max(df[clave])

    #df[clave + '_N'] = (((df[clave] - v_min) * (rango+1)) / (v_max - v_min))-rango # Normalización por rango
    df[clave + '_N'] = ((df[clave] - df[clave].mean()) / df[clave].std()).round(2)  # Normalización por z-score o Gaussian normalization
    return df


def normalizar(df):
    df_ret = pd.DataFrame(df)
    norma_key(df_ret, 'CIERRE', 1)
    norma_key(df_ret, 'APERTURA', 1)
    norma_key(df_ret, 'MAXIMO', 1)
    norma_key(df_ret, 'MINIMO', 1)
    #norma_key(df_ret, 'PRECIO PROMEDIO', 1)
    #norma_key(df_ret, 'VOLUMEN NOMINAL', 1)
    #norma_key(df_ret, 'MONTO NEGOCIADO', 1)
    norma_key(df_ret, 'RESULTADO', 1)
    #norma_key(df_ret, 'RES_PESADO', 1)
    #norma_key(df_ret, 'BULLISH', 0)
    #norma_key(df_ret, 'BEARISH', 0)
    return df_ret


def normalizar2(df):
    df_ret = pd.DataFrame(df)
    norma_key(df_ret, 'CIERRE', 1)
    norma_key(df_ret, 'APERTURA', 1)
    norma_key(df_ret, 'MAXIMO', 1)
    norma_key(df_ret, 'MINIMO', 1)
    #norma_key(df_ret, 'PRECIO PROMEDIO', 1)
    #norma_key(df_ret, 'VOLUMEN NOMINAL', 1)
    #norma_key(df_ret, 'MONTO NEGOCIADO', 1)
    return df_ret


def newBullishBearish(df):
    df_ret = pd.DataFrame(df)
    df_ret['BULLISH'] = 0
    df_ret['BEARISH'] = 0
    df_ret.loc[df_ret['RESULTADO'] > 0, 'BULLISH'] = 1
    df_ret.loc[df_ret['RESULTADO'] < 0, 'BEARISH'] = 1
    bull = 0
    bear = 0
    for indice, fila in df_ret.iterrows():
        if (fila['BULLISH'] > 0):
            df_ret.at[indice, 'BULLISH'] = fila['BULLISH'] + bull
            bull = bull + 1
        else:
            bull = 0
        if (fila['BEARISH'] > 0):
            df_ret.at[indice, 'BEARISH'] = fila['BEARISH'] + bear
            bear = bear + 1
        else:
            bear = 0
    return df_ret


def bullish_bearish(df):
    df_ret = pd.DataFrame(df)
    df_ret['BULLISH'] = 0
    df_ret['BEARISH'] = 0
    df_ret.loc[df_ret['RESULTADO'] > 0, 'BULLISH'] = 1
    df_ret.loc[df_ret['RESULTADO'] < 0, 'BEARISH'] = 1
    anteriores = ['1', '2', '3', '4', '5']
    for dias in anteriores:
        df_ret['BULLISH_' + dias] = 0
        df_ret['BEARISH_' + dias] = 0
        df_ret['BULLISH_' + dias] = df_ret['BULLISH'].shift(int(dias))
        df_ret['BEARISH_' + dias] = df_ret['BEARISH'].shift(int(dias))
        '''
        if dias == '1':
            df_ret['BULLISH_' + dias] = np.where(df_ret['BULLISH'].shift(int(dias)) > 0, df_ret['BULLISH'].shift(int(dias)), 0)
            df_ret['BEARISH_' + dias] = np.where(df_ret['BEARISH'].shift(int(dias)) > 0, df_ret['BEARISH'].shift(int(dias)), 0)
        else:
            df_ret['BULLISH_' + dias] = np.where(df_ret['BULLISH_' + dias].shift(int(dias)) > 0, df_ret['BULLISH'].shift(int(dias)), 0)
            df_ret['BEARISH_' + dias] = np.where(df_ret['BEARISH_' + dias].shift(int(dias)) > 0, df_ret['BEARISH'].shift(int(dias)), 0)
    '''
    df_ret['BULLISH_4'] = np.where(df_ret['BULLISH_4'] > 0, df_ret['BULLISH_4'] + df_ret['BULLISH_5'], 0)
    df_ret['BEARISH_4'] = np.where(df_ret['BEARISH_4'] > 0, df_ret['BEARISH_4'] + df_ret['BEARISH_5'], 0)
    df_ret['BULLISH_3'] = np.where(df_ret['BULLISH_3'] > 0, df_ret['BULLISH_3'] + df_ret['BULLISH_4'], 0)
    df_ret['BEARISH_3'] = np.where(df_ret['BEARISH_3'] > 0, df_ret['BEARISH_3'] + df_ret['BEARISH_4'], 0)
    df_ret['BULLISH_2'] = np.where(df_ret['BULLISH_2'] > 0, df_ret['BULLISH_2'] + df_ret['BULLISH_3'], 0)
    df_ret['BEARISH_2'] = np.where(df_ret['BEARISH_2'] > 0, df_ret['BEARISH_2'] + df_ret['BEARISH_3'], 0)
    df_ret['BULLISH_1'] = np.where(df_ret['BULLISH_1'] > 0, df_ret['BULLISH_1'] + df_ret['BULLISH_2'], 0)
    df_ret['BEARISH_1'] = np.where(df_ret['BEARISH_1'] > 0, df_ret['BEARISH_1'] + df_ret['BEARISH_2'], 0)
    df_ret['BULLISH'] = np.where(df_ret['BULLISH'] > 0, df_ret['BULLISH'] + df_ret['BULLISH_1'], 0)
    df_ret['BEARISH'] = np.where(df_ret['BEARISH'] > 0, df_ret['BEARISH'] + df_ret['BEARISH_1'], 0)

    return df_ret


def procesar_lote(actual, comprar, vender):
    nuevo = bullish_bearish(actual)
    nuevo = normalizar(nuevo)
    return nuevo


'''
    df_ret['BULL_ANT'] = np.where(df_ret['BULLISH'].shift(1) > 0, df_ret['BULLISH'].shift(1), 0)
    df_ret['BEAR_ANT'] = np.where(df_ret['BEARISH'].shift(1) > 0, df_ret['BEARISH'].shift(1), 0)
    df_ret['BULL_ANT2'] = np.where(df_ret['BULLISH'].shift(2) > 0, df_ret['BULLISH'].shift(2), 0)
    df_ret['BEAR_ANT2'] = np.where(df_ret['BEARISH'].shift(2) > 0, df_ret['BEARISH'].shift(2), 0)
    df_ret['BULL_ANT'] = np.where(df_ret['BULL_ANT'] > 0, df_ret['BULL_ANT'] + df_ret['BULL_ANT2'], 0)
    df_ret['BEAR_ANT'] = np.where(df_ret['BEAR_ANT'] > 0, df_ret['BEAR_ANT'] + df_ret['BEAR_ANT2'], 0)
    df_ret['BULLISH'] = np.where(df_ret['BULLISH'] > 0, df_ret['BULLISH'] + df_ret['BULL_ANT'], 0)
    df_ret['BEARISH'] = np.where(df_ret['BEARISH'] > 0, df_ret['BEARISH'] + df_ret['BEAR_ANT'], 0)

'''
# %%
