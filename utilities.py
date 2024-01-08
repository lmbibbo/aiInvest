import pandas as pd
import numpy as np

def read_data(file):
    df = pd.read_csv(file)
    return df


def convertir(df):
    df['FECHA'] = pd.to_datetime(df['FECHA'], format='%Y-%m-%d')
    df['CIERRE'] = df['CIERRE'].str.replace(',', '.').astype(float)
    df['APERTURA'] = df['APERTURA'].str.replace(',', '.').astype(float)
    df['MAXIMO'] = df['MAXIMO'].str.replace(',', '.').astype(float)
    df['MINIMO'] = df['MINIMO'].str.replace(',', '.').astype(float)
    df['PRECIO PROMEDIO'] = df['PRECIO PROMEDIO'].str.replace(',', '.').astype(float)
    df['CANTIDAD DE OPERACIONES'] = df['CANTIDAD DE OPERACIONES'].astype(int)
    df['VOLUMEN NOMINAL'] = df['VOLUMEN NOMINAL'].str.replace(',', '.').astype(float)
    df['MONTO NEGOCIADO'] = df['MONTO NEGOCIADO'].str.replace(',', '.').astype(float)
    df['RESULTADO DIA'] = df['CIERRE'] - df['APERTURA']
    df['PORCENTAJE'] = ((df['CIERRE'] - df['APERTURA']) / df['APERTURA'])
    return df


def norma_key(df, clave):
    v_min = np.min(df[clave])
    v_max = np.max(df[clave])

    #df[clave + '_N'] = (df[clave] - v_min) / (v_max - v_min)
    df[clave + '_N'] = ((df[clave] - df[clave].mean()) / df[clave].std()).round(4) # Normalización por z-score o Gaussian normalization
    return df


def normalizar(df):
    df_ret = pd.DataFrame(df)
    norma_key(df_ret, 'CIERRE')
    norma_key(df_ret, 'APERTURA')
    norma_key(df_ret, 'MAXIMO')
    norma_key(df_ret, 'MINIMO')
    norma_key(df_ret, 'PRECIO PROMEDIO')
    norma_key(df_ret, 'VOLUMEN NOMINAL')
    norma_key(df_ret, 'MONTO NEGOCIADO')
    return df_ret


def preparar_plot(plt_df):

    for clave_grupo in plt_df.groups.keys():
        print(clave_grupo)
    return plt_df
# %%
