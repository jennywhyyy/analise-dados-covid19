#!/usr/bin/env python
# coding: utf-8

# # Desafio de projeto - DIO | Criação de modelos com Python e Machine Learning para prever a evolução da COVID-19 no Brasil:

# In[68]:


#importação das bibliotecas:

import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import re
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose


# In[5]:


#importação dos dados que serão usados na análise:

df_covid = pd.read_csv("covid_19_data.csv", parse_dates = ['ObservationDate', 'Last Update'])


# In[6]:


#verificando as 5 primeiras linhas:

df_covid.head()


# In[7]:


#contabilizando quantas linhas e colunas há no dataframe:

df_covid.shape


# In[8]:


#conferindo quais são os tipos de dados:

df_covid.dtypes


# In[9]:


#criando uma função para transformar todos os caracteres dos nomes das colunas em caracteres minúsculos:

def correcao_colunas(col_name):
    return re.sub(r"[/| ]", "", col_name).lower()


# In[10]:


#aplicando a função e tornando os caracteres maiúsculos em minúsculos:

df_covid.columns = [correcao_colunas(coluna) for coluna in df_covid.columns]


# In[11]:


#verificando as alterações:

df_covid.head()


# ## Análises:

# In[12]:


#verificando quais são as informações disponíveis relativas ao Brasil:

df_covid.loc[df_covid.countryregion == "Brazil"]


# In[35]:


#criando um novo dataframe, com dados relativos apenas ao Brasil e aos casos confirmados da doença:

df_brasil = df_covid.loc[(df_covid.countryregion == "Brazil") & (df_covid.confirmed > 0)]


# In[14]:


#verificando as 5 primeiras linhas do novo dataframe criado:

df_brasil.head()


# In[16]:


#plotando um chart line para visualização dos dados, total de casos por mês:


px.line(df_brasil, 'observationdate', 'confirmed', labels={'observationdate':'data', 'confirmed':'número de casos confirmados'}, title='casos confirmados no brasil')


# In[37]:


#criaçao de uma função para contabilizar a quantidade de novos casos por dia no brasil:

df_brasil["novos_casos"] = list(map(lambda x: 0 if (x == 0) else df_brasil['confirmed'].iloc[x] - df_brasil['confirmed'].iloc[x-1], np.arange(df_brasil.shape[0])))


# In[38]:


#verificando se a nova coluna "novos_casos" foi criada:

df_brasil.head()


# In[39]:


#visualização da progressão dos novos casos:

px.line(df_brasil, x = 'observationdate', y = 'novos_casos', title = 'novos casos por dia',
       labels = {'observationdate':'data', 'novos_Casos':'novos casos'})


# In[40]:


#visualizando o número de mortes por covid-19 no brasil:

brasil_covid_mortes = go.Figure()

brasil_covid_mortes.add_trace(go.Scatter(x = df_brasil.observationdate, y = df_brasil.deaths,
                                         name = 'mortes por covid-19',
                                         mode = 'lines+markers',
                                         line = dict(color = 'red')))

brasil_covid_mortes.show()


# In[42]:


#adicionando as legendas ao gráfico anterior:

brasil_covid_mortes.update_layout(title = 'mortes por covid-19 no brasil',
                                 xaxis_title = 'data',
                                 yaxis_title = 'número de mortes')

brasil_covid_mortes.show()


# In[48]:


#criando uma função para calcular a taxa de crescimento da covid-19, desde o primeiro caso:

def taxa_crescimento(data, variable, data_inicio=None, data_fim=None):
    # Se data_inicio for None, define como a primeira data disponível no dataset
    if data_inicio == None:
        data_inicio = data.observationdate.loc[data[variable] > 0].min()
    else:
        data_inicio = pd.to_datetime(data_inicio)
        
    if data_fim == None:
        data_fim = data.observationdate.iloc[-1]
    else:
        data_fim = pd.to_datetime(data_fim)
    
    # Define os valores de presente e passado
    passado = data.loc[data.observationdate == data_inicio, variable].values[0]
    presente = data.loc[data.observationdate == data_fim, variable].values[0]
    
    # Define o número de pontos no tempo q vamos avaliar
    tempo = (data_fim - data_inicio).days
    
    # Calcula a taxa
    taxa = (presente/passado)**(1/tempo) - 1

    return taxa*100
    
    
    


# In[49]:


#calculando a taxa de crescimento médio:
cresc_medio = taxa_crescimento(df_brasil, 'confirmed')

print(f"O crescimento médio do COVID no Brasil no período avaliado foi de {cresc_medio.round(2)}%.")


# In[57]:


#definindo uma função para calcular a taxa de crescimento por dia:

def taxa_crescimento_diaria(data, variable, data_inicio = None):
    if data_inicio == None:
        data_inicio = data.observationdate.loc[data[variable] > 0].min()
    else:
        data_inicio = pd.to_datetime(data_inicio)
    
    data_fim = data.observationdate.max()
    tempo = (data_fim - data_inicio).days
    
    taxa = list(map(lambda x: (data[variable].iloc[x] - data[variable].iloc[x-1]) / data[variable].iloc[x-1],
        range(1,tempo+1)))
    
    return np.array(taxa)*100


# In[58]:


#calculando a taxa ao dia:

taxa_dia = taxa_crescimento_diaria(df_brasil, 'confirmed')

taxa_dia


# In[59]:


#calculando qual foi a data da primeira observação:

primeiro_dia = df_brasil.observationdate.loc[df_brasil.confirmed > 0].min()
primeiro_dia


# In[61]:


#plotando o gráfico de crescimento do número de casos, da data da primeira observação até a última:

px.line(x = pd.date_range(primeiro_dia, df_brasil.observationdate.max())[1:],
       y = taxa_dia,
       title = 'taxa de crescimento dos casos confirmados no brasil',
       labels = {'y':'taxa de crescimento', 'x':'data'})


# ## Predições:

# In[66]:


# construindo um modelo de séries temporais para realizar as predições de novos casos:

novos_casos = df_brasil.novos_casos
novos_casos.index = df_brasil.observationdate

res = seasonal_decompose(novos_casos)

fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize = (10, 8))
ax1.plot(res.observed)
ax2.plot(res.trend)
ax3.plot(res.seasonal)
ax4.scatter(novos_casos.index, res.resid)
ax4.axhline(0, linestyle = 'dashed', c = 'black')

plt.show()


# In[69]:


#instalação do pacote do PMDARIMA:
get_ipython().system('pip install pmdarima')


# In[70]:


#importando o módulo do AUTO-ARIMA do pacote pmdarima:
from pmdarima.arima import auto_arima


# In[73]:


#selecionando apenas os casos confirmados e suas respectivas datas:

casos_confirmados = df_brasil.confirmed

casos_confirmados.index = df_brasil.observationdate


# In[74]:


#observando o novo dataframe:

casos_confirmados.head()


# In[75]:


#plotando a figura relativa à evolução dos casos confirmados:

res_2 = seasonal_decompose(casos_confirmados)

fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize = (10, 8))
ax1.plot(res_2.observed)
ax2.plot(res_2.trend)
ax3.plot(res_2.seasonal)
ax4.scatter(casos_confirmados.index, res_2.resid)
ax4.axhline(0, linestyle = 'dashed', c = 'black')

plt.show()


# In[76]:


#construindo o modelo preditivo, com base nos casos confirmados:

modelo = auto_arima(casos_confirmados)


# In[79]:


#verificando o resultado:

modelo


# In[80]:


#plotando o gráfico relativo à predição de futuros casos, para 15 dias após o intervalo entre 20-05-20 e 20-06-05:

predicao = go.Figure(go.Scatter(x = casos_confirmados.index, y = modelo.predict_in_sample(), name = 'Predicted'))

predicao.add_trace(go.Scatter(x = pd.date_range('2020-05-20', '2020-06-05'), y = modelo.predict(15), name = 'Forecast'))

predicao.update_layout(title = 'Predição de casos confirmados para os próximos 15 dias',
                      yaxis_title = 'casos confirmados', xaxis_title = 'data')

predicao.show()

