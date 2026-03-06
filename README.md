## 📊 Previsão de Demanda de Aluguéis de Bicicletas utilizando Redes Neurais Recorrentes (RNN)

## 📌 Objetivo
Este projeto tem como objetivo prever a demanda diária de bicicletas compartilhadas em Dublin utilizando modelos de Deep Learning baseados em séries temporais. A previsão de demanda pode ajudar operadores de mobilidade urbana a otimizar alocação de bicicletas, planejamento operacional e manutenção.

---

## 📂 Dataset
O dataset contém registros diários de utilização do sistema público de bicicletas de Dublin.

Principais variáveis analisadas:
- Data
- Número total de viagens
- Distribuição temporal das viagens

---

## ⚙️ Processamento de Dados
O processamento de dados foi realizado utilizando tecnologias de Big Data.

Etapas:
- Ingestão de dados com **PySpark**
- Armazenamento em **Hadoop HDFS**
- Limpeza e transformação de dados
- Agregação de viagens por dia

---

## 🔎 Análise Exploratória de Dados (EDA)

Foram realizadas análises para entender o comportamento da demanda:

- Estatísticas descritivas
- Análise de distribuição
- Detecção de outliers
- Análise de tendência e sazonalidade
- Decomposição de séries temporais

Visualizações utilizadas:

- Histogramas
- Boxplots
- Scatter plots
- Decomposição temporal

---

## 🤖 Modelos de Machine Learning

Foram desenvolvidos três modelos baseados em redes neurais recorrentes:

- **Simple LSTM**
- **LSTM com otimização de hiperparâmetros**
- **GRU (Gated Recurrent Unit)**

Os modelos foram implementados utilizando:

- **TensorFlow**
- **Keras**

---

## 📈 Avaliação dos Modelos

A performance foi avaliada utilizando a métrica:

**RMSE (Root Mean Squared Error)**

Comparação entre:

- dataset de treino
- dataset de teste

---

## 🚀 Resultados

O modelo **GRU** apresentou melhor desempenho na previsão da demanda diária, demonstrando maior capacidade de capturar dependências temporais da série.

Observou-se forte padrão de **sazonalidade**, com maior uso durante meses mais quentes e dias úteis.

---

## 🛠 Tecnologias Utilizadas

- Python
- PySpark
- Hadoop
- TensorFlow
- Keras
- Matplotlib
- Seaborn

---

## 🔮 Melhorias Futuras

Possíveis melhorias para aumentar a precisão do modelo:

- inclusão de **dados meteorológicos**
- inclusão de **feriados e eventos**
- utilização de **modelos híbridos**
- tuning mais avançado de hiperparâmetros
