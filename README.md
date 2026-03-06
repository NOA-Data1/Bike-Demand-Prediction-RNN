# 📊 Previsão de Demanda de Aluguel de Bicicletas em Dublin usando Redes Neurais Recorrentes (RNN)

## 📌 Descrição do Projeto
Este projeto tem como objetivo prever a demanda diária de aluguel de bicicletas em Dublin utilizando modelos de Deep Learning baseados em séries temporais.

A previsão de demanda pode ajudar empresas de mobilidade urbana a otimizar a distribuição de bicicletas, planejar manutenção e melhorar a experiência dos usuários.

---

## 🎯 Problema de Negócio (Business Problem)

Sistemas de compartilhamento de bicicletas precisam prever a demanda futura para garantir que bicicletas estejam disponíveis nos locais corretos.

Previsões precisas permitem:

- melhor distribuição de bicicletas
- planejamento de operações
- redução de custos operacionais
- melhor experiência para usuários

Neste projeto, utilizamos **Redes Neurais Recorrentes (RNN)** para prever o número de viagens diárias.

---

## 🛠 Tecnologias Utilizadas

- Python
- PySpark
- Apache Hadoop
- TensorFlow
- Keras
- Matplotlib
- Seaborn

---

## 📂 Dataset

O dataset contém registros diários de uso do sistema de bicicletas compartilhadas em Dublin.

Os dados foram processados utilizando **PySpark** e armazenados no **Hadoop Distributed File System (HDFS)**.

Principais etapas de processamento:

- limpeza dos dados
- conversão de timestamps
- agregação de viagens por dia

---

## 🔎 Análise Exploratória (EDA)

Durante a análise exploratória foram avaliados:

- tendência da série temporal
- padrões de sazonalidade
- distribuição da demanda diária
- identificação de outliers

Também foi realizada **decomposição da série temporal** para separar:

- tendência
- sazonalidade
- ruído

---

## 🤖 Modelos Utilizados

Foram treinados três modelos de Deep Learning:

**1️⃣ LSTM Simples**
- uma camada LSTM
- arquitetura básica

**2️⃣ LSTM com otimização**
- múltiplas camadas
- regularização com dropout
- ajuste de hiperparâmetros

**3️⃣ GRU**
- arquitetura similar ao LSTM otimizado
- menor complexidade computacional

---

## 📈 Métrica de Avaliação

A performance dos modelos foi avaliada utilizando:

**RMSE (Root Mean Squared Error)**

Esta métrica mede a diferença entre os valores previstos e os valores reais.

---

## 🚀 Resultados

O modelo **GRU apresentou o menor RMSE**, indicando melhor capacidade de capturar dependências temporais da série.

Mesmo assim, os resultados indicam que a inclusão de mais variáveis pode melhorar a precisão.

---

## ⚙️ Como Executar o Projeto

1️⃣ Clonar o repositório
delos híbridos**
- tuning mais avançado de hiperparâmetros
