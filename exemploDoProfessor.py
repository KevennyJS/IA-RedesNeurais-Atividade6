import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import Perceptron

# Carregando dados do arquivo CSV
url = 'https://raw.githubusercontent.com/alcidesbenicasa/IA---2020.1---Exerc-cio---06---Rede-Neural-Artificial/main/dados_pacientes_treinamento.csv'
base_Treinamento = pd.read_csv(url, sep=';', encoding='latin1').values
print("---------------------------------")
print("Dados dos Pacientes - TREINAMENTO")
print("---------------------------------")
print(base_Treinamento)
print("---------------------------------")

# Extração dos Atributos a serem utilizadas pela rede
print("Atributos de Entrada")
print("---------------------------------")
print(base_Treinamento[:, 1:5])

print("----------------------------")
print("Classificação Supervisionada")
print("----------------------------")
print(base_Treinamento[:, 5])

# Binarizador de rótulo
lb = preprocessing.LabelBinarizer()

# A saída da transformação é também conhecido como codificação 1-de-n
# Transforma valores categóricos equidistantes em valores binários equidistantes.
# Atributos categóricos com valores sim e não
lb.fit(['sim', 'não'])
febre = lb.transform(base_Treinamento[:, 1])
enjoo = lb.transform(base_Treinamento[:, 2])
dores = lb.transform(base_Treinamento[:, 4])

# Atributos categóricos com valores pequenas e grandes
lb.fit(['grandes', 'pequenas'])
manchas = lb.transform(base_Treinamento[:, 3])

# Atributos categóricos com valores saudável e doente
lb.fit(['saudável', 'doente'])
classes = lb.transform(base_Treinamento[:, 5])

# Concatenação de Atributos (Colunas)
atributos_norm = np.column_stack((febre, enjoo, manchas, dores))
print("--------------------------------")
print("Atributos de Entrada - Numéricos")
print("--------------------------------")
print(atributos_norm)

print("----------------------------------------")
print("Classificação Supervisionada - Numéricos")
print("----------------------------------------")
diagnostico_norm = np.hstack((classes))
print(diagnostico_norm)

# Treinamento do Perceptron a partir dos atributos de entrada e classificações
modelo = Perceptron()
modelo.fit(atributos_norm, diagnostico_norm)

# Acurácia do modelo, que é : 1 - (predições erradas / total de predições)
# Acurácia do modelo: indica uma performance geral do modelo.
# Dentre todas as classificações, quantas o modelo classificou corretamente;
# (VP+VN)/N
print('Acurácia: %.3f' % modelo.score(atributos_norm, diagnostico_norm))

Luiz = [[0, 0, 1, 1]]
print("Luiz", modelo.predict(Luiz))
Laura = [[1, 1, 0, 1]]
print("Laura", modelo.predict(Laura))
