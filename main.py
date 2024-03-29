import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import Perceptron

mms = preprocessing.MinMaxScaler()


def preProcessar(base, isteste=False):
    aux = np.array(base[:, 0])
    pregnancies = mms.fit_transform(aux.reshape(-1, 1))

    aux = np.array(base[:, 1])
    glucose = mms.fit_transform(aux.reshape(-1, 1))

    aux = np.array(base[:, 2])
    bloodPressure = mms.fit_transform(aux.reshape(-1, 1))

    aux = np.array(base[:, 3])
    skinThickness = mms.fit_transform(aux.reshape(-1, 1))

    aux = np.array(base[:, 4])
    insulin = mms.fit_transform(aux.reshape(-1, 1))

    aux = np.array(base[:, 5])
    BMI = mms.fit_transform(aux.reshape(-1, 1))

    aux = np.array(base[:, 6])
    DPF = mms.fit_transform(aux.reshape(-1, 1))

    aux = np.array(base[:, 7])
    age = mms.fit_transform(aux.reshape(-1, 1))

    if isteste == False:
        outcome = base[:, 8]

    atributos_norm = np.column_stack((pregnancies, glucose, bloodPressure, skinThickness, insulin, BMI, DPF, age))
    print("--------------------------------")
    print("Atributos de Entrada - Numéricos")
    print("--------------------------------")
    print(atributos_norm)

    if isteste == False:
        print("----------------------------------------")
        print("Classificação Supervisionada - Numéricos")
        print("----------------------------------------")
        diagnostico_norm = np.hstack((outcome))
        print(diagnostico_norm)
        return atributos_norm, diagnostico_norm
    return atributos_norm

# ================================================= Carregando dados do arquivo CSV

url = 'https://raw.githubusercontent.com/KevennyJS/IA-RedesNeurais-Atividade6/main/diabetes.csv'
base_Treinamento = pd.read_csv(url, sep=',', encoding='latin1').values
print("---------------------------------")
print("Dados dos Pacientes - TREINAMENTO")
print("---------------------------------")
print(base_Treinamento)
print("---------------------------------")

# Extração dos Atributos a serem utilizadas pela rede
print("Atributos de Entrada")
print("---------------------------------")
print(base_Treinamento[:, 0:8])

print("----------------------------")
print("Classificação Supervisionada")
print("----------------------------")
print(base_Treinamento[:, 8])

# ================================================= PRE PROCESSAMENTO

atributos_norm, diagnostico_norm = preProcessar(base_Treinamento, False)

# ================================================= Treinamento do Neurônio Perceptron

modelo = Perceptron()
modelo.fit(atributos_norm, diagnostico_norm)
print('Acurácia: %.3f' % modelo.score(atributos_norm, diagnostico_norm))

# ================================================= Validação do Aprendizado

url_testes = 'https://raw.githubusercontent.com/KevennyJS/IA-RedesNeurais-Atividade6/main/testes.csv'
base_Testes = pd.read_csv(url_testes, sep=',', encoding='latin1').values
print("----------------------------")
print("Dados dos Pacientes - TESTES")
print("----------------------------")
print(base_Testes)
print("---------------------------------")

# Extração dos Atributos a serem utilizadas pela rede
print("Atributos de Entrada")
print("---------------------------------")
print(base_Testes[:, 0:8])

# ==========================================================
atributos_norm = preProcessar(base_Testes, True)

base_Predicao = modelo.predict((atributos_norm))
print("Classificações: ", base_Predicao)
