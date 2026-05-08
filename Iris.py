#Proyecto KNNN con el data set iris.data
#Implementar un algortmo de clasificación por KNN con el iris.data, el algoritmo debe recibir un nuevo conjunto de datos y 
# clasificar cual es el tipo deiris al que pertenece.

import pandas as pd
import numpy as np
import random

#Importando data set iris.data

df_normal = pd.read_csv("/home/ferxxo/Documents/Algoritmos Estructura de Datos y Programación/Proyectos/Proyecto_2_KNNIris/iris.data", header = None)

print("Data frame original: \n",df_normal.head()) 

#Aleatorizando el data set y dividiendolo en un 80% para base y un 20% para prueba.
df_aleatorio = df_normal.sample(frac = 1, random_state= 42).reset_index(drop=True)

limite = int(len(df_aleatorio) * 0.8)
df_base = df_aleatorio.iloc[:limite]
df_prueba = df_aleatorio.iloc[limite:]

print("Data frame base: \n", df_base.head())
print("Data frame prueba: \n",df_prueba.head())

#creando clase iris y prueba para calcular la distancia entre dos objetos de la clase iris.

class iris():
    def __init__(self, sepalo_longitud, sepalo_ancho, petalo_longitud, petalo_ancho, clase):
        self.s_l = sepalo_longitud
        self.s_a = sepalo_ancho
        self.p_l = petalo_longitud
        self.p_a = petalo_ancho
        self.distancia = 0
        self.clase = clase
    
    def __str__(self):
        return f"Sepalo Longitud: {self.s_l}, Sepalo Ancho: {self.s_a}, Petalo Longitud: {self.p_l}, Petalo Ancho: {self.p_a}, Clase: {self.clase}"

def distancia(iris1, iris2):
    d = (iris1.s_l - iris2.s_l) ** 2 + (iris1.s_a - iris2.s_a) ** 2 + (iris1.p_l - iris2.p_l) ** 2 + (iris1.p_a - iris2.p_a) ** 2
    return np.sqrt(d)

#Iris de prueba
def iris_prueba(df_prueba):
    indice_random = random.randint(0, len(df_prueba) - 1)
    iris2 = iris(df_prueba.iloc[indice_random, 0], df_prueba.iloc[indice_random, 1], df_prueba.iloc[indice_random, 2], df_prueba.iloc[indice_random, 3], df_prueba.iloc[indice_random, 4])
    return iris2

iris2 = iris_prueba(df_prueba)

def calculoDistancia(df, iris2):
    distancias_calculadas = []
    for i in range(len(df)):
        iris1 = iris(df.iloc[i, 0], df.iloc[i, 1], df.iloc[i, 2], df.iloc[i, 3], df.iloc[i, 4])
        distancias_calculadas.append(distancia(iris1, iris2))
    return distancias_calculadas

distancias = calculoDistancia(df_base, iris2)
D = pd.DataFrame(distancias)

#Creando función para encontrar el valor mínimo con un valor 1 de vecinos (K = 1) y regrese el valor mínimo, así como el tipo de iris que pertenece en el df

def minimo(D,df):
    minimo = D.iloc[0, 0]
    indice = 0
    for i in D.iloc[:, 0]:
        if i < minimo:
            minimo = i
            indice = D[D.iloc[:, 0] == i].index[0]
    clase = df.iloc[indice, 4]
    iris_base = iris(df.iloc[indice, 0], df.iloc[indice, 1], df.iloc[indice, 2], df.iloc[indice, 3], df.iloc[indice, 4])    
    return minimo, clase, iris_base

#Desempeño del modelo

def multiplesSimulaciones(n):
    resultados = []
    errores = []
    for _ in range(n):
        print("Simulación número:", _ + 1)
        iris2 = iris_prueba(df_prueba)
        distancias = calculoDistancia(df_base, iris2)
        D = pd.DataFrame(distancias)
        _distancia_minima, clase_predicha, _iris_vecino = minimo(D, df_base)
        print("El iris de prueba es:", iris2.clase)
        print("El iris predicho es:", clase_predicha)
        if clase_predicha == iris2.clase:
            resultados.append(1)
            print("El modelo predijo correctamente el tipo de iris en la iteración")
        else:
            resultados.append(0)
            print("El modelo predijo incorrectamente el tipo de iris en la iteración")
            errores.append(_+1)
        print("\n")
    presicion = (sum(resultados) / len(resultados))*100

    return presicion, errores

presicion, errores = multiplesSimulaciones(50)

print("La precisión del modelo es:", presicion, "%")
print("Los errores se presentaron en las iteraciones número:", errores)