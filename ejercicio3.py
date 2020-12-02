import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (12.0, 9.0)

# Preprocessing Input data
#Para este ejercicio utilice el mismo data set que el ejercicio anterior,donde es la informacion de casos de diabetes en distintos pacientes
data = pd.read_csv('diabetes.csv')
print(data.head(5))

#tome la primera columna que es la edad
X = data.iloc[:, 0]
#la tercera columna es el indice de masa corporal, decidi tomar esta columna para observar que tanta relación tienen el el IMC con la edad
Y = data.iloc[:, 2]
plt.scatter(X, Y)
plt.show()

m = 0
c = 0

L = 0.0001  # se declara la variable de la taza de aprendizaje 
epochs = 1000  # El numero de iteraciones para realizar el gradiente decendente

n = float(len(X)) # Number of elements in X
print(X)
print("************")
print(Y)

# Se realiza el gradiente descendente
for i in range(epochs): 
    Y_pred = m*X + c  # Primero se calcula el valor predictivo de Y utilizando la formula de la pendiente
    D_m = (-2/n) * sum(X * (Y - Y_pred))  # Se deriva m del valor calculado de Y
    D_c = (-2/n) * sum(Y - Y_pred)  # Se deriva el valor de c 
    m = m - L * D_m  # Se actulizan los valores
    c = c - L * D_c  
    
print (m, c)

Y_pred = m*X + c

#Se grafican los resultados obtenidos
plt.scatter(X, Y)
plt.plot([min(X), max(X)], [min(Y_pred), max(Y_pred)], color='red') 
plt.show()