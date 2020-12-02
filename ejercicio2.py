import pandas as pd 
import numpy as np
import sklearn
#import mglearn
from sklearn.datasets import  load_diabetes
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.express as px

#Primeramente se importa el data set de la libreria de sklearn, en este caso decidi utilizar un data set correspondiente a casos de diabetes en donde los
#datos muestran componentes como: edad, sexo, IMC, nivel de azucar en la sangre, entre otros.
diabetes = load_diabetes()
caracteristicas = diabetes.feature_names
print(caracteristicas)


pca=PCA(n_components=2)
pca.fit(diabetes.data)
#print(principalComponents)
transformada=pca.transform(diabetes.data)
print("Valores del data set original: ")
print(diabetes.data.shape)
print("Valores del data set despues de tranformar los datos: ")
print(transformada.data.shape)

#se grafican los componentes principales con el target de los datos
plt.scatter(transformada[:, 0], transformada[:, 1],c=diabetes.target, edgecolor='none', alpha=0.5,cmap=plt.cm.get_cmap('Blues_r', 10))

plt.xlabel('component 1')
plt.ylabel('component 2')
plt.colorbar()
plt.show()


dpca = PCA(n_components=1)
pca.fit(diabetes.data)
X_pca = pca.transform(diabetes.data)
X_new = pca.inverse_transform(X_pca)
plt.scatter(transformada[:, 0], transformada[:, 1], alpha=0.2)
plt.title('Comparacion de componentes principales')
plt.scatter(X_new[:, 0], X_new[:, 1], alpha=0.8)
plt.axis('equal')
plt.show();


#Se grafica la varianza explicada acumulativa 
pca = PCA().fit(diabetes.data)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('numero de componentes principales')
plt.ylabel('varianza explicada acumulativa')
plt.show()

