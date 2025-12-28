import streamlit as st
import numpy as np
import pandas as pd 
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
from pages.ACP import df_transformed
from pages.Visualisation import data,colonne_specifie

st.markdown("<h4> Classification Hiérarchique Ascendante:</h4>", unsafe_allow_html=True)
test=data.copy()




st.write("copie une version de notre data pour CAH")
dff_cah=df_transformed.copy()
dff_cah['EMB']=data['EMB']
st.write(dff_cah.describe())
st.write(" La creation de fonction euclidean_distance pour calculer la distance euclidienne entre deux points.")
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))
code = ''' 
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

'''
st.code(code, language='python')
st.write(" La creation du fonction qui calcule la matrice de proximité entre les points (en utilisant la distance euclidienne).")
def matrice_prox(dff_cah):
    n = dff_cah.shape[0]  # Nombre de lignes
    matrix = np.zeros((n, n))  # Matrice vide
    for i in range(n):
        for j in range(n):
            matrix[i, j] = euclidean_distance(dff_cah.iloc[i], dff_cah.iloc[j])
    return matrix

code0 = ''' 
def matrice_prox(dff_cah):
    n = dff_cah.shape[0]  # Nombre de lignes
    matrix = np.zeros((n, n))  # Matrice vide
    for i in range(n):
        for j in range(n):
            matrix[i, j] = euclidean_distance(dff_cah.iloc[i], dff_cah.iloc[j])
    return matrix

'''
st.code(code0, language='python')

prox_matrix = matrice_prox(dff_cah)
st.write(" Affichage de la matrice de proximité")
st.dataframe(prox_matrix) 


st.write("Calcul la matrice de linkage avec la methode  Ward")
st.code("""
from scipy.cluster.hierarchy import dendrogram, linkage
# Calcul de la matrice de linkage avec la méthode de Ward
Z = linkage(dff_cah, method='ward')
""", language='python')

# Calcul de la matrice de linkage avec la méthode de Ward
Z = linkage(dff_cah, method='ward')

# Afficher la matrice de linkage
st.write("### Matrice de linkage (Z) :")
st.write(Z)  # Affiche la matrice de linkage sous forme de tableau


st.code("""plt.figure(figsize=(55, 45))
dendrogram(Z, labels=dff_cah.index, leaf_rotation=90)
plt.title('Dendrogramme de la Classification Ascendante Hiérarchique (CAH)')
plt.xlabel('Index des Points')
plt.ylabel('Distance Euclidienne')
plt.grid(True)
plt.show()""",language='python')

st.write("### Dendrogramme de la CAH :")
fig, ax = plt.subplots(figsize=(55, 45))  # Taille de la figure
dendrogram(Z, labels=dff_cah.index, leaf_rotation=90, ax=ax)
ax.set_title('Dendrogramme de la Classification Ascendante Hiérarchique (CAH)')
ax.set_xlabel('Index des Points')
ax.set_ylabel('Distance Euclidienne')
ax.grid(True)

# Afficher la figure dans Streamlit
st.pyplot(fig)






