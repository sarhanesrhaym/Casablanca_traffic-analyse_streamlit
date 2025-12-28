import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pages.Visualisation import data
import numpy as np
import scipy.stats as stats


st.title("la standarisation de donnees:")
st.write("#La STANDARISTAION est une technique utilisée  en analyse de données pour transformer des variables en leur donnant une moyenne de 0 et un écart-type de 1. Cela permet d'améliorer la performance des modèles, en particulier ceux qui sont sensibles à l’échelle des données.#\n")

st.write("On commence par verification s'il  y a des valeurs nulles en executant :")
st.code("sns.heatmap(df.isna() , cbar=False) ")
df=data.copy()
fig,ax= plt.subplots()
sns.heatmap(df.isna() , cbar=False) 
st.pyplot(fig)

st.write("on voit qu'il n'y a pas des valeurs nulles.")
st.write("\n")

st.write("Apres on elimine les valeurs aberents en utilisant la methode de l'IQR")
st.write("Le code est :")
st.code('''
        colonne_specifie=['TTI at  0','TTI at  1','TTI at  2','TTI at  3','TTI at  4','TTI at  5','TTI at  6','TTI at  7','TTI at  8','TTI at  9','TTI at  10','TTI at  11','TTI at  12','TTI at  13','TTI at  14','TTI at  15','TTI at  16','TTI at  17','TTI at  18','TTI at  19','TTI at  20','TTI at  21','TTI at  22','TTI at  23']
# Pour chaque colonne numérique, calculer les bornes et filtrer les valeurs aberrantes
for colonne in colonne_specifie :
    # Calcul des quartiles et de l'IQR
    Q1 =df[colonne].quantile(0.25)  # Premier quartile
    Q3 = df[colonne].quantile(0.75)  # Troisième quartile
    IQR = Q3 - Q1  # Intervalle interquartile
    
    # Définir les bornes
    borne_inferieure = Q1 - 1.5 * IQR
    borne_superieure = Q3 + 1.5 * IQR
    
    # Supprimer les lignes avec des valeurs aberrantes pour cette colonne
    df = df[(df[colonne] >= borne_inferieure) & (df[colonne] <= borne_superieure)]''')


colonne_specifie=['TTI at  0','TTI at  1','TTI at  2','TTI at  3','TTI at  4','TTI at  5','TTI at  6','TTI at  7','TTI at  8','TTI at  9','TTI at  10','TTI at  11','TTI at  12','TTI at  13','TTI at  14','TTI at  15','TTI at  16','TTI at  17','TTI at  18','TTI at  19','TTI at  20','TTI at  21','TTI at  22','TTI at  23']
# Pour chaque colonne numérique, calculer les bornes et filtrer les valeurs aberrantes
for colonne in colonne_specifie :
    # Calcul des quartiles et de l'IQR
    Q1 =df[colonne].quantile(0.25)  # Premier quartile
    Q3 = df[colonne].quantile(0.75)  # Troisième quartile
    IQR = Q3 - Q1  # Intervalle interquartile
    
    # Définir les bornes
    borne_inferieure = Q1 - 1.5 * IQR
    borne_superieure = Q3 + 1.5 * IQR
    
    # Supprimer les lignes avec des valeurs aberrantes pour cette colonne
    df = df[(df[colonne] >= borne_inferieure) & (df[colonne] <= borne_superieure)]
    

st.write(f"les nouveaux dimensions sont: {df.shape[0]} lignes et {df.shape[1]} colonnes.")

df.reset_index(drop=True, inplace=True)
st.write("maintenant on fait l'encodage de variable COMMUNE et on obtient:")
commune_codes = {
'Echchallalate':1,
'Sidi Maarouf':2,
'Anfa':3,
'Assoukhour Assawda':4,

'Ben Msick':5,

'Bou Chentouf':6,

'El Maarif':7,

'Hay Hassani':8,

'Hay Mohammadi':9,

'Ain Chock':10,

'Ain Sebaa':11,

'Al Fida':12,

'Al Idrissia':13,

'Lissasfa':14,

'Mers Sultan':15,

'Moulay Rachid':16,

'Sidi Belyout':17,

'Sidi Moumen':18,

'Sidi Othmane':19,

'Ahl Laghlam':20,

'Sidi Bernoussi':21,

'Moulay Youssef':22

}

df['COMMUNE'] = df['COMMUNE'].map(commune_codes)

top=st.slider(label="Choisissez le nombre de lignes :",max_value=100,min_value=1,step=1)
st.dataframe(df.head(top))

st.write("et leur description est :")
st.dataframe(df.describe())
st.write("d'apres le tableau ci-dessus on voit que:")
st.write("1) le moyen n'est pas proche a le mediane.")
st.write("2) le min et le max est eloignee de le median plusou moins l'ecart-type")
st.write("ce qui indique que la distribution n'est pas normale, et pour verifier ca, on utilise le test de SHAPIRO-WILK de backage SCIPY.STATS a but de determiner le type de distribution de notre data")

st.title("le test de SHAPIRO-WILK 🖥️")

code = ''' 
import scipy.stats as stats
n_data=dff.select_dtypes(include=[np.number]) # n_data est une data continne juste les variables numeriques
stat, p_value = stats.shapiro(n_data)
print(f"Test de Shapiro-Wilk: Stat={stat:.3f}, p={p_value:.3f}")
if p_value > 0.05:
    print("Les données suivent une distribution normale ")
else:
    print("Les données ne suivent PAS une distribution normale ")
'''

st.code(code, language='python')


n_data=df.copy()
stat, p_value = stats.shapiro(n_data)
print(f"Test de Shapiro-Wilk: Stat={stat:.3f}, p={p_value:.3f}")
if p_value > 0.05:
    st.write("Les données suivent une distribution normale ")
else:
    st.write("Les données ne suivent PAS une distribution normale ")

st.write("*Dans notre cas on utilise l'un des methodes de FEATURE SCALING est la standarisation:")

st.write("Le code python pour standrisee notre data:")

code2='''
mean = np.mean(n_data, axis=0) #calculer le moyenne de n_data
std = np.std(n_data, axis=0) #calculer l'ecart-type de n_data

# Standardisation
n_data = (n_data - mean) / std

'''
st.code(code2, language='python')
mean = np.mean(df, axis=0) #calculer le moyenne de df
std = np.std(df, axis=0) #calculer l'ecart-type de df

target=df['EMB'].copy()
df=df.drop(columns=['EMB'])
df=df.drop(columns=colonne_specifie)
st.write("on va supprimer les colonnes de TTI et conserver juste le colonne qui contient le moyen")
st.write("et on va resrever le variable 'EMB', ca sera notre variable cible.")
# Standardisation
df = (df - mean) / std
st.write("en applicant ce code,on obtient :")
top=st.slider(label="Choisissez le nombre des lignes :",max_value=100,min_value=1,step=1)


df['EMB']=target
st.dataframe(df.head(top))

    # Afficher des statistiques descriptives avec describe()
st.write("Statistiques descriptives :")
st.write(df.describe())

st.write("maintenant on va passer a l'acp.")
st.sidebar.success("Lorsque vou terminer La standarisation, passer a l'acp")