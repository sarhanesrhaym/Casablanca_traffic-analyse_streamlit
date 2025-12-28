import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pages.Standarisation import df,colonne_specifie,target
import numpy as np
from sklearn.decomposition import PCA

dff=df.copy()
target.reset_index(drop=True, inplace=True)
st.title("ACP")
dff=dff.drop(columns=['EMB'])
dff=dff.drop(columns=colonne_specifie)
st.write("Dans cette partie on va reduire la dimensionnalite de notre dataset.")
st.write("Les donnes actuels sont :")
top=st.slider(label="Choisissez le nombre des lignes :",max_value=100,min_value=1,step=1)

st.dataframe(dff.head(top))
st.write("et la matrice de coorelation est:")
corr_matrix = dff.corr()

# Afficher la matrice de corrélation dans Streamlit
st.write("Matrice de corrélation:")
st.dataframe(corr_matrix)

# Visualiser la matrice de corrélation avec Seaborn
st.write("Heatmap de la matrice de corrélation:")
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
st.pyplot(plt)
plt.clf()

eigenvalues, eigenvectors = np.linalg.eig(corr_matrix)

total_variance = np.sum(eigenvalues)
contributions = [(eigenvalue / total_variance) * 100 for eigenvalue in eigenvalues]

cumulative_contributions = np.cumsum(contributions)

# Afficher les valeurs propres
st.write("Valeurs propres :")
st.write(eigenvalues)

# Afficher les vecteurs propres
st.write("Vecteurs propres :")
st.write(eigenvectors)

st.write("Contributions de chaque axe (% de variance expliquée) :")
st.write(contributions)

# Afficher les contributions cumulatives de chaque axe
st.write("Contributions cumulatives de chaque axe (% de variance expliquée) :")
st.write(cumulative_contributions)

st.write("D'apres le tableau ci-dessus, on peut conclure que 7 axes est le meilleur choix pour l'acp car il contient plus que 95% des informations.")
st.write("\n")
st.write("on va executer le code suivant :")
st.code('''pca = PCA(n_components=7)

pca.fit(dff)
        ''')
pca = PCA(n_components=7)

dfpca = pca.fit_transform(dff)

# Définir la fonction de tracé
def plot_var_explique(acp):
    var_explique = acp.explained_variance_ratio_
    plt.bar(np.arange(len(var_explique)) + 1, var_explique)
    plt.plot(np.arange(len(var_explique)) + 1, var_explique.cumsum(), c="red", marker='o')
    plt.xlabel("Rang de l'axe d'inertie")
    plt.ylabel("Pourcentage d'inertie")
    plt.title("Éboulis des valeurs propres")
    plt.show(block=False)
    

# Interface Streamlit
st.title("Visualisation de l'ACP")

# Appeler la fonction de tracé et afficher dans Streamlit
fig = plot_var_explique(pca)
st.pyplot(fig)
plt.clf()


df_transformed = pd.DataFrame(dfpca, columns=['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7'])

# Interface Streamlit
st.title("Choix des Colonnes pour le Scatter Plot")

# Multiselect pour choisir les colonnes
columns = df_transformed.columns.tolist()
selected_columns = st.multiselect('Sélectionnez les colonnes à tracer:', columns, default=['C1', 'C2'])

# Vérifier que deux colonnes ont été sélectionnées
if len(selected_columns) == 2:
    # Créer le scatter plot en utilisant les colonnes sélectionnées
    sns.scatterplot(data=df_transformed, x=selected_columns[0], y=selected_columns[1])
    plt.title('Projection des Données sur les Deux Composantes Principales')
    plt.xlabel(selected_columns[0])
    plt.ylabel(selected_columns[1])
    plt.grid(True)
    
    # Afficher le plot dans Streamlit
    st.pyplot(plt.gcf())
    plt.clf()
else:
    st.write("Veuillez sélectionner exactement deux colonnes.")
    


df_transformed['EMB']=target
st.write("maintenant on va passer a la classification hiearchique")
st.sidebar.success("Lorsque vou terminer L'ACP, passer a CAH")


