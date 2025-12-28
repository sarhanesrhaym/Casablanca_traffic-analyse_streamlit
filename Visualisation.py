import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import hvplot.pandas
from PIL import Image


st.header("visualisation des variables de notre dataset")

fichier = st.file_uploader(label="telecharger votre fichier",type=["xlsx"],accept_multiple_files=False)
# `data=pd.read_excel(fichier)` is reading the data from the Excel file uploaded by the user using
# Streamlit's file uploader widget. The uploaded Excel file is stored in the variable `fichier`, and
# `pd.read_excel(fichier)` reads the contents of this Excel file into a pandas DataFrame named `data`.
data=pd.read_excel(fichier)
st.dataframe(data.describe())

top=st.slider(label="Choisissez une valeur :",max_value=100,min_value=1,step=1)

st.dataframe(data.head(top))
st.text(f"les dimensions de notre data est : {data.shape[0]} lignes et {data.shape[1]} colonnes.")
st.subheader("voici l'histogramme de notre dataset:")

column = st.selectbox('Selectionner un colonne ', data.columns)
fig, ax = plt.subplots()
ax.hist(data[column], bins=20)
ax.set_xlabel(column)
ax.set_ylabel('Frequency')
ax.set_title(f'Histogram of {column}')
st.pyplot(fig)

st.subheader("Et voici les scatter plot:")
# Autres visualisations, par exemple, un scatter plot
colonne_x = st.selectbox('Sélectionnez la colonne pour l\'axe X', data.columns)
colonne_y = st.selectbox('Sélectionnez la colonne pour l\'axe Y', data.columns)

fig2, ax2 = plt.subplots()
ax2.scatter(data[colonne_x], data[colonne_y])
ax2.set_xlabel(colonne_x)
ax2.set_ylabel(colonne_y)
ax2.set_title(f'Scatter plot de {colonne_x} vs {colonne_y}')

# Afficher le scatter plot dans Streamlit
st.pyplot(fig2)
st.write("pour maintenant, on va calculer le moyen des colonnes de TTI dans un seul colonne et on va creer notre variable target nomme: EMB en executant ses codes:")
code= '''
colonne_specifie=['TTI at  0','TTI at  1','TTI at  2','TTI at  3','TTI at  4','TTI at  5','TTI at  6','TTI at  7','TTI at  8','TTI at  9','TTI at  10','TTI at  11','TTI at  12','TTI at  13','TTI at  14','TTI at  15','TTI at  16','TTI at  17','TTI at  18','TTI at  19','TTI at  20','TTI at  21','TTI at  22','TTI at  23']
data['TTI'] = data[colonne_specifie].mean(axis=1)
data['TTI']=data['TTI'].astype(float)
def trns_emb(tti):
    if 1.0 <= tti < 2.0:
        return 0
    elif 2.0 <= tti < 3.0:
        return 1
    elif tti >= 3.0:
        return 2
    return None  

data['EMB'] = data['TTI'].apply(trns_emb)


'''
st.code(code,language='python')

colonne_specifie=['TTI at  0','TTI at  1','TTI at  2','TTI at  3','TTI at  4','TTI at  5','TTI at  6','TTI at  7','TTI at  8','TTI at  9','TTI at  10','TTI at  11','TTI at  12','TTI at  13','TTI at  14','TTI at  15','TTI at  16','TTI at  17','TTI at  18','TTI at  19','TTI at  20','TTI at  21','TTI at  22','TTI at  23']
data['TTI'] = data[colonne_specifie].mean(axis=1)
data['TTI']=data['TTI'].astype(float)
def trns_emb(tti):
    if 1.0 <= tti < 2.0:
        return 0
    elif 2.0 <= tti < 3.0:
        return 1
    elif tti >= 3.0:
        return 2
    return None  

data['EMB'] = data['TTI'].apply(trns_emb)
st.write("le code est bien execute")
st.dataframe(data.describe())
st.write("puis on va visualiser notre variable target 'EMB' :")
st.write("\n")
st.write("on comence par un countplot :")
# Select a column to plot
column = st.selectbox('Select column for count plot', data.columns)

# Create a count plot using Seaborn
plt.figure(figsize=(30,30))
fig, ax = plt.subplots()
sns.countplot(x=column,hue="EMB", data=data, ax=ax)
ax.set_title(f'Count Plot of {column}')

# Display the count plot in Streamlit
st.pyplot(fig)

dff=data.copy()

st.write("La distribution de variable target est :")
# Calculate the frequency and percentage distribution of the 'Attrition' column
attrition_freq = dff[['EMB']].apply(lambda x: x.value_counts())  # Fill in the blanks to select the column and count values
attrition_freq['frequency_percent'] = round((100 * attrition_freq / attrition_freq.sum()), 2)  # Fill in the blank to calculate percentages
st.write(attrition_freq)
plot = attrition_freq[['frequency_percent']].plot(kind="bar")  # Fill in the blank to select the percentage column for plotting
plot.set_title("EMB Distribution", fontsize=10)
plot.grid(color='lightgray', alpha=0.5)
st.pyplot(plot.figure)

st.write("notre heatmap est sous la forme :")

st.write("CORRELATION HEATMAP")
numeric_data=dff.select_dtypes(include=[np.number])
correlation_matrix= numeric_data.corr()
plt.figure(figsize=(45,45))
sns.heatmap(correlation_matrix,annot=True,cmap='coolwarm',fmt='.2f',cbar=True)
plt.title("Correlation Heatmap")
st.pyplot(plt)

st.write("l'embouteillage dans chaque cmmune est distribue sous la forme :")
CommuneCrossTab =pd.crosstab(dff['COMMUNE'], dff['EMB'], margins=True)
st.write(CommuneCrossTab)
st.write("Les 3 photos ci-dessous represente l'execution de la code suivant:")
st.code('''
        dff.hvplot.hist(y='N° of Secondery roads', by='EMB', subplots=False, width=900, height=300, bins=60)
        ''')

st.write("on voit que si la distance augmente la chance d'avoir une embouteillage forte diminu.")
# Load an image from a file
image = Image.open('pages\img\img0.jpg')

# Display the image
st.image(image, caption="Representation de l'emouteillage faible")

# Load an image from a file
image = Image.open('pages\img\img1.jpg')

# Display the image
st.image(image, caption="Representation de l'emouteillage moyenne")

# Load an image from a file
image = Image.open('pages\img\img2.jpg')

# Display the image
st.image(image, caption="Representation de l'emouteillage forte")
st.write("\n")
st.write("Et apres la visualisation on peut conclure :")
st.write("LE POURCENTAGE DE VARIABLE TARGE 'EMB' SOIT AUGMENTE SOIT DIMUNUE PAR RAPPORT A CES VARIABLES SUIVANTES:")

st.write("1) Distance(m)") 
st.write("si la distance entre deux points dans une commune quelconque est grand , alors directement le pourcentage d'emboutaillage est DIMINUE")
st.write("2) N° of Highways") 
st.write("si une commune possede un nombre elevee des routes HIGHWAYS (c-a-d routes de bon qualite) , alors le pourcentage d'emboutaillage dans cette commune DIMINUE")
st.write("3) tram-Station")
st.write("si le nombre de station de TRAM est grand dans une commune, donc le pourcentage d'emboutaillage DIMINUE")
st.write("4) Bus-Station")
st.write("si le nombre de station de BUS est grand dans une commune, donc le pourcentage d'emboutaillage AUGMENTE")
st.write("5) Density") 
st.write("si la density d'une population dans une commune est grande , Donc le pourcentage d'emboutaillage AUGMENTE")
st.sidebar.success("Lorsque vou terminer La visualisation, passer a la standarisation")

