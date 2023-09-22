# Bibliotecas

# conda install pandas
import pandas as pd

# conda install -c anaconda scikit-learn
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

from sklearn import tree

# conda install python-graphviz
import graphviz

# Obtenemos el dataset a entrenar
df = pd.read_csv('ratings.csv')

# --- PREPARACION DE DATOS ---

# Agrupo los tipos de contenido
df['Title Type'] = df['Title Type'].replace('tvMovie', 'Movie')
df['Title Type'] = df['Title Type'].replace('tvMiniSeries', 'tvSeries')
df['Title Type'] = df['Title Type'].replace('tvSpecial', 'short')
df['Title Type'] = df['Title Type'].replace('tvEpisode', 'short')
df['Title Type'] = df['Title Type'].replace('video', 'short')
df['Title Type'] = df['Title Type'].replace('tvShort', 'short')

# Borro las columnas no utilizadas
df.drop('Const', axis=1, inplace=True)
df.drop('URL', axis=1, inplace=True)
df.drop('Date Rated', axis=1, inplace=True)
df.drop('Release Date', axis=1, inplace=True)
df.drop('Runtime (mins)', axis=1, inplace=True)
df.drop('Directors', axis=1, inplace=True)

# Verificamos que todas las tuplas tengan datos
print(df.info())
# #   Column              Non-Null Count  Dtype  
#---  ------              --------------  -----  
# 0   Your Rating         408 non-null    int64  
# 1   Title               408 non-null    object 
# 2   Title Type          408 non-null    object 
# 3   IMDb Rating         407 non-null    float64
# 4   Runtime (mins)      400 non-null    float64
# 5   Year                408 non-null    int64  
# 6   Genres              408 non-null    object 
# 7   Num Votes           407 non-null    float64
# 8   Directors           288 non-null    object 
# 9   Like                408 non-null    bool   
# 10  Year Rated          408 non-null    int32  
# 11  Month-Date Rated    408 non-null    object 
# 12  Month-Day Released  408 non-null    object 

# Completamos el IMDb Rating y NumVotes del titulo al cual le faltaba 
promedio = df['IMDb Rating'].mean()
df['IMDb Rating'].fillna(promedio, inplace=True)
df['Num Votes'].fillna(0, inplace=True)

# Transformamos features
megusta = 7
nicho = 25000
mainstream = 600000
df['Like'] = df['Your Rating'].apply(lambda x: True if x >= megusta else False)
df['Niche'] = df['Num Votes'].apply(lambda x: True if x <= nicho else False)
df['Mainstream'] = df['Num Votes'].apply(lambda x: True if x >= mainstream else False)
df.drop('Your Rating', axis=1, inplace=True)
df.drop('Num Votes', axis=1, inplace=True)

# Los generos estan separados por coma, por lo que los dividimos y aplicamos one-hot encoding
df['Genres'] = df['Genres'].str.split(', ')
genres_dummies = df['Genres'].str.join('|').str.get_dummies()
genres_dummies = genres_dummies.add_prefix('Genres_')
df = pd.concat([df.drop(columns=['Genres']), genres_dummies], axis=1)

# Se aplica one-hot encoding a los tipos de contenido
df = pd.get_dummies(data=df, columns=['Title Type'], drop_first=True)

# Separamos las features y target

#features
x = df.drop(columns=['Title', 'Like'])

#target
y = df['Like']

# Dividimos el dataset en sets de entrenamiento y de testeo
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Creamos el modelo
model = DecisionTreeClassifier(max_depth=4, min_samples_leaf=5)

# Entrenamos el modelo
model.fit(X_train,y_train)

# pasamos las features y el target para que nos diga que tan bien predice
print("Prediccion resultante")
print(model.score(X_test,y_test))

# Visualizando el arbol
dot_data = tree.export_graphviz(model, out_file=None, feature_names=x.columns.tolist(), class_names=['False', 'True'], filled=True, rounded=True, special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("arbolPreview")
tree.export_text(model, feature_names=x.columns.tolist())

