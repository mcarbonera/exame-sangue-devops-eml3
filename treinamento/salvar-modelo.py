## Importar pacotes
import pandas as pd
# Para normalizar
from sklearn.preprocessing import MinMaxScaler
#importar KMeans para fazer o agrupamento
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
layout = {
  'width': 1410,
  'height': 525
}

dfHVC = pd.read_csv('./treinamento/notebook-atividade/hcvdata.csv')

stats = dfHVC.describe()
print(stats)

## Substituir os valores nulos pela mediana.
#dfHVC[pd.isna(dfHVC['ALB'])]
dfHVC['ALB'] = dfHVC['ALB'].apply(lambda x: stats['ALB']['50%'] if pd.isna(x) else x)
#dfHVC[pd.isna(dfHVC['ALP'])]
dfHVC['ALP'] = dfHVC['ALP'].apply(lambda x: stats['ALP']['50%'] if pd.isna(x) else x)
#dfHVC[pd.isna(dfHVC['ALT'])]
dfHVC['ALT'] = dfHVC['ALT'].apply(lambda x: stats['ALT']['50%'] if pd.isna(x) else x)
#dfHVC[pd.isna(dfHVC['CHOL'])]
dfHVC['CHOL'] = dfHVC['CHOL'].apply(lambda x: stats['CHOL']['50%'] if pd.isna(x) else x)
#dfHVC[pd.isna(dfHVC['PROT'])]
dfHVC['PROT'] = dfHVC['PROT'].apply(lambda x: stats['PROT']['50%'] if pd.isna(x) else x)

## Definir a lista de atributos categóricos e atributos contínuos
categorical_features = ['Sex']
continuous_features = ['Age', 'ALB', 'ALP', 'ALT', 'AST', 'BIL', 'CHE', 'CHOL', 'CREA', 'GGT', 'PROT']

## Transformar atributos categóricos em binários com a função get_dummies()
for col in categorical_features:
  dummies = pd.get_dummies(dfHVC[col], prefix=col)
  dfHVC = pd.concat([dfHVC, dummies], axis=1)

## Normalizar os atributos contínuos com MinMaxScaler()
df_transformedHVC = pd.DataFrame(dfHVC)
del df_transformedHVC['Unnamed: 0']
del df_transformedHVC['Category']
del df_transformedHVC['Sex']
# remover sexo
del df_transformedHVC['Sex_f']
del df_transformedHVC['Sex_m']

mms = MinMaxScaler()
mms.fit(df_transformedHVC)

columns = df_transformedHVC.columns
print(columns.to_list())
mmsTransformed = mms.transform(df_transformedHVC)
df_transformedHVC = pd.DataFrame(mmsTransformed)
df_transformedHVC.columns = columns

## Aplicando o método do cotovelo
K = range(1,31)

Sum_of_squared_distances = []
for k in K:
  km = KMeans(n_clusters=k,n_init=20)
  km = km.fit(df_transformedHVC)
  Sum_of_squared_distances.append(km.inertia_)

## Determinar a melhor quantidade de clusters
model = KMeans(n_init=20, init = 'random')
visualizer = KElbowVisualizer(model, k=K, size=(layout['width'], layout['height']))
visualizer.fit(df_transformedHVC.values)
#visualizer.show()

kBest = visualizer.elbow_value_
print(kBest)

# Melhor modelo:
kmeans = KMeans(n_clusters = kBest, n_init = 20, init = 'random')
kmeans.fit(df_transformedHVC.values)
kmeans.cluster_centers_

import pickle;

mediansToStore = stats.iloc[5]
del mediansToStore['Unnamed: 0']
modelToDump = {
  'preprocessor': mms,
  'model': kmeans,
  'medians': mediansToStore
}
pickle.dump(modelToDump, open("model.pkl", "wb"))