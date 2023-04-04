import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
import sklearn as skl
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# %%
df = pd.read_csv('cometa_csv/delicious.csv')

# %%
m_delicious = df.loc[:, "TAG_.imported":"TAG_youtube"]

# %%
mt_delicious = np.array(m_delicious).T
display(mt_delicious)

# %% [markdown]
# ### Analisando dados (maior/menor)

# %% [markdown]
# #### Verificando quem é o menor valor diferente de zero
# A máscara booleana é criada usando o operador != para comparar cada elemento do ndarray com zero. O resultado é um array booleano com valores True para os elementos diferentes de zero e False para os elementos iguais a zero.

# %%
mask = m_euclidiana!= 0
min_value = np.min(m_euclidiana[mask])
print(min_value)

# %%
mask = m_euclidiana!= 1
max_value = np.max(m_euclidiana[mask])
print(max_value)

# %%
mask = m_jaccard != 0
min_value = np.min(m_jaccard[mask])
print(min_value)

# %%
mask = m_jaccard != 1
max_value = np.max(m_jaccard[mask])
print(max_value)

# %%
median = (max_value-min_value)/2
median

# %% [markdown]
# ## Kmeans usando distância Euclidiana

# %%
m_euclidiana = skl.metrics.pairwise_distances(mt_delicious, metric='euclidean')
m_euclidiana

# %% [markdown]
# #### Recorrendo ao Kmeans ++ para otimizar a escolha do centroide

# %%
# metodo cotovelo 
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 50)
    kmeans.fit(m_euclidiana)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11), wcss)
plt.title('Método do cotovelo')
plt.xlabel('Número de clusters')
plt.ylabel('Within-Clister-Sum-of-Squares')
plt.show()

# %% [markdown]
# #### Silhueta 
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html

# %%
# Silhueta
from sklearn.metrics import silhouette_samples, silhouette_score

for i in range(2, 11):
    clusterer = KMeans(n_clusters=i)
    preds = clusterer.fit_predict(m_euclidiana)
    score = silhouette_score(m_euclidiana, preds)
    print('Silhueta para ' + str(i) + ' clusters : ' + str(score))

# %% [markdown]
# #### Supondo k=2

# %%
kmeans = KMeans(n_clusters = 2, init='k-means++', max_iter=300, n_init=30)
clusters = kmeans.fit_predict(m_euclidiana)
clusters  # consigo ver quem pertence a quais conjuntos

# %%
x = m_euclidiana

plt.figure(figsize=(8, 8))
plt.scatter(
    x[clusters == 0,0], x[clusters == 0,1],
    s=40,c='cyan',
    #edgecolor='black',
    label='Cluster A'
)

plt.scatter(
    x[clusters == 1,0], x[clusters == 1,1],
    s=40,c='blue',
    #edgecolor='black',
    label='Cluster B'
)

# centroide
plt.scatter(
    kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], 
    s=90,c='red',
    #edgecolor='black',
    label='centroides'
)

plt.title('K-means utilizando distância euclidiana e k=3 na base Delicious.')
plt.legend()
plt.show()

# %% [markdown]
# #### Supondo k=3

# %%
kmeans = KMeans(n_clusters = 3, init='k-means++', max_iter=300, n_init=30)
clusters = kmeans.fit_predict(m_euclidiana)
clusters  # consigo ver quem pertence a quais conjuntos

# %% [markdown]
# #### Duvidas
# - Devo normalizar os dados? A justificativo seria para que a variação em um atributo não ofusque as variações em outros atributos.
# 

# %%
x = m_euclidiana

plt.figure(figsize=(8, 8))
plt.scatter(
    x[clusters == 0,0], x[clusters == 0,1],
    s=40,c='cyan',
    #edgecolor='black',
    label='Cluster A'
)

plt.scatter(
    x[clusters == 1,0], x[clusters == 1,1],
    s=40,c='blue',
    #edgecolor='black',
    label='Cluster B'
)
plt.scatter(
    x[clusters == 2,0], x[clusters == 2,1],
    s=40,c='greenyellow',
    #edgecolor='black',
    label='CLuster C'
)

# centroide
plt.scatter(
    kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], 
    s=90,c='red',
    #edgecolor='black',
    label='centroides'
)

plt.title('K-means utilizando distância euclidiana e k=3 na base Delicious.')
plt.legend()
plt.show()

# %% [markdown]
# ## Implementando para outras distâncias 
# 

# %% [markdown]
# ### Jaccard

# %%
m_jaccard = skl.metrics.pairwise_distances(mt_delicious, metric='jaccard')
m_jaccard

# %%
# metodo cotovelo 
wcss_jaccard = []
for i in range(1,13):
    kmeans_jaccard = KMeans(n_clusters = i, init = 'k-means++', max_iter = 400, n_init = 30)
    kmeans_jaccard.fit(m_jaccard)
    wcss_jaccard.append(kmeans_jaccard.inertia_)

plt.plot(range(1,13), wcss_jaccard)
plt.title('Método do cotovelo')
plt.xlabel('Número de clusters')
plt.ylabel('Within-Clister-Sum-of-Squares')
plt.show()

# %%
for i in range(2, 11):
    clusterer = KMeans(n_clusters=i)
    preds = clusterer.fit_predict(m_jaccard)
    score = silhouette_score(m_jaccard, preds)
    print('Silhueta para ' + str(i) + ' clusters : ' + str(score))

# %% [markdown]
# ### Dúvida:
# - Não tende a zero tal como no caso da distância euclidiana -> jaccard é aplicável ainda sim?

# %% [markdown]
# ### Tanimoto

# %% [markdown]
# #### Cotovelo para achar k

# %%
m_tanimoto = skl.metrics.pairwise_distances(mt_delicious, metric='rogerstanimoto')
m_tanimoto

# %%
# metodo cotovelo 
wcss_tanimoto = []
for i in range(1,13):
    kmeans_tanimoto = KMeans(n_clusters = i, init = 'k-means++', max_iter = 400, n_init = 30)
    kmeans_tanimoto.fit(m_tanimoto)
    wcss_tanimoto.append(kmeans_tanimoto.inertia_)

plt.plot(range(1,13), wcss_tanimoto)
plt.title('Método do cotovelo')
plt.xlabel('Número de clusters')
plt.ylabel('Within-Clister-Sum-of-Squares')
plt.show()

# %%
for i in range(2, 11):
    clusterer = KMeans(n_clusters=i)
    preds = clusterer.fit_predict(m_tanimoto)
    score = silhouette_score(m_tanimoto, preds)
    print('Silhueta para ' + str(i) + ' clusters : ' + str(score))

# %% [markdown]
# ### Supondo k=3  (Tanimoto)

# %%
kmeans_tanimoto = KMeans(n_clusters = 3, init='k-means++', max_iter=300, n_init=30)
clusters = kmeans.fit_predict(m_tanimoto)
clusters  # consigo ver quem pertence a quais conjuntos

# %%
x_tanimoto= m_tanimoto

plt.figure(figsize=(8,8))
plt.scatter(
    x_tanimoto[clusters == 0,0], x_tanimoto[clusters == 0,1],
    s=20, c='salmon',
    #edgecolor='black',
    label='Cluster A',
)

plt.scatter(
    x_tanimoto[clusters == 1,0], x_tanimoto[clusters == 1,1], 
    s=40,c='violet',
    #edgecolor='black',
    label='Cluster B',
    
)
plt.scatter(
    x_tanimoto[clusters == 2,0], x_tanimoto[clusters == 2,1],
    s=40,c='aqua',
    #edgecolor='black',
    label='CLuster C',
)

# centroide
plt.scatter(
    kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1],
    c='red',
    edgecolor='black',
    label='centroides',
    s=90
)

plt.title('K-means utilizando distância tanimoto e k=3 na base Delicious.')
plt.legend()
plt.show()
