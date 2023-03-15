# %% [markdown]
# # Base Delicious - KNN

# %% [markdown]
# # 0. Estruturação do dataframe

# %%
import csv
import numpy as np
import pandas as pd
import seaborn as sns 
import sklearn as skl
import networkx as nx
import matplotlib.pyplot as plt
from scipy.io import arff
from sklearn.metrics import jaccard_score
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph

# %%
# criando um dataframe do arquivo lido pelo pandas 
df = pd.read_csv('cometa_csv/delicious.csv')

# selecionando as tags
m_delicious = df.loc[:, "TAG_.imported":"TAG_youtube"]

# uma vez que a distância entre as instâncias é calculada por comparação entre colunas, transponho a m_flags 
mt_delicious = np.array(m_delicious).T

# configurando as casas decimais que irão ser impressas 
np.set_printoptions(precision=3)

# %% [markdown]
# # 1. Definindo funções que serão utilizadas 

# %% [markdown]
# - **draw_graph**: utilizada para gerar o grafo completo a partir da da matriz de distância usando como métrica a distância inserida
#     - Cada elemento na matriz representa a distância (por exemplo Jaccard) entre duas amostras específicas.
# 
# - **draw_knn_graph**: utilizada para gerar o grafo esparsificado por knn
#     - o grafo é gerado tendo como base uma matriz esparsa 
#     - O argumento k especifica o número de vizinhos mais próximos a serem considerados para cada amostra. 
#     - O argumento mode especifica se a distância entre as amostras será armazenada como uma distância ou como uma similaridade. Se o mode for definido como 'distance', então as distâncias serão armazenadas na matriz. 
#     - O argumento metric especifica a métrica de distância a ser usada para calcular a distância entre as amostras. 
#     - O argumento include_self especifica se cada amostra será incluída como um dos seus próprios vizinhos mais próximos.
#     

# %%
# criar o grafo completo
def draw_graph(metric, color, matriz):
    m = skl.metrics.pairwise_distances(matriz, metric=metric)
    G = nx.from_numpy_array(m)
    list(nx.selfloop_edges(G))
    G.remove_edges_from(nx.selfloop_edges(G))

    plt.figure(num=None, figsize=(10, 10), dpi=80)
    nx.draw(G, with_labels=False, node_color=color, node_size = 100) 
    plt.title(f'Base "Delicious" - Grafo completo usando {metric}', fontsize=17)
    plt.show()

# %%
def draw_knn_graph(k, metric, cor, matriz):
    sparse_matrix_knn = kneighbors_graph(matriz, k, mode='distance', metric=metric, include_self=False)
    G_sparse_knn = nx.from_numpy_array(sparse_matrix_knn)
    G_sparse_knn.remove_edges_from(nx.selfloop_edges(G_sparse_knn))

    plt.figure(num=None, figsize=(10, 10), dpi=80)
    nx.draw(G_sparse_knn, with_labels=False, node_color=cor, node_size = 100, edgecolors='gray') 
    plt.title(f'Base "Delicious" - Grafo esparsificado por KNN usando {metric} e k={k}', fontsize=17)
    plt.show()

# %% [markdown]
# # 2. Aplicando Knn para as distâncias definidas 

# %% [markdown]
# ## 2.1) Jaccard

# %% [markdown]
# #### Grafo completo

# %%
draw_graph('jaccard', 'Red', mt_delicious)

# %% [markdown]
# #### Grafo esparsificado (knn)

# %% [markdown]
# ##### 1-10

# %%
draw_knn_graph(1,'jaccard','Red', mt_delicious)
draw_knn_graph(2,'jaccard','Red', mt_delicious)
draw_knn_graph(3,'jaccard','Red', mt_delicious)
draw_knn_graph(4,'jaccard','Red', mt_delicious)
draw_knn_graph(5,'jaccard','Red', mt_delicious)

# %%
draw_knn_graph(6,'jaccard', 'Red',mt_delicious)
draw_knn_graph(7,'jaccard','Red', mt_delicious)
draw_knn_graph(8,'jaccard','Red', mt_delicious)
draw_knn_graph(9,'jaccard','Red', mt_delicious)
draw_knn_graph(10,'jaccard','Red', mt_delicious)

# %% [markdown]
# ##### 10-20

# %%
draw_knn_graph(11,'jaccard','Red', mt_delicious)
draw_knn_graph(12,'jaccard','Red', mt_delicious)
draw_knn_graph(13,'jaccard','Red', mt_delicious)
draw_knn_graph(14,'jaccard','Red', mt_delicious)
draw_knn_graph(15,'jaccard','Red', mt_delicious)

# %%
draw_knn_graph(16,'jaccard','Red', mt_delicious)
draw_knn_graph(17,'jaccard','Red', mt_delicious)
draw_knn_graph(18,'jaccard','Red', mt_delicious)
draw_knn_graph(19,'jaccard','Red', mt_delicious)
draw_knn_graph(20,'jaccard','Red', mt_delicious)

# %%
draw_knn_graph(25,'jaccard','Red', mt_delicious)

# %% [markdown]
# ## 2.2) Rogers tanimoto

# %% [markdown]
# #### Grafo completo

# %%
draw_graph('rogerstanimoto', 'Blue', mt_delicious)

# %% [markdown]
# #### Grafos esparsificados

# %% [markdown]
# ##### 1-10

# %%
draw_knn_graph(1,'rogerstanimoto', 'Blue', mt_delicious)
draw_knn_graph(2,'rogerstanimoto','Blue',  mt_delicious)
draw_knn_graph(3,'rogerstanimoto','Blue',  mt_delicious)
draw_knn_graph(4,'rogerstanimoto', 'Blue', mt_delicious)
draw_knn_graph(5,'rogerstanimoto','Blue',  mt_delicious)

# %%
draw_knn_graph(6,'rogerstanimoto','Blue',  mt_delicious)
draw_knn_graph(7,'rogerstanimoto','Blue',  mt_delicious)
draw_knn_graph(8,'rogerstanimoto','Blue',  mt_delicious)
draw_knn_graph(9,'rogerstanimoto', 'Blue', mt_delicious)
draw_knn_graph(10,'rogerstanimoto', 'Blue', mt_delicious)

# %% [markdown]
# ##### 15-25

# %%
draw_knn_graph(15,'rogerstanimoto','Blue',  mt_delicious)
draw_knn_graph(20,'rogerstanimoto','Blue',  mt_delicious)
draw_knn_graph(25,'rogerstanimoto','Blue', mt_delicious)

# %% [markdown]
# ## 2.3) Cityblock

# %% [markdown]
# #### Grafo completo

# %%
draw_graph('cityblock', 'purple', mt_delicious)

# %% [markdown]
# ### Grafos esparsificados

# %% [markdown]
# ##### 1 - 10

# %%
draw_knn_graph(1,'cityblock','Purple', mt_delicious)
draw_knn_graph(2,'cityblock','Purple', mt_delicious)
draw_knn_graph(3,'cityblock','Purple', mt_delicious)
draw_knn_graph(4,'cityblock','Purple', mt_delicious)


# %%
draw_knn_graph(5,'cityblock','Purple', mt_delicious)
draw_knn_graph(6,'cityblock','Purple', mt_delicious)
draw_knn_graph(7,'cityblock','Purple', mt_delicious)
draw_knn_graph(8,'cityblock','Purple', mt_delicious)
draw_knn_graph(9,'cityblock','Purple', mt_delicious)
draw_knn_graph(10,'cityblock','Purple', mt_delicious)


# %% [markdown]
# ##### 11-25

# %%
draw_knn_graph(11,'cityblock','Purple', mt_delicious)
draw_knn_graph(12,'cityblock','Purple', mt_delicious)
draw_knn_graph(13,'cityblock','Purple', mt_delicious)
draw_knn_graph(14,'cityblock','Purple', mt_delicious)

# %%
draw_knn_graph(15,'cityblock','Purple', mt_delicious)


# %%
draw_knn_graph(16,'cityblock','Purple', mt_delicious)
draw_knn_graph(17,'cityblock','Purple', mt_delicious)
draw_knn_graph(18,'cityblock','Purple', mt_delicious)
draw_knn_graph(19,'cityblock','Purple', mt_delicious)

# %%
draw_knn_graph(20,'cityblock','Purple', mt_delicious)


# %%
draw_knn_graph(25,'cityblock','Purple', mt_delicious)


# %% [markdown]
# ## 2.4) Chebyshev

# %% [markdown]
# #### Grafo completo

# %%
draw_graph('chebyshev', 'green', mt_delicious)

# %% [markdown]
# #### Grafos esparsificados

# %%
draw_knn_graph(1,'chebyshev','Green', mt_delicious)


# %%
draw_knn_graph(5,'chebyshev','Green', mt_delicious)


# %% [markdown]
# ## 2.5) Minkowski

# %% [markdown]
# #### Grafo completo 

# %%
draw_graph('minkowski','C0', mt_delicious)

# %% [markdown]
# #### Grafos esparsificados

# %% [markdown]
# ##### 1-10

# %%
draw_knn_graph(1, 'minkowski', 'C0', mt_delicious)
draw_knn_graph(2, 'minkowski', 'C0', mt_delicious)
draw_knn_graph(3, 'minkowski', 'C0', mt_delicious)
draw_knn_graph(4, 'minkowski', 'C0', mt_delicious)
draw_knn_graph(5, 'minkowski', 'C0', mt_delicious)

# %%
draw_knn_graph(6, 'minkowski', 'C0', mt_delicious)
draw_knn_graph(7, 'minkowski', 'C0', mt_delicious)
draw_knn_graph(8, 'minkowski', 'C0', mt_delicious)
draw_knn_graph(9, 'minkowski', 'C0', mt_delicious)
draw_knn_graph(10, 'minkowski', 'C0', mt_delicious)

# %% [markdown]
# ##  2.6) Hamming

# %% [markdown]
# - A distância de Hamming pode ser usada para medir a diferença entre duas sequências binárias de comprimento igual.
# - A matriz m é uma matriz de distâncias que contém a distância de Hamming entre todas as combinações possíveis de linhas da matriz mt_delicious.

# %% [markdown]
# #### Grafo completo

# %%
draw_graph('hamming', 'cyan', mt_delicious)

# %% [markdown]
# #### Grafos esparsificados

# %% [markdown]
# ##### 1-10

# %%
draw_knn_graph(1,'hamming', 'cyan', mt_delicious)
draw_knn_graph(2,'hamming', 'cyan', mt_delicious)
draw_knn_graph(3,'hamming', 'cyan', mt_delicious)
draw_knn_graph(4,'hamming', 'cyan', mt_delicious)
draw_knn_graph(5,'hamming', 'cyan', mt_delicious)

# %%
draw_knn_graph(6,'hamming', 'cyan', mt_delicious)
draw_knn_graph(7,'hamming', 'cyan', mt_delicious)
draw_knn_graph(8,'hamming', 'cyan', mt_delicious)
draw_knn_graph(9,'hamming', 'cyan', mt_delicious)
draw_knn_graph(10,'hamming', 'cyan', mt_delicious)

# %% [markdown]
# ##### 15-20

# %%
draw_knn_graph(15,'hamming', 'cyan', mt_delicious)
draw_knn_graph(20,'hamming', 'cyan', mt_delicious)

# %% [markdown]
# ## 2.7) Cosseno 

# %% [markdown]
# #### Grafo completo

# %%
draw_graph('cosine', 'C2', mt_delicious)

# %% [markdown]
# #### Grafos esparsificados 

# %% [markdown]
# ##### 1 - 10

# %%
draw_knn_graph(1, 'cosine', 'C2', mt_delicious)

# %%
draw_knn_graph(2, 'cosine', 'C2', mt_delicious)
draw_knn_graph(3, 'cosine', 'C2', mt_delicious)
draw_knn_graph(4, 'cosine', 'C2', mt_delicious)
draw_knn_graph(5, 'cosine', 'C2', mt_delicious)

# %%
draw_knn_graph(6, 'cosine', 'C2', mt_delicious)
draw_knn_graph(7, 'cosine', 'C2', mt_delicious)
draw_knn_graph(8, 'cosine', 'C2', mt_delicious)
draw_knn_graph(9, 'cosine', 'C2', mt_delicious)
draw_knn_graph(10, 'cosine', 'C2', mt_delicious)


