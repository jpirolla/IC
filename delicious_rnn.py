# %% [markdown]
# # Base Delicious - RNN
 
# %% [markdown]
# # 0. Estruturação do dataframe

# %%
import pandas as pd
import seaborn as sns
import numpy as np
import networkx as nx
import sklearn as skl
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances

# %
# criando um dataframe do arquivo lido pelo pandass
df = pd.read_csv('cometa_csv/delicious.csv')

# selecionando as tags
m_delicious = df.loc[:, "TAG_.imported":"TAG_youtube"]

# uma vez que a distância entre as instâncias é calculada por comparação entre colunas, transponho a m_flags 
mt_delicious = np.array(m_delicious).T

# configurando as casas decimais que irão ser impressas 
np.set_printoptions(precision=3)

# %%
df.head()

# %%
colunas = df.columns[df.columns.get_loc("TAG_.imported"):df.columns.get_loc("TAG_youtube") + 1]
print(colunas)


# %% [markdown]
# # 1. Definindo funções que serão utilizadas 

# %% [markdown]
# - **draw_graph**: utilizada para gerar o grafo completo a partir da da matriz de distância usando como métrica a distância inserida
#     - Cada elemento na matriz representa a distância (por exemplo Jaccard) entre duas amostras específicas.
#     
# 
# - **draw_rnc_graph**: utilizada para gerar o grafo esparsificado por rnc
#     - o grafo é gerado tendo como base uma matriz esparsa 
#     - O grafo é construído de tal forma que cada ponto da matriz matriz é conectado com outros pontos que estão dentro de uma distância específica (definida pelo argumento radius).
#     - O argumento r especifica o raio a ser utilizado para incluir os vizinhos mais próximos para cada amostra. 
#     - O argumento mode especifica se a distância entre as amostras será armazenada como uma distância ou como uma similaridade. Se o mode for definido como 'distance', então as distâncias serão armazenadas na matriz. 
#     - O argumento metric especifica a métrica de distância a ser usada para calcular a distância entre as amostras. 
#     - O argumento include_self especifica se cada amostra será incluída como um dos seus próprios vizinhos mais próximos.
#     
#     **OBS**: os valores de máximo e minimo são uma forma de visualizar o intervalo que devo estipular o histograma para ver a distribuição
# 
# 
# - **draw_histogram**: utilizado para visualizar a distribuição dos valores da matriz m 
#     - utilzado para ter uma ideia de qual intervalo faz mais sentido estipular um raio. 

# %%
def draw_graph(metric, color, matriz):
    m = skl.metrics.pairwise_distances(matriz, metric=metric)


    G = nx.from_numpy_array(m)
    list(nx.selfloop_edges(G))
    G.remove_edges_from(nx.selfloop_edges(G))

    minimum_value = np.min(m)
    maximum_value = np.max(m)

    print("Minimum value in matrix:", minimum_value)
    print("Maximum value in matrix:", maximum_value)

    plt.figure(num=None, figsize=(10, 10), dpi=80)
    nx.draw(G, with_labels=False, node_color=color, node_size = 100) 
    plt.title(f'Base "Delicious" - Grafo completo usando {metric}', fontsize=17)
    plt.show()

# %%
def draw_rnc_graph(radius, metric, cor, matriz):
    sparse_matrix_rnc = radius_neighbors_graph(matriz, radius, mode='distance', metric=metric, include_self=False)

    G_sparse_rnc = nx.from_numpy_array(sparse_matrix_rnc)
    G_sparse_rnc.remove_edges_from(nx.selfloop_edges(G_sparse_rnc))

    plt.figure(num=None, figsize=(10, 10), dpi=80)
    nx.draw(G_sparse_rnc, with_labels=False, node_color=cor, node_size = 100, edgecolors='gray') 
    plt.title(f'Base "Delicious" - Grafo esparsificado por Radius Neighbors Classifier usando {metric} e raio={radius}', fontsize=17)
    plt.show()


# %%
def draw_knn_graph_grid(k_list, metric, matriz, linhas, colunas, cor):
    
    # Gerar matrizes esparsas
    sparse_matrix_list = []  # lista que vai armazenar as matrizes 
    for k in k_list:
        sparse_matrix_knn = kneighbors_graph(matriz, k, mode='distance', metric=metric, include_self=False)
        sparse_matrix_list.append(sparse_matrix_knn)
    
    # Gerar lista de grafos
    graph_list = []    # lista que vai armazenar os grafos
    for sparse_matrix in sparse_matrix_list:
        G_sparse_knn = nx.from_numpy_array(sparse_matrix)
        G_sparse_knn.remove_edges_from(nx.selfloop_edges(G_sparse_knn))
        graph_list.append(G_sparse_knn)
    
    # Desenhar subplots dos grafos
    n_rows = linhas
    n_cols = colunas
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
    for i, ax in enumerate(axs.flatten()):
            if i >= len(k_list):
                break
            nx.draw(graph_list[i], ax=ax, with_labels=False, node_color=cor, node_size=20)
            ax.set_title(f'KNN usando {metric} e k={k_list[i]}', fontsize=12)
    
    # Exibir a figura
    plt.show()


# %%
def draw_histogram(matriz, metric, xinf, xsup):
    m = skl.metrics.pairwise_distances(matriz, metric=metric)
    plt.hist(m, bins=20)
    plt.xlabel("Paired Distance")
    plt.ylabel("Frequency")
    plt.title("Histogram of Paired Distances")
    plt.xlim(xinf, xsup)
    plt.show()

# %% [markdown]
# #### Sobre o MST graph 
# - Calcula-se a distância euclidiana entre todas as amostras, resultando na matriz D. 
# - Em seguida, é calculada a Árvore Geradora Mínima (Minimum Spanning Tree - MST) da matriz D com a função minimum_spanning_tree
# - Retorna uma matriz direcionada, adj_directed, representando o grafo da MST.

# %% [markdown]
# #### Transformando o grafo dir em não direcionado
# Para transformar o grafo direcionado em um grafo não-direcionado, o código **adiciona a transposta de adj_directed a si mesma e define todas as arestas com valor maior que zero como 1**
# . Por fim, a diagonal da matriz resultante é preenchida com 0 para evitar auto-loops no grafo.

# %%
def mst_graph(X):
    D = euclidean_distances(X, X)
    adj_directed = minimum_spanning_tree(D).toarray()
    adj = adj_directed + adj_directed.T
    adj[adj > 0] = 1
    np.fill_diagonal(adj,0)

    return csr_matrix(adj)

# %%
def draw_conex_graph(raio, metric,  color, matriz):
    W = radius_neighbors_graph(matriz, raio, mode="distance", metric=metric, include_self=False)
    W = W + mst_graph(matriz)
    
    G = nx.from_numpy_array(W)
    plt.figure(num=None, figsize=(10, 10), dpi=80)
    nx.draw(G, with_labels=False, node_color=color, node_size = 100, edgecolors='gray') 
    plt.title(f'Base "Delicious" - Grafo conexo usando RNN com raio = {raio} e métrica {metric}', fontsize=17)
    plt.show()


# %%
# W = radius_neighbors_graph(mt_delicious, 0.8, mode="distance", metric="euclidean", include_self=False)
# W = W + mst_graph(mt_delicious)

# %%
# G = nx.from_numpy_array(W)
# plt.figure(num=None, figsize=(10, 10), dpi=80)
# nx.draw(G, with_labels=False, node_color='lightpink', node_size = 100, edgecolors='gray') 
# plt.title(f'Base "Delicious" - Grafo conexo usando RNN e metrica {euclidiana}', fontsize=17)
# plt.show()


# %% [markdown]
# **Gerando um grafo conexo** com MST
# 
# É gerado um grafo esparsificado pelo raio vizinho com a função radius_neighbors_graph, utilizando um raio de 0,8, modo de distância, métrica euclidiana e sem incluir a amostra consigo mesma. Este grafo é adicionado ao grafo da MST gerado anteriormente para obter o resultado final. O resultado final é retornado como uma matriz de coo (compressed sparse row).

# %% [markdown]
# # 2. Aplicando Knn para as distâncias definidas 

# %% [markdown]
# ## 2.1) Euclidiana

# %% [markdown]
# #### Grafo completo

# %% [markdown]
# O código acima gera um grafo a partir da matriz de recursos mt_delicious de forma que suas arestas conectem todos os vértices que estejam a uma distância de até 0,8, medida pela distância Euclidiana, do ponto de partida. Em seguida, o grafo gerado é adicionado ao resultado da geração do grafo mínimo de árvore gerador.
# 
# A função radius_neighbors_graph da biblioteca scikit-learn é usada para gerar o primeiro grafo. Ela aceita como entrada uma matriz de recursos, um raio de distância, um modo de distância (neste caso, distância Euclidiana), uma métrica (também distância Euclidiana) e uma opção para incluir ou não o próprio vértice como um vizinho.
# 
# A função mst_graph é usada para gerar o segundo grafo, que é o grafo mínimo de árvore gerador. Ela recebe como entrada uma matriz de recursos e retorna a matriz de adjacência do grafo gerado.
# 
# Ao final, o grafo gerado pela função radius_neighbors_graph é adicionado ao grafo gerado pela função mst_graph, resultando em um grafo completo.

# %%
draw_graph('euclidean', 'lightpink', mt_delicious)

# %%
draw_histogram(mt_delicious, 'euclidean', 5,60)

# %%
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import RadiusNeighborsClassifier

# Definir a grade de parâmetros a ser testada
param_grid = {'radius': [10, 15, 20], 'metric': ['euclidean', 'manhattan']}

# Criar o estimador
estimator = RadiusNeighborsClassifier()

# Criar o objeto GridSearchCV
grid_search = GridSearchCV(estimator, param_grid, cv=2)

target =  colunas

# Ajustar o modelo aos dados
grid_search.fit(m_delicious, target)

# Exibir os melhores parâmetros e a pontuação
print('Melhores parâmetros:', grid_search.best_params_)
print('Melhor pontuação:', grid_search.best_score_)


# %% [markdown]
# ### Grafos esparsificados

# %%
draw_rnc_graph(0.7,'euclidean', 'lightpink', mt_delicious)


# %%
draw_conex_graph(0.7, "euclidean", "lightpink", mt_delicious)

# %%
draw_rnc_graph(7,'euclidean', 'lightpink', mt_delicious)

# %%
draw_conex_graph(mt_delicious, 0.8, 'euclidean', 'lightpink')

# %%
draw_rnc_graph(10,'euclidean', 'lightpink', mt_delicious)

# %%
draw_conex_graph(mt_delicious, 10, 'euclidean', 'lightpink')

# %%
draw_rnc_graph(15,'euclidean', 'lightpink', mt_delicious)

# %%
draw_conex_graph(mt_delicious, 15, 'euclidean', 'lightpink')

# %%
draw_rnc_graph(25,'euclidean', 'lightpink', mt_delicious)

# %% [markdown]
# ## 2.2) Jaccard

# %% [markdown]
# #### Grafo completo e histograma

# %%
draw_graph('jaccard', 'Red', mt_delicious)

# %%
draw_histogram(mt_delicious, 'jaccard', 0.9, 1.0)

# %% [markdown]
# ### Grafos esparsificados

# %% [markdown]
# ##### 0.1 - 0.8

# %%
draw_rnc_graph(0.1, 'jaccard', 'red', mt_delicious)
draw_rnc_graph(0.2, 'jaccard', 'red', mt_delicious)
draw_rnc_graph(0.3, 'jaccard', 'red', mt_delicious)
draw_rnc_graph(0.4, 'jaccard', 'red', mt_delicious)
draw_rnc_graph(0.5, 'jaccard', 'red', mt_delicious)
draw_rnc_graph(0.6, 'jaccard', 'red', mt_delicious)
draw_rnc_graph(0.7, 'jaccard', 'red', mt_delicious)

# %% [markdown]
# ##### 0.8 - 1.0

# %%
draw_rnc_graph(0.75, 'jaccard', 'red', mt_delicious)
draw_conex_graph(mt_delicious, 0.75, 'jaccard', 'red')


# %%
draw_rnc_graph(0.8, 'jaccard', 'red', mt_delicious)
draw_conex_graph(mt_delicious, 0.8, 'jaccard', 'red')

# %%
draw_rnc_graph(0.9, 'jaccard', 'red', mt_delicious)
draw_conex_graph(mt_delicious, 0.9 , 'jaccard',  'red')

# %%
draw_rnc_graph(0.95, 'jaccard', 'red', mt_delicious)
draw_conex_graph(mt_delicious, 0.95, 'jaccard', 'red')

# %% [markdown]
# ## 2.3) Rogers Tanimoto 

# %% [markdown]
# #### Grafo completo e histograma

# %%
draw_graph('rogerstanimoto', 'blue', mt_delicious)

# %%
    draw_histogram(mt_delicious, 'rogerstanimoto', 0.1, 0.9)

# %%
draw_histogram(mt_delicious, 'rogerstanimoto', 0.1, 0.25)

# %% [markdown]
# #### Grafos esparsificados

# %% [markdown]
# ##### 0.11 - 0.15

# %%
draw_rnc_graph(0.11,'rogerstanimoto', 'Blue', mt_delicious)
draw_conex_graph( 0.11, 'jaccard',  'Blue', mt_delicious)

# %%
draw_rnc_graph(0.12,'rogerstanimoto', 'Blue', mt_delicious)
draw_conex_graph( 0.12, 'jaccard',  'Blue', mt_delicious)

# %%
draw_rnc_graph(0.13,'rogerstanimoto', 'Blue', mt_delicious)
draw_conex_graph( 0.13, 'jaccard',  'Blue', mt_delicious)

# %%
draw_rnc_graph(0.14,'rogerstanimoto', 'Blue', mt_delicious)
draw_conex_graph( 0.14, 'jaccard',  'Blue', mt_delicious)

# %%
draw_rnc_graph(0.15,'rogerstanimoto', 'Blue', mt_delicious)
draw_conex_graph( 0.15, 'jaccard',  'Blue', mt_delicious)

# %% [markdown]
# ##### 0.18 - 0.24

# %%
draw_rnc_graph(0.18,'rogerstanimoto', 'Blue', mt_delicious)
draw_conex_graph( 0.18, 'jaccard',  'Blue', mt_delicious)


# %%
draw_rnc_graph(0.21,'rogerstanimoto', 'Blue', mt_delicious)
draw_conex_graph( 0.21, 'jaccard',  'Blue', mt_delicious)


# %%
draw_rnc_graph(0.24,'rogerstanimoto', 'Blue', mt_delicious)
draw_conex_graph( 0.24, 'jaccard',  'Blue', mt_delicious)

# %% [markdown]
# #### Grafo completo e histograma

# %% [markdown]
# ## 2.5) Minkowski - verificar 

# %% [markdown]
# #### Grafo completo e histograma

# %%
draw_graph('minkowski', 'C0', mt_delicious)

# %%
draw_histogram(mt_delicious, 'minkowski', 0, 89)

# %%
draw_histogram(mt_delicious, 'minkowski', 0, 89)

# %% [markdown]
# #### Grafos esparsificados

# %%
draw_rnc_graph(8,'minkowski', 'C0', mt_delicious)

# %%
draw_conex_graph(8,'minkowski', 'C0', mt_delicious)

# %%
draw_rnc_graph(10,'minkowski', 'C0', mt_delicious)

# %%
draw_conex_graph(10,'minkowski', 'C0', mt_delicious)

# %%
draw_rnc_graph(15,'minkowski', 'C0', mt_delicious)

# %%
draw_conex_graph(15,'minkowski', 'C0', mt_delicious)

# %%
draw_rnc_graph(20,'minkowski', 'C0', mt_delicious)

# %%
draw_conex_graph(20,'minkowski', 'C0', mt_delicious)

# %%
draw_rnc_graph(30,'minkowski', 'C0', mt_delicious)

# %%
draw_conex_graph(30,'minkowski', 'C0', mt_delicious)

# %% [markdown]
# ## 2.6) Hamming

# %% [markdown]
# #### Grafo completo e histograma

# %%
draw_graph('hamming', 'cyan', mt_delicious)

# %%
draw_histogram(mt_delicious,'hamming', 0,0.3)

# %% [markdown]
# #### Grafos esparsificados

# %%
draw_rnc_graph(0.02, 'hamming', 'cyan', mt_delicious)

# %%
draw_conex_graph( 0.02, 'hamming',  'cyan' , mt_delicious)

# %%
draw_rnc_graph(0.03, 'hamming', 'cyan', mt_delicious)
draw_conex_graph( 0.03, 'hamming',  'cyan' , mt_delicious)

# %%
draw_rnc_graph(0.04, 'hamming', 'cyan', mt_delicious)
draw_conex_graph( 0.04, 'hamming',  'cyan' , mt_delicious)


# %%
draw_rnc_graph(0.05, 'hamming', 'cyan', mt_delicious)
draw_conex_graph( 0.05, 'hamming',  'cyan' , mt_delicious)


# %%
draw_rnc_graph(0.06, 'hamming', 'cyan', mt_delicious)
draw_conex_graph( 0.06, 'hamming',  'cyan' , mt_delicious)


# %%
draw_rnc_graph(0.10, 'hamming', 'cyan', mt_delicious)
draw_conex_graph( 0.10, 'hamming',  'cyan' , mt_delicious)


# %%
draw_rnc_graph(0.13, 'hamming', 'cyan', mt_delicious)
draw_conex_graph( 0.13, 'hamming',  'cyan' , mt_delicious)


# %% [markdown]
# ## 2.7) Cosseno

# %% [markdown]
# #### Grafo completo e histograma

# %%
draw_graph('cosine', 'C2', mt_delicious)

# %%
draw_histogram(mt_delicious, 'cosine', 0.7,1)

# %% [markdown]
# #### Grafos esparsificados

# %%
draw_rnc_graph(0.75,'cosine', 'C2', mt_delicious)

# %%
draw_conex_graph(0.75,'cosine', 'C2', mt_delicious)

# %%
draw_rnc_graph(0.83,'cosine', 'C2', mt_delicious)

# %%
draw_conex_graph(0.83,'cosine', 'C2', mt_delicious)

# %%
draw_rnc_graph(0.87,'cosine', 'C2', mt_delicious)

# %%
draw_conex_graph(0.87,'cosine', 'C2', mt_delicious)

# %%
draw_rnc_graph(0.92,'cosine', 'C2', mt_delicious)

# %%
draw_conex_graph(0.92,'cosine', 'C2', mt_delicious)

# %%
draw_rnc_graph(0.95,'cosine', 'C2', mt_delicious)

# %%
draw_conex_graph(0.95,'cosine', 'C2', mt_delicious)

# %% [markdown]
# # Grid search 

# %%
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import RadiusNeighborsClassifier

# Definir a grade de parâmetros a ser testada
param_grid = {'radius': [0.1, 0.5, 1.0], 'metric': ['euclidean', 'manhattan']}

# Criar o estimador
estimator = RadiusNeighborsClassifier()

# Criar o objeto GridSearchCV
grid_search = GridSearchCV(estimator, param_grid, cv=5)

# Ajustar o modelo aos dados
grid_search.fit(matriz, target)

# Exibir os melhores parâmetros e a pontuação
print('Melhores parâmetros:', grid_search.best_params_)
print('Melhor pontuação:', grid_search.best_score_)



