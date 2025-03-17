# Uso de redes complexas para explorar a correlação entre rótulos no contexto de aprendizado supervisionado
Notebooks e scripts utilizados ao longo do projeto de iniciação científica. A seguir estão explicitados alguns tópicos do prójeto de modo a familiarizar-se com os procedimentos que foram realizados. 

## Resumo 
De maneira resumida e simplificada, objetivo do presente trabalho foi realizar a modelagem do espaço de rótulos usando redes e analisar a correlação entre eles por meio de medidas de centralidade, estrutura de comunidades e os resultados estatísticas obtidos a partir das redes modeladas. Além disto, como iniciação científica, o projeto visa a introdução do discente ao meio da pesquisa científica.

## Objetivos
- Extração e manipulação dos dados presentes no repositório: ''\textit{Tips, guidelines and tools for managing multi-label datasets: The mldr.datasets R package and the Cometa data repository}'' (Charte_2018).
- Construção da matriz de similaridade e análise das métricas de similaridade e dissimilaridade evocando índices como Jaccard e Rogers e Tanimoto;
- Detendo os dados do item anterior, construiremos os grafos mediante à diversos algoritmos já consolidados nas bibliotecas disponíveis na ferramenta Python; 
- Esparsificação da rede de rótulos gerada, uma vez que ganha-se em tempo de processamento e clareza na análise quando retiramos arestas de menor relevância.
- A posteriori, serão realizadas múltiplas análises das medidas de centralidade cabíveis na rede, tais como Centralidade de Intermediação, Centralidade de Proximidade, entre outras.
- Assim sendo, passaremos a analisar a estrutura topológica da rede e as possíveis propriedades emergentes, incluindo o processo de detecção de comunidades e sua relação com o todo. Nessa etapa, também buscaremos analisar o quão robusta é a rede gerada.
- Detendo as análises desenvolvidas ao longo do projeto, objetiva-se que os resultados sejam expostos no Congresso de Iniciação Científica da UFSCar (CIC) e nos demais eventos que forem possíveis e condizentes.
- Por fim, um dos principais objetivos do presente trabalho é complementar a formação e fomentar a introdução do discente no meio da pesquisa científica, promovendo desde já o contato com o método científico.

## Conjunto de dados
Inicialmente o presente trabalho tomou como fontes principais os dados presentes em  "The Extreme Classification Repository: Multi-label Datasets & Code" (Bhatia16) e "Tips, guidelines and tools for managing multi-label datasets: The mldr.datasets R package and the Cometa data repository" (Charte_2018), que contempla base de dados de diferentes domínios de aplicação onde cada instância está associada a mais de um rótulo. Porém, até o momento, fizemos uso dos dados conttidos apenas na segunda base. 

### Tratamento de dados

Para que pudessemos trabalhar com os dados, primeiro foi necessário recorrer ao WEKA para converter os arquivos do repositório COMETA de .arff para .csv. Além disso, apesar de tratar-se de dados já rotulados, optamos por realizar verificações  e/ou  aplicar técnicas de pré processamento de dados de modo a selecionar as características relevantes (os rótulos já pré definidos) e garantir qu não há atributos faltantes. 

## Geração dos grafos e análise

Recorrendo à biblioteca Networkx, foram definidas funções de modo a otimizar o processo de construção dos grafos e retirada de self loops, bem como a formulação dos grafos baseados em métodos específicos como KNN e RNN. Diante as variações de parâmetros possíveis, observou-se também a necessidade de aplicarmos algoritmos de árvore geradora mínima (MST) para auxiliar na visualização dos grafos gerados. 
Nesse cenário, o intuito da construção de diversos grafos era coletar dados dos parâmetros que indicavam a formação de comunidades, por exemplo.
