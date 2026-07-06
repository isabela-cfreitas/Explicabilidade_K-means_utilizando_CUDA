# ExGreedy-CUDA

Adaptação e paralelização em GPU (CUDA) do algoritmo **ExGreedy**, desenvolvida
como parte de "Explicabilidade para Kmeans com árvores de decis˜ao em GPU" para a disciplina: INF 494 - Tópicos Especiais em Processamento de Alto Desempenho Com Gpus.

Este trabalho é derivado do algoritmo originalmente proposto na tese de doutorado
de Lucas Saadi Murtinho (PUC-Rio, 2025), disponível em: [\[ShallowTree\]](https://github.com/lmurtinho/ShallowTree/tree/main)

## Citação do trabalho original

```bibtex
@phdthesis{murtinho2025clustering,
  author  = {Murtinho, Lucas Saadi},
  title   = {Clustering under constraints: explainability via decision trees and separability with minimum size},
  school  = {Pontifícia Universidade Católica do Rio de Janeiro (PUC-Rio)},
  year    = {2025},
  address = {Rio de Janeiro, Brasil},
  month   = {August}
}
```

## Sobre este repositório

O código original do ExGreedy é dividido em dois módulos: a lógica de avaliação
de cortes (`best_cut`, originalmente em C) e a lógica de construção da árvore
(`shallow_tree`, originalmente em Python). Para viabilizar uma comparação de
desempenho justa entre a versão sequencial e a versão paralela, sem que a
diferença de linguagem (Python vs. CUDA/C) influenciasse os resultados, o
módulo `shallow_tree` foi traduzido para C e integrado ao módulo `best_cut`
original, preservando fielmente a lógica de decisão do algoritmo. Essa versão
(`tree_full.c`) é usada como baseline sequencial neste repositório.

A partir dessa baseline, foi desenvolvida uma versão paralela em CUDA
(`shallow_tree3.cu`).

## O que foi modificado em relação ao algoritmo original

- Tradução do módulo `shallow_tree` (originalmente em Python) para C, para
  compor a baseline sequencial (`tree_full.c`).
- Paralelização em GPU do cálculo da matriz de distâncias entre pontos e
  centroides (kernel CUDA).
- Paralelização em GPU das ordenações necessárias ao algoritmo (ordenação dos
  pontos por dimensão e dos centroides por distância), realizadas uma única
  vez de forma global antes da construção da árvore, em vez de recalculadas a
  cada nó.
- Reestruturação da construção da árvore de busca em profundidade (DFS) para
  busca em largura (BFS), permitindo que cada nó filho reaproveite, por
  filtragem, as ordenações herdadas do nó pai.
