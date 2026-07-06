/*
 * Adaptado do algoritmo ExGreedy (Murtinho et al., 2025)
 * Original: [https://github.com/lmurtinho/ShallowTree/tree/main]
 * Modificações: paralelização em CUDA, mudança de DFS para BFS,
 * reaproveitamento de ordenações entre nós da árvore.
 */


#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>
#include <float.h>
#include <string.h>
#include <cub/cub.cuh>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>

//essa struct é para facilitar a ordenação por dimensão que o sequencial fazia com qsort com a variável global "array"
//usar variável global aqui era pergoso porque vai rodar em paralelo e podia dar condição de corrida
//ela guarda valor e índice prque ordenamos por valor mas ainda precisamos saber a posição que aquele elemento ocupava no dataset original
typedef struct {
    double val;
    int idx;
} ValIndexer;

int compare_val_indexer(const void *a, const void *b) { //comparador de val_indexer, pq o c não deixa fazer funçao em classe :(
    double va = ((ValIndexer*)a)->val;
    double vb = ((ValIndexer*)b)->val;
    if (va < vb) return -1;
    if (va > vb) return 1;
    return 0;
}


double get_height_cost(int curr_height, int N, int K, double alpha, double beta) {//nao é nosso objetivo usar isso agora, mas é para calcular quantos níveis ainda precisaria depois no nível atual, serve mais se fosse usar a penalidade de altura
    int N_l, N_r, K_l, K_r;
    if (K == 1) {
        return N * curr_height;
    }

    K_l = (int)ceil(K * beta);
    K_l = (K_l > 1) ? K_l : 1;
    K_l = (K_l < K) ? K_l : K - 1;
    K_r = K - K_l;
    if (N == 1) {
        if (K_l > K_r) N_l = 1;
        else N_l = 0;
    } else {
        N_l = (int)ceil(N * alpha);
        N_l = (N_l > 1) ? N_l : 1;
        N_l = (N_l < N) ? N_l : N - 1;
    }

    N_r = N - N_l;

    return get_height_cost(curr_height + 1, N_l, K_l, alpha, beta) +
           get_height_cost(curr_height + 1, N_r, K_r, alpha, beta);
}

double get_cur_height_cost(int N_lAux, int N_rAux, int K_lAux, int K_rAux, int n,
                           float alpha, float beta, bool cut_left, bool cut_right) {//também mais voltado para penalidade de altura. soma custo hipotético das subárvores esquerda e direita geradas pelo corte candidato atual
    double height_cost_left = get_height_cost(1, N_lAux, K_lAux, alpha, beta);
    double height_cost_right = get_height_cost(1, N_rAux, K_rAux, alpha, beta);
    if (cut_left) {
        height_cost_left -= (double)N_lAux;
    }
    if (cut_right) {
        height_cost_right -= (double)N_rAux;
    }
    return (height_cost_left + height_cost_right) / n;
}

//essa agora é a que vai avaliar o melhor corte da dimensão
void best_cut_single_dim(double *data, int *data_order, int *data_count, double *centers,double *distances, int *dist_order, int n, int k,double *ans, double height_factor,bool cut_left, bool cut_right) { 
    //ela usa a dimensão já ordenada, aí vai testando os cortes possíveis da esquerda para a direita (tudo isso já é ideia do artigo original, não nossa) e essa parte continua sequencial, não valia a pena paralelizar pq tem dependência forte do corte candidato anterior
    //oq mudamos é que data_order agora é parâmetro, não alocamos nem ordenamos aqui, ele já vem ordenado do pré processamento feito com gpu e filtrado em cpu para pegar só as amostras desse nodo
    int i, j, c, cur_c, ix, ic;
    int idx_data = 0;
    int idx_centers = 0;
    int N_l = 0, N_r = 0, K_r = 0, K_l = 0;
    int N_lAux, N_rAux, K_lAux, K_rAux;
    double nxt_cut, best_cut, alpha, beta, cur_cost = 0;
    double init_dist_cost = 0, cur_dist_cost = 0, cur_height_cost = 0;
    double best_cost, old_data_cost, max_cut;

    int *left_data_mask = (int *)malloc(sizeof(int) * n);
    int *left_centers_mask = (int *)malloc(sizeof(int) * k);
    int *best_in_left = (int *)malloc(sizeof(int) * n);
    int *best_in_right = (int *)malloc(sizeof(int) * n);
    double *cur_dist_costs = (double *)malloc(sizeof(double) * n);
    int *cur_centers = (int *)malloc(sizeof(int) * n);
    int *centers_order = (int *)malloc(sizeof(int) * k);
    bool valid = false;

    //centers_order continua calculado aqui, não valia a pena pré-ordenar isso com cpu e ficar filtrando pq o k costuma não ser grande o suficiente para valer o overhead da gpu
    ValIndexer *centers_sort = (ValIndexer *)malloc(sizeof(ValIndexer) * k);
    for (i = 0; i < k; i++) {//inicializa centroides no vetor que vai ordenar
        centers_sort[i].val = centers[i];
        centers_sort[i].idx = i;
    }
    qsort(centers_sort, k, sizeof(ValIndexer), compare_val_indexer);//entao continua usando qsort para ordenar os centros
    for (i = 0; i < k; i++) centers_order[i] = centers_sort[i].idx;//salva os indices ordenados em center_order
    free(centers_sort);

    c = centers_order[0];//pega o primeiro centroide daquela dimensao
    nxt_cut = centers[c]; //e a coordenada desse centroide é o primeiro corte
    c = centers_order[k - 1];//pega o último tbm
    max_cut = centers[c];//e estabelece que os cortes só podem ir até aquele último, não dá para ter um corte sem centroide indo para um lado

    for (i = 0; i < n; i++) {
        c = dist_order[i * k];
        cur_centers[i] = c;
        cur_dist_costs[i] = distances[i * k + c] * data_count[i];
        cur_dist_cost += cur_dist_costs[i];
    }

    //vai calcular o custo inicial começando com a linha de corte o mais à esquerda, mandando todos os pontos para a direita
    init_dist_cost = cur_dist_cost;
    c = centers_order[0];

    for (i = 0; i < n; i++) best_in_left[i] = c;
    for (i = 0; i < n; i++) best_in_right[i] = 0;

    ix = data_order[idx_data];
    while ((data[ix] <= nxt_cut) && idx_data < n) {
        idx_data++;
        if (idx_data < n) ix = data_order[idx_data];
    }

    while ((centers[c] <= nxt_cut) && (idx_centers < k)) {
        idx_centers++;
        if (idx_centers < k) {
            c = centers_order[idx_centers];
            if (centers[c] <= nxt_cut) {
                for (i = 0; i < n; i++) {
                    cur_c = best_in_left[i];
                    if (distances[i * k + cur_c] > distances[i * k + c]) {
                        best_in_left[i] = c;
                    }
                }
            }
        }
    }

    if (idx_centers == k) {
        ans[0] = -1; ans[1] = INFINITY;
        goto cleanup;
    }

    for (i = 0; i < n; i++) {
        left_data_mask[i] = data[i] <= nxt_cut;
        if (left_data_mask[i]) N_l++;
        else N_r++;
    }

    for (i = 0; i < k; i++) {
        left_centers_mask[i] = centers[i] <= nxt_cut;
        if (left_centers_mask[i]) K_l++;
        else K_r++;
    }

    for (i = 0; i < n; i++) {
        cur_c = cur_centers[i];
        if (left_data_mask[i]) {
            c = best_in_left[i];
        } else {
            j = best_in_right[i];
            c = dist_order[i * k + j];
            while (left_centers_mask[c]) {
                j++;
                c = dist_order[i * k + j];
            }
            best_in_right[i] = j;
        }
        if (c != cur_c) {
            old_data_cost = cur_dist_costs[i];
            cur_dist_costs[i] = distances[i * k + c] * data_count[i];
            cur_dist_cost += (cur_dist_costs[i] - old_data_cost);
            cur_centers[i] = c;
        }
    }

    N_lAux = (N_l > 1) ? N_l : 1;
    N_lAux = (N_lAux == n) ? n - 1 : N_lAux;
    N_rAux = n - N_lAux;
    K_lAux = (K_l > 1) ? K_l : 1;
    K_lAux = (K_lAux == k) ? k - 1 : K_lAux;
    K_rAux = k - K_lAux;
    alpha = (double)N_lAux / n;
    beta = (double)K_lAux / k;

    cur_height_cost = get_cur_height_cost(N_lAux, N_rAux, K_lAux, K_rAux, n, alpha, beta, cut_left, cut_right);
    cur_cost = cur_dist_cost / init_dist_cost + height_factor * cur_height_cost;

    if ((idx_centers != 0) && (idx_centers != k) && (idx_data >= idx_centers) && ((n - idx_data) >= (k - idx_centers))) {
        best_cut = nxt_cut;
        best_cost = cur_cost;
    } else {
        best_cut = -1;
        best_cost = INFINITY;
    }

    while ((idx_data < n) && (idx_centers < k)) {
        ix = data_order[idx_data];
        ic = centers_order[idx_centers];
        nxt_cut = (data[ix] < centers[ic]) ? data[ix] : centers[ic];
        if (nxt_cut >= max_cut) break;

        while ((idx_data < n) && (data[ix] <= nxt_cut)) {
            old_data_cost = cur_dist_costs[ix];
            left_data_mask[ix] = 1; N_l++; N_r--;
            c = best_in_left[ix];
            cur_centers[ix] = c;
            cur_dist_costs[ix] = distances[ix * k + c] * data_count[ix];
            cur_dist_cost += (cur_dist_costs[ix] - old_data_cost);
            idx_data++;
            if (idx_data < n) ix = data_order[idx_data];
        }

        while ((idx_centers < k) && (centers[ic] <= nxt_cut)) {
            left_centers_mask[ic] = 1; K_l++; K_r--;
            for (i = 0; i < n; i++) {
                old_data_cost = cur_dist_costs[i];
                cur_c = best_in_left[i];
                if (distances[i * k + ic] < distances[i * k + cur_c]) {
                    best_in_left[i] = ic;
                }
                if (left_data_mask[i]) {
                    if (best_in_left[i] == ic) {
                        cur_centers[i] = ic;
                        cur_dist_costs[i] = distances[i * k + ic] * data_count[i];
                        cur_dist_cost += (cur_dist_costs[i] - old_data_cost);
                    }
                } else if (cur_centers[i] == ic) {
                    j = best_in_right[i];
                    c = dist_order[i * k];
                    while (left_centers_mask[c]) {
                        j++;
                        c = dist_order[i * k + j];
                    }
                    best_in_right[i] = j;
                    cur_dist_costs[i] = distances[i * k + c] * data_count[i];
                    cur_dist_cost += (cur_dist_costs[i] - old_data_cost);
                    cur_centers[i] = c;
                }
            }
            idx_centers++;
            if (idx_centers < k) ic = centers_order[idx_centers];
        }

        if ((idx_centers != 0) && (idx_centers != k) && (idx_data >= idx_centers) && ((n - idx_data) >= (k - idx_centers))) {
            valid = true;
            N_lAux = (N_l > 1) ? N_l : 1;
            N_lAux = (N_lAux > n) ? n - 1 : N_lAux;
            N_rAux = n - N_lAux;
            K_lAux = (K_l > 1) ? K_l : 1;
            K_lAux = (K_lAux > k) ? k - 1 : K_lAux;
            K_rAux = k - K_lAux;
            alpha = (double)N_lAux / n;
            beta = (double)K_lAux / k;

            if (K_lAux <= 0 || K_rAux <= 0) {
                cur_height_cost = 100000;
            } else {
                cur_height_cost = get_cur_height_cost(N_lAux, N_rAux, K_lAux, K_rAux, n, alpha, beta, cut_left, cut_right);
            }
            cur_cost = cur_dist_cost / init_dist_cost + height_factor * cur_height_cost;
        } else {
            valid = false;
        }

        if (valid && (cur_cost < best_cost)) {
            best_cut = nxt_cut;
            best_cost = cur_cost;
        }
    }

    ans[0] = best_cut;//threshold do corte
    ans[1] = best_cost;//custo do melhor corte

cleanup:
    free(cur_dist_costs); free(left_data_mask); free(left_centers_mask);
    free(best_in_left); free(best_in_right); free(cur_centers);
    free(centers_order);
}


typedef struct TreeNode {
    int feature;
    double value;
    int cluster;
    bool is_leaf;
    struct TreeNode* left;
    struct TreeNode* right;
} TreeNode; //struct da árvore

typedef struct {
    double dist;
    int index;
} DistHelper; //struct para facilitar armazenamento das distâncias ordenadas. igual no valindexer precisa guardar os índices além do valor que vai ser usado para as operações, nesse caso a distância

int compare_dist_helper(const void* a, const void* b) {
    double diff = ((DistHelper*)a)->dist - ((DistHelper*)b)->dist;
    if (diff < 0) return -1;
    if (diff > 0) return 1;
    return 0;
}//também função para usar o struct e fazer comaparardor de distâncias já que não dá para criar função dentro de classe/struct em C

TreeNode* create_node() {
    TreeNode* node = (TreeNode*)malloc(sizeof(TreeNode));
    node->feature = -1; node->value = 0.0; node->cluster = -1;
    node->is_leaf = false; node->left = NULL; node->right = NULL;
    return node;
}//construtor tbm fora do struct devido a restrições da linguagem

typedef struct {
    TreeNode* node;
    bool* valid_data;
    bool* valid_centers;
    int** cuts_matrix;
    int n_valid;              //quant de pontos no nó
    int k_valid;               //quant de centros no nó
    int** data_order_global;   //ids globais de ponto ordenados por dim
    int* dist_order_global;    //ids glovais de centro ordenados por distância
    int* map_data;             //[n_valid]local->global (ordem = ordem de dist_order_global)
} NodeTask;//estrutrua para a bfs que vai construir a árvore, que guarda tudo que o nodo vai precisar

int** clone_cuts_matrix(int** original, int d) {
    int** copy = (int**)malloc(d * sizeof(int*));
    for (int i = 0; i < d; i++) {
        copy[i] = (int*)malloc(2 * sizeof(int));
        copy[i][0] = original[i][0]; copy[i][1] = original[i][1];
    }
    return copy;
}

bool* clone_bool_array(bool* original, int size) {
    bool* copy = (bool*)malloc(size * sizeof(bool));
    for (int i = 0; i < size; i++) copy[i] = original[i];
    return copy;
}

//filtra array já ordenado, mantendo só os que passam a máscara desse nó mantendo ordenaçao anterior sem precisar ordenar dnv aqui
int* filter_ordered_array(int* ordered_in, int len_in, bool* mask, int* len_out) {
    int* out = (int*)malloc(len_in * sizeof(int));
    int count = 0;
    for (int i = 0; i < len_in; i++) {
        int id = ordered_in[i];
        if (mask[id]) {
            out[count++] = id;
        }
    }
    *len_out = count;
    return out;
}

//dist_order_global do pai,e o map_data do pai tbm pra saber de que amostra cada linha ta falando
//esse arqui filtra para o nodo tanto por amostra quando por centroide, para atualizar aquela matriz de amostras e centroides ordenados por distâncias para cada amostra sabe
int* filter_dist_order(int* parent_dist_order, int* parent_map_data,
                       int n_valid_parent, int k_valid_parent,
                       bool* child_valid_data, bool* child_valid_centers,
                       int k_valid_child,
                       int* out_n_valid) {
    int* temp = (int*)malloc((size_t)n_valid_parent * k_valid_child * sizeof(int));
    int n_valid_child = 0;

    for (int row = 0; row < n_valid_parent; row++) {
        int global_point = parent_map_data[row];
        if (!child_valid_data[global_point]) continue;

        int col_count = 0;
        int* row_ptr = parent_dist_order + (size_t)row * k_valid_parent;
        for (int c = 0; c < k_valid_parent; c++) {
            int global_center = row_ptr[c];
            if (child_valid_centers[global_center]) {
                temp[(size_t)n_valid_child * k_valid_child + col_count] = global_center;
                col_count++;
            }
        }
        n_valid_child++;
    }
    *out_n_valid = n_valid_child;
    return temp;
}

//esse é um dos pontos principais da gpu. cada thread calcula distância (euclidiana) entre um ponto e um centroide
__global__ void calc_distances_kernel(const double* data, const double* centers, double* dists, int n, int k, int d) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < k) {
        double sum = 0.0;
        for (int j = 0; j < d; j++) {
            double diff = data[row * d + j] - centers[col * d + j];
            sum += diff * diff;
        }
        dists[row * k + col] = sum;
    }
}

//função que vai chamar o calc_distances_kernel. o bom é que ela já calcula tbm o número de threads e blocos de acordo com a dimensão do dataset entao nao precisa ficar mudando isso manualemtne no codigo
//n x k pares de threads
double* get_distances(double** data, int n, int d, double** centers, int k) {
    double* h_data_flat = (double*)malloc(n * d * sizeof(double));
    double* h_centers_flat = (double*)malloc(k * d * sizeof(double));

    for(int i = 0; i < n; i++) {
        for(int j = 0; j < d; j++) h_data_flat[i * d + j] = data[i][j];
    }
    for(int i = 0; i < k; i++) {
        for(int j = 0; j < d; j++) h_centers_flat[i * d + j] = centers[i][j];
    }

    double *d_data, *d_centers, *d_dists;
    cudaMalloc((void**)&d_data, n * d * sizeof(double));
    cudaMalloc((void**)&d_centers, k * d * sizeof(double));
    cudaMalloc((void**)&d_dists, n * k * sizeof(double));

    cudaMemcpy(d_data, h_data_flat, n * d * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_centers, h_centers_flat, k * d * sizeof(double), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((k + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (n + threadsPerBlock.y - 1) / threadsPerBlock.y);

    calc_distances_kernel<<<numBlocks, threadsPerBlock>>>(d_data, d_centers, d_dists, n, k, d);

    cudaDeviceSynchronize();
    double* h_dists = (double*)malloc(n * k * sizeof(double));
    cudaMemcpy(h_dists, d_dists, n * k * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_data); cudaFree(d_centers); cudaFree(d_dists);
    free(h_data_flat); free(h_centers_flat);

    return h_dists;
}

//aqui é para ordenar por amostra os centroides mais próximos. use primitiva do cuda cub::DeviceSegmentedSort para ordenar para todas as amostras de uma vez
//entao mesmo essa funçao nao sendo kernel ela usa cuda
int* compute_global_dist_order(double* global_distances, int n, int k) {
    double *d_keys_in, *d_keys_out;
    int *d_values_in, *d_values_out, *d_offsets;

    cudaMalloc(&d_keys_in, (size_t)n * k * sizeof(double));
    cudaMalloc(&d_keys_out, (size_t)n * k * sizeof(double));
    cudaMalloc(&d_values_in, (size_t)n * k * sizeof(int));
    cudaMalloc(&d_values_out, (size_t)n * k * sizeof(int));
    cudaMalloc(&d_offsets, (n + 1) * sizeof(int));

    cudaMemcpy(d_keys_in, global_distances, (size_t)n * k * sizeof(double), cudaMemcpyHostToDevice);

    int* h_values_in = (int*)malloc((size_t)n * k * sizeof(int));
    int* h_offsets = (int*)malloc((n + 1) * sizeof(int));
    for (int i = 0; i < n; i++) {
        h_offsets[i] = i * k;
        for (int c = 0; c < k; c++) h_values_in[i * k + c] = c;
    }
    h_offsets[n] = n * k;

    cudaMemcpy(d_values_in, h_values_in, (size_t)n * k * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets, h_offsets, (n + 1) * sizeof(int), cudaMemcpyHostToDevice);

    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    cub::DeviceSegmentedSort::SortPairs(d_temp_storage, temp_storage_bytes,
        d_keys_in, d_keys_out, d_values_in, d_values_out,
        n * k, n, d_offsets, d_offsets + 1);//a primeira chamada é só pra biblioteca calcular quanta memoria auxiliar ela precisa

    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    cub::DeviceSegmentedSort::SortPairs(d_temp_storage, temp_storage_bytes,
        d_keys_in, d_keys_out, d_values_in, d_values_out,
        n * k, n, d_offsets, d_offsets + 1);//aqui sim ordena

    int* h_order = (int*)malloc((size_t)n * k * sizeof(int));
    cudaMemcpy(h_order, d_values_out, (size_t)n * k * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_keys_in); cudaFree(d_keys_out);
    cudaFree(d_values_in); cudaFree(d_values_out);
    cudaFree(d_offsets); cudaFree(d_temp_storage);
    free(h_values_in); free(h_offsets);

    return h_order;
}


//isso é outra coisa que a gpu ajuda bastante também. aqui tbm não usa kernel mas usa bibliotecas do cuda, no caso o thrust
//essa é a função que ordena as amostras por dimensão. diferente do sequencial ela roda só uma vez, não por nó, aí dps cada nó vai filtrar
int* compute_global_data_order(double** data, int n, int d) {
    int* h_order = (int*)malloc((size_t)d * n * sizeof(int));

    double* h_keys = (double*)malloc(n * sizeof(double));
    thrust::device_vector<double> d_keys(n);
    thrust::device_vector<int> d_idx(n);

    for (int dim = 0; dim < d; dim++) {
        for (int i = 0; i < n; i++) h_keys[i] = data[i][dim];

        thrust::copy(h_keys, h_keys + n, d_keys.begin());
        thrust::sequence(d_idx.begin(), d_idx.end());

        thrust::sort_by_key(d_keys.begin(), d_keys.end(), d_idx.begin());

        thrust::copy(d_idx.begin(), d_idx.end(), h_order + (size_t)dim * n);
    }

    free(h_keys);
    return h_order;
}

//aqui sim que a árvore vai ser construída
TreeNode* build_tree_bfs(double** data, int* data_count, double** centers, double* global_distances,
                         int* global_data_order, int* global_dist_order,
                         bool* initial_valid_data, bool* initial_valid_centers,
                         int n_total, int k_total, int d,
                         double depth_factor, int** initial_cuts_matrix) {

    TreeNode* root = create_node();
    int queue_capacity = k_total * 4;
    NodeTask* queue = (NodeTask*)malloc(queue_capacity * sizeof(NodeTask));
    int head = 0, tail = 0;

    queue[tail].node = root;
    queue[tail].valid_data = clone_bool_array(initial_valid_data, n_total);
    queue[tail].valid_centers = clone_bool_array(initial_valid_centers, k_total);
    queue[tail].cuts_matrix = clone_cuts_matrix(initial_cuts_matrix, d);

    queue[tail].n_valid = n_total;
    queue[tail].k_valid = k_total;//raiz tem todos os dados por isso ta recebendo tudo

    queue[tail].map_data = (int*)malloc(n_total * sizeof(int));
    for (int i = 0; i < n_total; i++) queue[tail].map_data[i] = i;

    //data_order_global da raiz é o global_data_order calculado antes
    queue[tail].data_order_global = (int**)malloc(d * sizeof(int*));
    for (int dim = 0; dim < d; dim++) {
        queue[tail].data_order_global[dim] = (int*)malloc(n_total * sizeof(int));
        memcpy(queue[tail].data_order_global[dim],
            global_data_order + (size_t)dim * n_total,
            n_total * sizeof(int));
    }

    //dist_order_global da raiz é o global_dist_order já calculado
    queue[tail].dist_order_global = (int*)malloc((size_t)n_total * k_total * sizeof(int));
    memcpy(queue[tail].dist_order_global, global_dist_order, (size_t)n_total * k_total * sizeof(int));

    tail++;

    while (head < tail) {//head é o proximo nodo e tail é onde q ele entraria na fila da bfs
        NodeTask current_task = queue[head++];

        TreeNode* node = current_task.node;
        bool* valid_data = current_task.valid_data;
        bool* valid_centers = current_task.valid_centers;
        int** cuts_matrix = current_task.cuts_matrix;

        //contar quantos pontos e centroides sao validos nesse nodo
        int n_valid = 0, k_valid = 0, first_valid_center = -1;
        for (int i = 0; i < n_total; i++) if (valid_data[i]) n_valid++;
        for (int i = 0; i < k_total; i++) {
            if (valid_centers[i]) {
                k_valid++;
                if (first_valid_center == -1) first_valid_center = i;
            }
        }

        if (k_valid == 1) {//se tiver só um centroide ja vira folha
            node->is_leaf = true; node->cluster = first_valid_center;
            free(valid_data); free(valid_centers);
            for (int i = 0; i < d; i++) free(cuts_matrix[i]); free(cuts_matrix);
            for (int dim = 0; dim < d; dim++) free(current_task.data_order_global[dim]);
            free(current_task.data_order_global);
            free(current_task.dist_order_global);
            free(current_task.map_data);
            continue;
        }

        double* data_f = (double*)malloc(n_valid * sizeof(double));
        int* data_count_f = (int*)malloc(n_valid * sizeof(int));
        double* centers_f = (double*)malloc(k_valid * sizeof(double));
        double* dists_f = (double*)malloc(n_valid * k_valid * sizeof(double));
        int* dist_order_f = (int*)malloc(n_valid * k_valid * sizeof(int));

        //mapeia índices válidos
        int* map_data = (int*)malloc(n_valid * sizeof(int));
        int idx_d = 0;
        for (int i = 0; i < n_total; i++) {
            if (valid_data[i]) {
                map_data[idx_d] = i;
                data_count_f[idx_d] = data_count[i];
                idx_d++;
            }
        }

        int* map_centers = (int*)malloc(k_valid * sizeof(int));
        int idx_c = 0;
        for (int i = 0; i < k_total; i++) {
            if (valid_centers[i]) {
                map_centers[idx_c] = i;
                idx_c++;
            }
        }

        //lookups reverso: dado um id global, qual o índice local dele nesse nodo
        int* local_id_of_data = (int*)malloc(n_total * sizeof(int));
        for (int i = 0; i < n_valid; i++) local_id_of_data[map_data[i]] = i;

        int* local_id_of_center = (int*)malloc(k_total * sizeof(int));
        for (int c = 0; c < k_valid; c++) local_id_of_center[map_centers[c]] = c;

        //dists_fpreenchido, custo de distância para cálculo do custo
        for (int i = 0; i < n_valid; i++) {
            int orig_i = map_data[i];
            for (int c = 0; c < k_valid; c++) {
                int orig_c = map_centers[c];
                dists_f[i * k_valid + c] = global_distances[orig_i * k_total + orig_c];
            }
        }

        //dist_order_f convertido do dist_order_global herdado para ids locais, sem nenhuma ordenação
        for (int i = 0; i < n_valid; i++) {
            for (int c = 0; c < k_valid; c++) {
                int global_center = current_task.dist_order_global[(size_t)i * k_valid + c];
                dist_order_f[i * k_valid + c] = local_id_of_center[global_center];
            }
        }

        int best_dim = -1;
        double best_cut = -INFINITY;
        double best_cost = INFINITY;

        int* data_order_f = (int*)malloc(n_valid * sizeof(int)); //cria aqui fora pq nao tem pq ficar recriadno isso para cada dimensao

        for (int dim = 0; dim < d; dim++) {
            for (int i = 0; i < n_valid; i++) {
                data_f[i] = data[map_data[i]][dim];
            }
            for (int c = 0; c < k_valid; c++) {
                centers_f[c] = centers[map_centers[c]][dim];
            }

            for (int i = 0; i < n_valid; i++) {
                int global_point = current_task.data_order_global[dim][i];
                data_order_f[i] = local_id_of_data[global_point];
            }

            bool cut_left = cuts_matrix[dim][0] > 0;
            bool cut_right = cuts_matrix[dim][1] > 0;
            double ans[2] = {0.0, 0.0};

            best_cut_single_dim(data_f, data_order_f, data_count_f, centers_f, dists_f, dist_order_f, n_valid, k_valid, ans, depth_factor, cut_left, cut_right);

            if (ans[1] < best_cost) {
                best_cost = ans[1];
                best_cut = ans[0];
                best_dim = dim;
            }
        }

        free(data_order_f);
        free(local_id_of_data); free(local_id_of_center);
        free(map_data); free(map_centers);
        free(data_f); free(data_count_f); free(centers_f);
        free(dists_f); free(dist_order_f);

        if (best_cost == INFINITY || best_cut == -INFINITY) {
            node->is_leaf = true; node->cluster = first_valid_center;
            free(valid_data); free(valid_centers);
            for (int i = 0; i < d; i++) free(cuts_matrix[i]); free(cuts_matrix);
            for (int dim = 0; dim < d; dim++) free(current_task.data_order_global[dim]);
            free(current_task.data_order_global);
            free(current_task.dist_order_global);
            free(current_task.map_data);
            continue;
        }

        node->feature = best_dim; node->value = best_cut;
        node->left = create_node(); node->right = create_node();

        //filho esquerdo
        queue[tail].node = node->left;
        queue[tail].valid_data = clone_bool_array(valid_data, n_total);
        queue[tail].valid_centers = clone_bool_array(valid_centers, k_total);
        queue[tail].cuts_matrix = clone_cuts_matrix(cuts_matrix, d);
        for (int i = 0; i < n_total; i++) {
            if (valid_data[i] && data[i][best_dim] > best_cut) queue[tail].valid_data[i] = false;
        }
        for (int i = 0; i < k_total; i++) {
            if (valid_centers[i] && centers[i][best_dim] > best_cut) queue[tail].valid_centers[i] = false;
        }
        queue[tail].cuts_matrix[best_dim][0] += 1;

        {
            int k_valid_left = 0;
            for (int i = 0; i < k_total; i++) if (queue[tail].valid_centers[i]) k_valid_left++;

            int len_map;
            queue[tail].map_data = filter_ordered_array(current_task.map_data, n_valid,
                                                          queue[tail].valid_data, &len_map);

            queue[tail].data_order_global = (int**)malloc(d * sizeof(int*));
            int len_check;
            for (int dim = 0; dim < d; dim++) {
                queue[tail].data_order_global[dim] = filter_ordered_array(
                    current_task.data_order_global[dim], n_valid,
                    queue[tail].valid_data, &len_check);
            }

            int n_valid_left;
            queue[tail].dist_order_global = filter_dist_order(
                current_task.dist_order_global, current_task.map_data,
                n_valid, k_valid,
                queue[tail].valid_data, queue[tail].valid_centers,
                k_valid_left, &n_valid_left);

            queue[tail].n_valid = len_map;
            queue[tail].k_valid = k_valid_left;
        }
        tail++;

        //filho direito
        queue[tail].node = node->right;
        queue[tail].valid_data = clone_bool_array(valid_data, n_total);
        queue[tail].valid_centers = clone_bool_array(valid_centers, k_total);
        queue[tail].cuts_matrix = clone_cuts_matrix(cuts_matrix, d);
        for (int i = 0; i < n_total; i++) {
            if (valid_data[i] && data[i][best_dim] <= best_cut) queue[tail].valid_data[i] = false;
        }
        for (int i = 0; i < k_total; i++) {
            if (valid_centers[i] && centers[i][best_dim] <= best_cut) queue[tail].valid_centers[i] = false;
        }
        queue[tail].cuts_matrix[best_dim][1] += 1;

        {
            int k_valid_right = 0;
            for (int i = 0; i < k_total; i++) if (queue[tail].valid_centers[i]) k_valid_right++;

            int len_map;
            queue[tail].map_data = filter_ordered_array(current_task.map_data, n_valid,
                                                          queue[tail].valid_data, &len_map);

            queue[tail].data_order_global = (int**)malloc(d * sizeof(int*));
            int len_check;
            for (int dim = 0; dim < d; dim++) {
                queue[tail].data_order_global[dim] = filter_ordered_array(
                    current_task.data_order_global[dim], n_valid,
                    queue[tail].valid_data, &len_check);
            }

            int n_valid_right;
            queue[tail].dist_order_global = filter_dist_order(
                current_task.dist_order_global, current_task.map_data,
                n_valid, k_valid,
                queue[tail].valid_data, queue[tail].valid_centers,
                k_valid_right, &n_valid_right);

            queue[tail].n_valid = len_map;
            queue[tail].k_valid = k_valid_right;
        }
        tail++;

        if (tail >= queue_capacity - 2) {
            queue_capacity *= 2;
            queue = (NodeTask*)realloc(queue, queue_capacity * sizeof(NodeTask));
        }

        free(valid_data); free(valid_centers);
        for (int i = 0; i < d; i++) free(cuts_matrix[i]); free(cuts_matrix);
        for (int dim = 0; dim < d; dim++) free(current_task.data_order_global[dim]);
        free(current_task.data_order_global);
        free(current_task.dist_order_global);
        free(current_task.map_data);
    }

    free(queue);
    return root;
}

void print_tree_preorder(TreeNode* node, FILE* f) {
    if (!node) return;
    if (node->is_leaf) {
        fprintf(f, "LEAF %d\n", node->cluster);
    } else {
        fprintf(f, "SPLIT %d %.10f\n", node->feature, node->value);
    }
    if (!node->is_leaf) {
        print_tree_preorder(node->left, f);
        print_tree_preorder(node->right, f);
    }
}

int main() {
    int n, d, k;
    double depth_factor = 0.00;

    FILE* fmeta = fopen("meta.txt", "r");
    if (!fmeta) {
        printf("Erro ao abrir meta.txt!\n");
        return 1;
    }
    if (fscanf(fmeta, "%d %d %d", &n, &d, &k) != 3) {
        fclose(fmeta); return 1;
    }
    fclose(fmeta);

    printf("Dataset carregado: N=%d, D=%d, K=%d\n", n, d, k);

    double** data = (double**)malloc(n * sizeof(double*));
    int* data_count = (int*)malloc(n * sizeof(int));
    for(int i = 0; i < n; i++) {
        data[i] = (double*)malloc(d * sizeof(double));
        data_count[i] = 1;
    }

    double** centers = (double**)malloc(k * sizeof(double*));
    for(int i = 0; i < k; i++) centers[i] = (double*)malloc(d * sizeof(double));

    FILE* fx = fopen("X.bin", "rb");
    if (!fx) { printf("Erro ao abrir X.bin!\n"); return 1; }
    for (int i = 0; i < n; i++) {
        size_t lidos = fread(data[i], sizeof(double), d, fx);
        if ((int)lidos != d) {
            printf("Erro: leitura incompleta na linha %d de X.bin (lidos %zu de %d)\n", i, lidos, d);
            fclose(fx);
            return 1;
        }
    }
    fclose(fx);

    FILE* fs = fopen("S.bin", "rb");
    if (!fs) { printf("Erro ao abrir S.bin!\n"); return 1; }
    for (int i = 0; i < k; i++) {
        size_t lidos = fread(centers[i], sizeof(double), d, fs);
        if ((int)lidos != d) {
            printf("Erro: leitura incompleta no centroide %d de S.bin (lidos %zu de %d)\n", i, lidos, d);
            fclose(fs);
            return 1;
        }
    }
    fclose(fs);

    cudaFree(0); //warm up pq o tempo estava ficando estranho
    printf("Construindo arvore...\n");
    //clock_t inicio = clock();
    struct timespec inicio;
    clock_gettime(CLOCK_MONOTONIC, &inicio);

    int** cuts_matrix = (int**)malloc(d * sizeof(int*));
    for (int i = 0; i < d; i++) {
        cuts_matrix[i] = (int*)malloc(2 * sizeof(int));
        cuts_matrix[i][0] = 0; cuts_matrix[i][1] = 0;
    }

    bool* valid_data = (bool*)malloc(n * sizeof(bool));
    for (int i = 0; i < n; i++) valid_data[i] = true;

    bool* valid_centers = (bool*)malloc(k * sizeof(bool));
    for (int i = 0; i < k; i++) valid_centers[i] = true;

    double* global_distances = get_distances(data, n, d, centers, k);

    int* global_dist_order = compute_global_dist_order(global_distances, n, k);
    int* global_data_order = compute_global_data_order(data, n, d);

    TreeNode* root = build_tree_bfs(data, data_count, centers, global_distances,
                                 global_data_order, global_dist_order,
                                 valid_data, valid_centers, n, k, d, depth_factor, cuts_matrix);

    // clock_t fim = clock();
    // double tempo_ms = ((double)(fim - inicio) / CLOCKS_PER_SEC) * 1000.0;
    struct timespec fim;
    clock_gettime(CLOCK_MONOTONIC, &fim);
    double tempo_ms = (fim.tv_sec - inicio.tv_sec) * 1000.0 +(fim.tv_nsec - inicio.tv_nsec) / 1e6;
    printf("Arvore construida em %.3f ms!\n", tempo_ms);

    printf("Salvando estrutura da arvore em tree_v3.txt...\n");
    FILE* ftree = fopen("tree_v3.txt", "w");
    if (ftree == NULL) {
        printf("Erro ao criar o arquivo tree_v3.txt!\n");
    } else {
        print_tree_preorder(root, ftree);
        fclose(ftree);
        printf("Arquivo tree_v3.txt gerado com sucesso!\n");
    }

    free(global_distances);
    free(valid_data);
    free(valid_centers);
    for (int i = 0; i < d; i++)
        free(cuts_matrix[i]);
    free(cuts_matrix);
    for (int i = 0; i < n; i++)
        free(data[i]);
    free(data);
    free(data_count);
    for (int i = 0; i < k; i++)
        free(centers[i]);
    free(centers);
    free(global_dist_order);
    free(global_data_order);

    return 0;
}