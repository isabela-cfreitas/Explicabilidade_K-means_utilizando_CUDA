%%writefile tree_full.c
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>
#include <float.h>
#include <string.h>

//bestcut.c

double *array;

int compare(const void *a, const void *b) {
  int ia = *(int *) a;
  int ib = *(int *) b;
  int ans = 0;
  if (array[ia] < array[ib]) {
    ans = -1;
  }
  else if (array[ia] > array[ib]) {
    ans = 1;
  }
  return ans;
}

typedef struct {
    double val;
    int idx;
} ValIndexer;

int compare_val_indexer(const void *a, const void *b) {
    double va = ((ValIndexer*)a)->val;
    double vb = ((ValIndexer*)b)->val;
    if (va < vb) return -1;
    if (va > vb) return 1;
    return 0;
}

int *order_data(double *data, int n) {
  int i;
    int *ans = (int *)malloc(sizeof(int) * n);
  for (i = 0; i < n; i++) {
    ans[i] = i;
  }
  array = data;
  qsort(ans, n, sizeof(*ans), compare);
  return ans;
}

double get_height_cost(int curr_height,int N, int K, double alpha, double beta){
  int N_l, N_r,K_l,K_r;
  if (K==1){
    return N*curr_height;
  }

  K_l = (int)ceil(K*beta);
  K_l = (K_l>1)?K_l:1;
  K_l = (K_l<K)?K_l:K-1;
  K_r = K - K_l;
  if(N==1){
    if(K_l>K_r) N_l=1;
    else N_l =0;
  }
  else{
    N_l = (int)ceil(N*alpha);
    N_l = (N_l>1)? N_l:1;
    N_l = (N_l<N)? N_l:N-1;
  }

  N_r = N - N_l;

  return get_height_cost(curr_height+1,N_l,K_l,alpha,beta)+
         get_height_cost(curr_height+1,N_r,K_r,alpha,beta);
}

double get_cur_height_cost(int N_lAux, int N_rAux,
                           int K_lAux, int K_rAux,
                           int n,
                           float alpha, float beta,
                           bool cut_left, bool cut_right) {
  double ans = 0;
  double height_cost_left = get_height_cost(1,N_lAux,K_lAux,alpha,beta);
  double height_cost_right = get_height_cost(1,N_rAux,K_rAux,alpha,beta);
  if (cut_left) {
    height_cost_left -= (double)N_lAux;
  }
  if (cut_right) {
    height_cost_right -= (double)N_rAux;
  }
  ans = (height_cost_left + height_cost_right)/n;
  return ans;
}

void best_cut_single_dim(double *data, int *data_order, int *data_count, double *centers,
                         double *distances, int *dist_order, int n, int k,
                         double *ans, double height_factor,
                         bool cut_left, bool cut_right) {
  int i, j, c, cur_c, ix, ic;
  int idx_data = 0;
  int idx_centers = 0;
  int N_l=0;
  int N_r =0;
  int K_r =0;
  int K_l =0;
  int N_lAux,N_rAux,K_lAux,K_rAux;
  double nxt_cut;
  double best_cut;
  double alpha;
  double beta;
  double cur_cost = 0;
  double init_dist_cost=0;
  double cur_dist_cost =0;
  double cur_height_cost =0;
  double best_cost;
  double old_data_cost;
  double max_cut;
  int *left_data_mask = (int *)malloc(sizeof(int) * n);
  int *left_centers_mask = (int *)malloc(sizeof(int) * k);
  int *best_in_left = (int *)malloc(sizeof(int) * n);
  int *best_in_right = (int *)malloc(sizeof(int) * n);
  double *cur_dist_costs = (double *)malloc(sizeof(double) * n);
  int *cur_centers = (int *)malloc(sizeof(int) * n);
  int *centers_order = (int *)malloc(sizeof(int) * k);
  bool valid = false;

  for (i = 0; i < k; i++) {
    centers_order[i] = i;
  }
  array = centers;
  qsort(centers_order, k, sizeof(*centers_order), compare);

  c = centers_order[0];
  nxt_cut = centers[c];
  c = centers_order[k-1];
  max_cut = centers[c];

  for (i = 0; i < n; i++) {
    c = dist_order[i * k];
    cur_centers[i] = c;
    cur_dist_costs[i] = distances[i*k + c] * data_count[i];
    cur_dist_cost += cur_dist_costs[i];
  }
  init_dist_cost = cur_dist_cost;
  c = centers_order[0];

  for (i = 0; i < n; i++) {
    best_in_left[i] = c;
  }

  for (i = 0; i < n; i++) {
    best_in_right[i] = 0;
  }

  ix = data_order[idx_data];
  while ( (data[ix] <= nxt_cut) && idx_data < n) {
    idx_data++;
    if (idx_data < n) {
      ix = data_order[idx_data];
    }
  }

  while ( (centers[c] <= nxt_cut) && (idx_centers < k) ) {
    idx_centers++;
    if (idx_centers < k) {
      c = centers_order[idx_centers];
      if (centers[c] <= nxt_cut) {
        for (i = 0; i < n; i++) {
          cur_c = best_in_left[i];
          if (distances[i*k + cur_c] > distances[i*k + c]) {
            best_in_left[i] = c;
          }
        }
      }
    }
  }

  if (idx_centers == k) {
    ans[0] = -1;
    ans[1] = INFINITY;
    free(cur_dist_costs);
    free(left_data_mask);
    free(left_centers_mask);
    free(best_in_left);
    free(best_in_right);
    free(cur_centers);
    //free(data_order);
    free(centers_order);
    return;
  }

  for (i = 0; i < n; i++) {
    left_data_mask[i] = data[i] <= nxt_cut;
    if(left_data_mask[i]) N_l++;
    else N_r++;
  }

  for (i = 0; i < k; i++) {
    left_centers_mask[i] = centers[i] <= nxt_cut;
    if(left_centers_mask[i]) K_l++;
    else K_r++;
  }

  for (i = 0; i < n; i++) {
    cur_c = cur_centers[i];
    if (left_data_mask[i]) {
      c = best_in_left[i];
    }
    else {
      j = best_in_right[i];
      c = dist_order[i*k + j];
      while (left_centers_mask[c]) {
        j++;
        c = dist_order[i*k + j];
      }
      best_in_right[i] = j;
    }
    if (c != cur_c) {
      old_data_cost = cur_dist_costs[i];
      cur_dist_costs[i] = distances[i*k + c] * data_count[i];
      cur_dist_cost += (cur_dist_costs[i] - old_data_cost);
      cur_centers[i] = c;
    }
  }

  N_lAux = (N_l>1)? N_l:1;
  N_lAux = (N_lAux==n)? n-1:N_lAux;
  N_rAux = n - N_lAux;
  K_lAux = (K_l>1)?K_l:1;
  K_lAux = (K_lAux==k)?k-1:K_lAux;
  K_rAux = k - K_lAux;
  alpha = (double)N_lAux/n;
  beta = (double) K_lAux/k;

  cur_height_cost = get_cur_height_cost(N_lAux, N_rAux, K_lAux, K_rAux,
                                        n, alpha, beta, cut_left, cut_right);
  cur_cost = cur_dist_cost/init_dist_cost+height_factor*cur_height_cost;

  if ((idx_centers != 0) && (idx_centers != k) &&
      (idx_data >= idx_centers) &&
      ( (n - idx_data) >= (k - idx_centers) ) ) {
       best_cut = nxt_cut;
       best_cost = cur_cost;
  }
  else {
    best_cut = -1;
    best_cost = INFINITY;
  }

  while ( (idx_data < n) && (idx_centers < k) ) {
    ix = data_order[idx_data];
    ic = centers_order[idx_centers];
    if (data[ix] < centers[ic]) {
      nxt_cut = data[ix];
    }
    else {
      nxt_cut = centers[ic];
    }
    if (nxt_cut >= max_cut) {
      break;
    }

    while ( (idx_data < n) && (data[ix] <= nxt_cut) ) {
      old_data_cost = cur_dist_costs[ix];
      left_data_mask[ix] = 1;
      N_l++;
      N_r--;
      c = best_in_left[ix];
      cur_centers[ix] = c;
      cur_dist_costs[ix] = distances[ix*k + c] * data_count[ix];
      cur_dist_cost += (cur_dist_costs[ix] - old_data_cost);
      idx_data++;
      if (idx_data < n) {
          ix = data_order[idx_data];
      }
    }

    while ( (idx_centers < k) && (centers[ic] <= nxt_cut) ) {
      left_centers_mask[ic] = 1;
      K_l++;
      K_r--;
      for (i = 0; i < n; i++) {
        old_data_cost = cur_dist_costs[i];
        cur_c = best_in_left[i];
        if (distances[i*k + ic] < distances[i*k + cur_c]) {
          best_in_left[i] = ic;
        }
        if (left_data_mask[i]) {
          if (best_in_left[i] == ic) {
            cur_centers[i] = ic;
            cur_dist_costs[i] = distances[i*k + ic] * data_count[i];
            cur_dist_cost += (cur_dist_costs[i] - old_data_cost);
          }
        }
        else if (cur_centers[i] == ic) {
          j = best_in_right[i];
          c = dist_order[i*k];
          while (left_centers_mask[c]) {
            j++;
            c = dist_order[i*k + j];
          }
          best_in_right[i] = j;
          cur_dist_costs[i] = distances[i*k + c] * data_count[i];
          cur_dist_cost += (cur_dist_costs[i] - old_data_cost);
          cur_centers[i] = c;
        }
      }
      idx_centers++;

      if (idx_centers < k) {
        ic = centers_order[idx_centers];
      }
    }

  N_lAux = (N_l>1)? N_l:1;
  N_lAux = (N_lAux>n)? n-1:N_lAux;
  N_rAux = n - N_lAux;
  K_lAux = (K_l>1)?K_l:1;
  K_lAux = (K_lAux>k)?k-1:K_lAux;
  K_rAux = k - K_lAux;
  alpha = (double)N_lAux/n;
  beta = (double) K_lAux/k;

  if ((idx_centers != 0) && (idx_centers != k) && (idx_data >= idx_centers)
      && ( (n - idx_data) >= (k - idx_centers) )  ) {
    valid = true;
    N_lAux = (N_l>1)? N_l:1;
    N_lAux = (N_lAux>n)? n-1:N_lAux;
    N_rAux = n - N_lAux;
    K_lAux = (K_l>1)?K_l:1;
    K_lAux = (K_lAux>k)?k-1:K_lAux;
    K_rAux = k - K_lAux;
    alpha = (double)N_lAux/n;
    beta = (double) K_lAux/k;

    if(K_lAux<=0 || K_rAux<=0){
      cur_height_cost = 100000;
      printf("weird case \n");
    }
    else{
      cur_height_cost = get_cur_height_cost(N_lAux, N_rAux, K_lAux, K_rAux,
                                            n, alpha, beta, cut_left,
                                            cut_right);
    }
    cur_cost = cur_dist_cost/init_dist_cost+height_factor*cur_height_cost;
  }
  else{
    valid = false;
  }

  if(valid){
    if (valid  && (cur_cost < best_cost) ) {
         best_cut = nxt_cut;
         best_cost = cur_cost;
    }
  }
  }

  ans[0] = best_cut;
  ans[1] = best_cost;

  free(cur_dist_costs);
  free(left_data_mask);
  free(left_centers_mask);
  free(best_in_left);
  free(best_in_right);
  free(cur_centers);
  //free(data_order);
  free(centers_order);
  return;
}

// Pré-computa, para cada dimensão, a ordem dos pontos (equivalente ao
// compute_global_data_order do CUDA, mas usando qsort ao invés de thrust)
int* compute_global_data_order(double** data, int n, int d) {
    int* h_order = (int*)malloc((size_t)d * n * sizeof(int));
    double* tmp_vals = (double*)malloc(n * sizeof(double));
    int* tmp_idx = (int*)malloc(n * sizeof(int));

    for (int dim = 0; dim < d; dim++) {
        for (int i = 0; i < n; i++) {
            tmp_vals[i] = data[i][dim];
            tmp_idx[i] = i;
        }
        array = tmp_vals;
        qsort(tmp_idx, n, sizeof(int), compare);
        memcpy(h_order + (size_t)dim * n, tmp_idx, n * sizeof(int));
    }

    free(tmp_vals);
    free(tmp_idx);
    return h_order;
}

// Pré-computa, para cada ponto, a ordem dos centros por distância
// (equivalente ao compute_global_dist_order do CUDA, mas com qsort por linha)
int* compute_global_dist_order(double* global_distances, int n, int k) {
    int* h_order = (int*)malloc((size_t)n * k * sizeof(int));
    ValIndexer* row = (ValIndexer*)malloc(k * sizeof(ValIndexer));

    for (int i = 0; i < n; i++) {
        for (int c = 0; c < k; c++) {
            row[c].val = global_distances[(size_t)i * k + c];
            row[c].idx = c;
        }
        qsort(row, k, sizeof(ValIndexer), compare_val_indexer);
        for (int c = 0; c < k; c++) {
            h_order[(size_t)i * k + c] = row[c].idx;
        }
    }

    free(row);
    return h_order;
}


//structs p arvore

typedef struct TreeNode {
    int feature;
    double value;
    int cluster;
    bool is_leaf;
    struct TreeNode* left;
    struct TreeNode* right;
} TreeNode;

typedef struct {
    double dist;
    int index;
} DistHelper;

int compare_dist_helper(const void* a, const void* b) {
    double diff = ((DistHelper*)a)->dist - ((DistHelper*)b)->dist;
    if (diff < 0) return -1;
    if (diff > 0) return 1;
    return 0;
}

TreeNode* create_node() {
    TreeNode* node = (TreeNode*)malloc(sizeof(TreeNode));
    node->feature = -1;
    node->value = 0.0;
    node->cluster = -1;
    node->is_leaf = false;
    node->left = NULL;
    node->right = NULL;
    return node;
}

typedef struct {
    TreeNode* node;
    bool* valid_data;
    bool* valid_centers;
    int** cuts_matrix;
    int n_valid;
    int k_valid;
    int** data_order_global;//data_f[data_order_f[0]], data_f[data_order_f[1]]
    int* dist_order_global;
    int* map_data;//map_data[idx_d] = i significa i = índice global; idx_d = índice local (0..n_valid-1)
} NodeTask;

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

double* get_distances(double** data, int n_data, int d, double** centers, int n_centers) {
    double* dists = (double*)malloc(n_data * n_centers * sizeof(double));
    for (int i = 0; i < n_data; i++) {
        for (int c = 0; c < n_centers; c++) {
            double sum = 0.0;
            for (int j = 0; j < d; j++) {
                double diff = data[i][j] - centers[c][j];
                sum += diff * diff;
            }
            dists[i * n_centers + c] = sum;
        }
    }
    return dists;
}

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
    queue[tail].k_valid = k_total;

    queue[tail].map_data = (int*)malloc(n_total * sizeof(int));
    for (int i = 0; i < n_total; i++) queue[tail].map_data[i] = i;

    queue[tail].data_order_global = (int**)malloc(d * sizeof(int*));
    for (int dim = 0; dim < d; dim++) {
        queue[tail].data_order_global[dim] = (int*)malloc(n_total * sizeof(int));
        memcpy(queue[tail].data_order_global[dim],
            global_data_order + (size_t)dim * n_total,
            n_total * sizeof(int));
    }

    queue[tail].dist_order_global = (int*)malloc((size_t)n_total * k_total * sizeof(int));
    memcpy(queue[tail].dist_order_global, global_dist_order, (size_t)n_total * k_total * sizeof(int));

    tail++;

    while (head < tail) {
        NodeTask current_task = queue[head++];

        TreeNode* node = current_task.node;
        bool* valid_data = current_task.valid_data;
        bool* valid_centers = current_task.valid_centers;
        int** cuts_matrix = current_task.cuts_matrix;

        int n_valid = 0, k_valid = 0, first_valid_center = -1;
        for (int i = 0; i < n_total; i++) if (valid_data[i]) n_valid++;
        for (int i = 0; i < k_total; i++) {
            if (valid_centers[i]) {
                k_valid++;
                if (first_valid_center == -1) first_valid_center = i;
            }
        }

        if (k_valid == 1) {
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

        int* local_id_of_data = (int*)malloc(n_total * sizeof(int));
        for (int i = 0; i < n_valid; i++) local_id_of_data[map_data[i]] = i;

        int* local_id_of_center = (int*)malloc(k_total * sizeof(int));
        for (int c = 0; c < k_valid; c++) local_id_of_center[map_centers[c]] = c;

        for (int i = 0; i < n_valid; i++) {
            int orig_i = map_data[i];
            for (int c = 0; c < k_valid; c++) {
                int orig_c = map_centers[c];
                dists_f[i * k_valid + c] = global_distances[orig_i * k_total + orig_c];
            }
        }

        for (int i = 0; i < n_valid; i++) {
            for (int c = 0; c < k_valid; c++) {
                int global_center = current_task.dist_order_global[(size_t)i * k_valid + c];
                dist_order_f[i * k_valid + c] = local_id_of_center[global_center];
            }
        }

        int best_dim = -1;
        double best_cut = -INFINITY;
        double best_cost = INFINITY;

        int* data_order_f = (int*)malloc(n_valid * sizeof(int));

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
        printf("Erro ao abrir meta.txt! Rode a celula do Python primeiro para gerar este arquivo.\n");
        return 1;
    }
    if (fscanf(fmeta, "%d %d %d", &n, &d, &k) != 3) {
        printf("Erro ao ler os metadados do arquivo meta.txt!\n");
        fclose(fmeta);
        return 1;
    }
    fclose(fmeta);

    printf("Dataset carregado com sucesso: N=%d amostras, D=%d dimensoes, K=%d clusters\n", n, d, k);

    double** data = (double**)malloc(n * sizeof(double*));
    int* data_count = (int*)malloc(n * sizeof(int));
    for(int i = 0; i < n; i++) {
        data[i] = (double*)malloc(d * sizeof(double));
        data_count[i] = 1;
    }

    double** centers = (double**)malloc(k * sizeof(double*));
    for(int i = 0; i < k; i++) {
        centers[i] = (double*)malloc(d * sizeof(double));
    }

    FILE* fx = fopen("X.bin", "rb");
    if (!fx) {
        printf("Erro ao abrir X.bin! Rode o Python primeiro.\n");
        return 1;
    }
    for (int i = 0; i < n; i++) {
        size_t lidos = fread(data[i], sizeof(double), d, fx);
        if (lidos != d) {
            printf("Aviso: Falha ao ler a linha %d completa de X.bin\n", i);
        }
    }
    fclose(fx);

    FILE* fs = fopen("S.bin", "rb");
    if (!fs) {
        printf("Erro ao abrir S.bin! Rode o Python primeiro.\n");
        return 1;
    }
    for (int i = 0; i < k; i++) {
        size_t lidos = fread(centers[i], sizeof(double), d, fs);
        if (lidos != d) {
            printf("Aviso: Falha ao ler o centroide %d completo de S.bin\n", i);
        }
    }
    fclose(fs);

    printf("Vai comecar a arvore\n");
    //clock_t inicio = clock();
    struct timespec inicio;
    clock_gettime(CLOCK_MONOTONIC, &inicio); //usando essa função para o tempo ficar justo entre as versões

    int** cuts_matrix = (int**)malloc(d * sizeof(int*));
    for (int i = 0; i < d; i++) {
        cuts_matrix[i] = (int*)malloc(2 * sizeof(int));
        cuts_matrix[i][0] = 0;
        cuts_matrix[i][1] = 0;
    }

    bool* valid_data = (bool*)malloc(n * sizeof(bool));
    for (int i = 0; i < n; i++) valid_data[i] = true;

    bool* valid_centers = (bool*)malloc(k * sizeof(bool));
    for (int i = 0; i < k; i++) valid_centers[i] = true;

    //medir o tempo só do pré processamento
    struct timespec inicio_pre, fim_pre;
    clock_gettime(CLOCK_MONOTONIC, &inicio_pre);

    //pre processamento, isso é oq muda do sequencial para o cuda
    double* global_distances = get_distances(data, n, d, centers, k);
    int* global_dist_order = compute_global_dist_order(global_distances, n, k);
    int* global_data_order = compute_global_data_order(data, n, d);

    clock_gettime(CLOCK_MONOTONIC, &fim_pre);
    double tempo_pre_ms = (fim_pre.tv_sec - inicio_pre.tv_sec) * 1000.0 +
                        (fim_pre.tv_nsec - inicio_pre.tv_nsec) / 1e6;
    printf("Pre-processamento (CPU): %.3f ms\n", tempo_pre_ms);

    TreeNode* root = build_tree_bfs(data, data_count, centers, global_distances,
                                     global_data_order, global_dist_order,
                                     valid_data, valid_centers, n, k, d,
                                     depth_factor, cuts_matrix);

    // clock_t fim = clock();
    // double tempo_ms = ((double)(fim - inicio) / CLOCKS_PER_SEC) * 1000.0;
    struct timespec fim;
    clock_gettime(CLOCK_MONOTONIC, &fim);
    double tempo_ms = (fim.tv_sec - inicio.tv_sec) * 1000.0 + (fim.tv_nsec - inicio.tv_nsec) / 1e6;

    printf("Arvore construida em %.3f ms!\n", tempo_ms);

    printf("Exportando a estrutura da arvore para tree_v2.txt...\n");
    FILE* f_tree = fopen("tree_v2.txt", "w");
    if (f_tree != NULL) {
        print_tree_preorder(root, f_tree);
        fclose(f_tree);
        printf("Arvore exportada com sucesso!\n");
    } else {
        printf("Erro ao criar o arquivo tree_v1.txt!\n");
    }


    free(global_distances);
    free(global_dist_order);
    free(global_data_order);
    free(valid_data);
    free(valid_centers);
    for (int i = 0; i < d; i++) free(cuts_matrix[i]);
    free(cuts_matrix);
    for (int i = 0; i < n; i++) free(data[i]);
    free(data);
    free(data_count);
    for (int i = 0; i < k; i++) free(centers[i]);
    free(centers);

    return 0;
}