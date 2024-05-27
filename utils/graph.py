import numpy as np
import pickle
import scipy.sparse as sp

from scipy.sparse import linalg

def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj) # 转换为坐标格式的稀疏矩阵
    d = np.array(adj.sum(1)) # 度数矩阵
    d_inv_sqrt = np.power(d + 1e-6, -0.5).flatten() # 倒数平方根
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt) # 对角度数矩阵
    # 计算标准化的拉普拉斯矩阵 L = I - D^-1/2·A·D^-1/2 
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo() 
    return normalized_laplacian

def calculate_random_walk_matrix(adj_mx):
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))
    d_inv = np.power(d + 1e-6, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()
    return random_walk_mx

def calculate_reverse_random_walk_matrix(adj_mx):
    return calculate_random_walk_matrix(np.transpose(adj_mx))

def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T]) # 返回两个矩阵对应位置的元素最大值-》对称
    L = calculate_normalized_laplacian(adj_mx) # 计算标准化的拉普拉斯矩阵 L = I - D^-1/2·A·D^-1/2 
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM') # 计算L的最大特征值。
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L) # 转换为 CSR 格式的稀疏矩阵。
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype) # CSR 格式的稀疏单位矩阵。
    L = (2 / lambda_max * L) - I # 缩放操作
    return L.astype(np.float32)

def load_graph_data(pkl_filename):
    sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(pkl_filename)
    adj_mx = adj_mx - np.eye(adj_mx.shape[0])
    return sensor_ids, sensor_id_to_ind, adj_mx

def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data

def calculate_cheb_poly(L, Ks):
    n = L.shape[0]
    LL = [np.eye(n), L[:]]
    for i in range(2, Ks):
        LL.append(np.matmul(2 * L, LL[-1]) - LL[-2])
    return np.asarray(LL)

def sym_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum + 1e-6, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    # 计算对称标准化的邻接矩阵 L = D^-1/2·A·D^-1/2 
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()

def asym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum + 1e-6, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)
    # 计算非对称标准化的邻接矩阵 L = D^-1·A
    return d_mat.dot(adj).astype(np.float32).todense()