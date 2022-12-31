import numpy as np
import torch

def jacobi_numpy(A, b, K):
    """Solves the equation Ax=b for x using the Jacobi method with K iterations."""
    inv_diag = np.diag(A)**-1
    inv_diag[inv_diag == np.inf] = 0
    off_diag = A - np.diag(np.diag(A))
    J = - np.diag(inv_diag)@off_diag
    b_1 = np.diag(inv_diag)@b
    x_k = b_1.copy()
    for k in range(K):
        x_k = J@x_k + b_1
    return x_k

def jacobi_dense(A, b, K):
    """Solves the equation Ax=b for x using the Jacobi method with K iterations.
    
    Parameters
    ----------
    A : torch.Tensor, shape (n, n)
    b : torch.Tensor, shape (n,1)
    K : int

    Returns
    -------
    x_k : torch.Tensor, shape (n,), solution estimate of Ax=b
    """
    
    inv_diag = torch.diag(A)**-1
    inv_diag[inv_diag == torch.inf] = 0
    off_diag = A - torch.diag(torch.diag(A))
    J = - torch.diag(inv_diag)@off_diag
    b_1 = torch.diag(inv_diag)@b
    x_k = b_1.clone()
    for k in range(K):
        x_k = J@x_k + b_1
    return x_k

def get_diag(edge_index, edge_weight):
    """Computes the diagonal coefficients of the matrix A from the edge_index and edge_weight"""
    mask = (edge_index[0, :] == edge_index[1, :])
    selected_edge_weight = edge_weight[mask]
    unique_edge_index = torch.unique(edge_index[1, :])
    output = torch.zeros_like(unique_edge_index, dtype=edge_weight.dtype)
    output.scatter_add_(0, edge_index[1, mask], selected_edge_weight)
    return output

def jacobi_sparse(edge_index,edge_weight,b,K, num_nodes):
    """Solves the equation Ax=b for x using the Jacobi method with K iterations
    for a square sparse matrix A.
    
    Parameters
    ----------
    edge_index : torch.Tensor, shape (2, n), indices of nonzero entries of A
    edge_weight : torch.Tensor, shape (n,), values of nonzero entries of A
    b : torch.Tensor, shape (n,p)
    K : int, number of iterations to approximate solution
    num_nodes : int, size of the full square matrix A

    Returns
    -------
    x_k : torch.Tensor, shape (n,p), solution estimate of Ax=b
    """
    off_diag_mask = edge_index[0]!=edge_index[1]
    off_diag_index = edge_index[:,off_diag_mask]
    diag_value = get_diag(edge_index,edge_weight)
    diag_index = torch.vstack([torch.arange(num_nodes),torch.arange(num_nodes)])
    diag_index = diag_index.to(edge_weight.device)
    
    inv_diag = diag_value**-1
    inv_diag[inv_diag == torch.inf] = 0

    off_diag = edge_weight[off_diag_mask]

    #sparse_coo_tensor to make the dot product Diag^-1 @ OffDiag
    J = - torch.sparse.mm(torch.sparse_coo_tensor(diag_index,inv_diag,(num_nodes,num_nodes)),torch.sparse_coo_tensor(off_diag_index,off_diag,(num_nodes,num_nodes)))#.to_dense()
    b_1 = torch.sparse.mm(torch.sparse_coo_tensor(diag_index,inv_diag,torch.Size([num_nodes,num_nodes])),b)
    x_k = b_1.clone()
    
    for k in range(K):
        x_k = torch.sparse.mm(J,x_k) + b_1

    return x_k 


