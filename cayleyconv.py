from typing import Optional
import torch
from torch import Tensor
from torch.nn import Parameter

from torch.nn import Linear
import torch
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import OptTensor
from torch_geometric.utils import (
    add_self_loops,
    get_laplacian,
    to_dense_adj,
    add_self_loops
)

from jacobi import jacobi_sparse



class LinearComplex(torch.nn.Module):

    ''' 
    Linear layer for complex valued weights and inputs

    For a complex valued input $z = a + ib $ and a complex valued weight $M=M_R+iM_b, the output is
    $Mz = M_R a - M_I b + i ( M_I a + M_R b)$

    Parameters
    ----------
    in_features : int
        Number of input features
    out_features : int
        Number of output features
    bias : bool
        If True, adds a learnable complex bias to the output
    '''

    def __init__(self,in_features,out_features,bias=False) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        self.linear_real = torch.nn.Linear(in_features,out_features,bias=False)
        self.linear_imag = torch.nn.Linear(in_features,out_features,bias=False)

        if bias:
            self.bias_real = torch.nn.Parameter(torch.zeros(out_features))
            self.bias_imag = torch.nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias_real',None)
            self.register_parameter('bias_imag',None)

    def reset_parameters(self):
        self.linear_real.reset_parameters()
        self.linear_imag.reset_parameters()
        if self.bias:
            self.bias_real.data.zero_()
            self.bias_imag.data.zero_()

    def forward(self,x):
        real_output = self.linear_real(x.real) - self.linear_imag(x.imag)
        imag_output = self.linear_real(x.imag) + self.linear_imag(x.real)
        if self.bias:
            return torch.complex(real_output,imag_output) + torch.complex(self.bias_real,self.bias_imag)
        else:
            return torch.complex(real_output,imag_output)   
         
    def __repr__(self):
        return (f'{self.__class__.__name__}({self.in_features}, '
                f'{self.out_features}, bias={self.bias})')


class CayleyNetConvExact(MessagePassing):
    '''
    Implement the CayleyNet layer from the paper "CayleyNet: A Neural Network"
    <https://arxiv.org/abs/1906.04032>`_. Here we implement a version not using the Jacobi method.

    The CayleyNet layer is a generalization of the ChebNet layer. It is a
    convolutional layer that uses the Cayley transform to approximate the
    spectral decomposition of the graph Laplacian. 

    Parameters
    ----------
    in_channels : int, Number of input features
    out_channels : int, Number of output features
    r : int, Order of the Cayley transform
    normalization : str, Normalization to use on the Laplacian. Can be None, 'sym' or 'rw'. Default is None
    complexLayer : bool, If True, the layers used will be the LinearComplex layer. Else, they will be regular pytorch layers. Default is True
    **kwargs : Additional arguments to be passed to the MessagePassing class


    Forward
    -------
    x : Tensor, Input features
    edge_index : Tensor, Edge indices
    edge_weight : Tensor : Optionnal, Edge weights



    Example
    -------
    >>> import torch
    >>> from torch_geometric.data import Data
    >>> edge_index = torch.tensor([[0, 1, 1, 2],
    ...                            [1, 0, 2, 1]], dtype=torch.long)
    >>> x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
    >>> data = Data(x=x, edge_index=edge_index)
    >>> conv = CayleyNetConvExact(1, 16, r=5)
    >>> conv(data.x, data.edge_index)
    '''


    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 r :int = 5,
                 normalization : Optional[str] = None,
                 complexLayer :bool = True,
                 **kwargs
        ):

        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        assert normalization in [None, 'sym', 'rw'], 'Invalid normalization'
        assert r>0, 'r must be positive'
        
        # Define hyperparameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalization = normalization
        self.r = r

        # Define learnable parameters
        self.h = Parameter(torch.ones(1))
        self.c0 = Linear(in_channels,out_channels,bias=False)
        if complexLayer:
            self.c = torch.nn.ModuleList([LinearComplex(in_channels, out_channels) for _ in range(self.r)])
        else:
            self.c = torch.nn.ModuleList([Linear(in_channels, out_channels,dtype=torch.complex64,bias=False) for _ in range(self.r)])
        self.reset_parameters()


    def reset_parameters(self):
        self.h = Parameter(torch.ones(1))
        self.c0.reset_parameters()
        for lin in self.c:
            lin.reset_parameters()

    
    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: OptTensor = None,
        ):

        #Get the laplacian
        edge_index, edge_weight = get_laplacian(edge_index,edge_weight = edge_weight,
                                                normalization =self.normalization, dtype = x.dtype,num_nodes=x.size(self.node_dim))
        
        out =self.c0(x)

        x = x.to(torch.complex64)

        norm = to_dense_adj(edge_index, edge_attr= edge_weight).to(torch.complex64) # Very inefficient/expensive operation 
        
        zoomed_lap = self.h * norm
        id = torch.eye(x.size(0),device=x.device,dtype=torch.complex64)

        C_1 = zoomed_lap - 1j * id
        temp = torch.inverse(zoomed_lap + 1j * id)
        C_1 = torch.matmul(C_1, temp)
        C_j = C_1.clone()

        out_complex = self.c[0](torch.matmul(C_1, x))
        for j in range(1,self.r):
            C_j = torch.matmul(C_j, C_1)
            out_complex += self.c[j](torch.matmul(C_j, x))

        out += 2*torch.real(out_complex[0])

        return out

    def message(self, x_j: Tensor, norm: Tensor) -> Tensor:
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, r={self.r}, '
                f'normalization={self.normalization})')


class CayleyNetConv(MessagePassing):
    '''
    Implement the CayleyNet layer from the paper "CayleyNet: A Neural Network"
    <https://arxiv.org/abs/1906.04032>`_.
    
    The CayleyNet layer is a generalization of the ChebNet layer. It is a
    convolutional layer that uses the Cayley transform to approximate the
    spectral decomposition of the graph Laplacian. 

    Parameters
    ----------
    in_channels : int, Number of input features
    out_channels : int, Number of output features
    r : int, Order of the Cayley transform
    K : int, Number of iterations for the Jacobi method
    normalization : str, Normalization to use on the Laplacian. Can be None, 'sym' or 'rw'. Default is None
    complexLayer : bool, If True, the layers used will be the LinearComplex layer. Else, they will be regular pytorch layers. Default is True
    **kwargs : Additional arguments to be passed to the MessagePassing class

    Forward
    -------
    x : Tensor, Input features
    edge_index : Tensor, Edge indices
    edge_weight : Tensor : Optionnal, Edge weights

    Example
    -------
    >>> import torch
    >>> from torch_geometric.data import Data
    >>> edge_index = torch.tensor([[0, 1, 1, 2],
    ...                            [1, 0, 2, 1]], dtype=torch.long)
    >>> x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
    >>> data = Data(x=x, edge_index=edge_index)
    >>> conv = CayleyNetConvJacobi(1, 16, r=5)
    >>> conv(data.x, data.edge_index)
    '''


    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 r: int,
                 K : Optional[int] = 6,
                 normalization : Optional[str] = None,
                 complexLayer :bool = True,
                 **kwargs
        ):

        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        assert normalization in [None, 'sym', 'rw'], 'Invalid normalization'
        assert K>0, 'K must be positive'
        assert r>0, 'r must be positive'

        
        # Define hyperparameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalization = normalization
        self.K = K
        self.r = r

        # Define learnable parameters
        self.c0 = Linear(in_channels, out_channels, bias=False)
        self.h = Parameter(torch.ones(1))
        if complexLayer:
            self.c = torch.nn.ModuleList([LinearComplex(in_channels, out_channels) for _ in range(self.r)])
        else:
            self.c = torch.nn.ModuleList([Linear(in_channels, out_channels,dtype=torch.complex64,bias=False) for _ in range(self.r)])
        self.reset_parameters()
    
    def reset_parameters(self):
        for lin in self.c:
            lin.reset_parameters()
        self.h = Parameter(torch.ones(1))
        self.c0.reset_parameters()

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: OptTensor = None,
        batch: OptTensor = None):

        num_nodes = x.shape[0]

        #Get the laplacian
        edge_index, norm = get_laplacian(edge_index,normalization= self.normalization, edge_weight =  edge_weight,
                                        num_nodes=x.shape[0],dtype=torch.complex64)

        zoomed_lap = self.h * norm  

        neg_idx, neg_norm = add_self_loops(edge_index=edge_index,edge_attr=zoomed_lap,fill_value=torch.tensor(-1j))  # h*Delta - i*Id
        pos_idx, pos_norm = add_self_loops(edge_index=edge_index,edge_attr=zoomed_lap,fill_value=torch.tensor(1j))  # h*Delta + i*Id         

        out = self.c0(x)
        
        #Jacobi method
        out_complex = 0+0j
        y_j = x.to(torch.complex64)
        factor = torch.sparse_coo_tensor(neg_idx, neg_norm, torch.Size([num_nodes,num_nodes]),device=x.device)
        b_j = torch.sparse.mm(factor,y_j)
        for j in range(self.r):
            y_j = jacobi_sparse(neg_idx,neg_norm,b_j,self.K,num_nodes)
            out_complex += self.c[j](y_j)
            b_j = torch.sparse.mm(factor,y_j)
        
        
        return out+2*out_complex.real

    def message(self, x_j: Tensor, norm: Tensor):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, r={self.r}, K={self.K}, '
                f'normalization={self.normalization})')