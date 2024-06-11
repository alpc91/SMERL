from typing import Optional, Tuple, Callable, Any, List
import jax
import jax.numpy as jnp
from flax import linen
from jax import random
from scipy.spatial.distance import cdist
from brax.training import networks
# from .scatter import scatter
from jax import vmap
from jax import lax
from jraph import segment_sum, segment_mean
import e3nn_jax as e3nn
from flax.linen.initializers import constant, orthogonal

ActivationFn = Callable[[jnp.ndarray], jnp.ndarray]
Initializer = Callable[..., Any]


def compute_distance_matrix(team0, team1):

    # 计算距离矩阵
    distance_matrix = jnp.sqrt(jnp.sum((team0[:, None, :] - team1[None, :, :]) ** 2, axis=2))
    return distance_matrix

@jax.jit
def simplified_matching(distance_matrix):
    N, M = distance_matrix.shape

    # 初始化匹配
    match = jnp.full((M,), 0)
    rev_match = jnp.full((N,), N)
    for i in range(M):
        # 找到距离最近的智能体
        j = jnp.argmin(distance_matrix[:, i])
        match = match.at[i].set(j)
        rev_match = rev_match.at[j].set(i+N)
        # 为避免重复匹配，将已匹配的智能体距离设置为极大值
        distance_matrix = distance_matrix.at[match[i], :].set(jnp.inf)

    return match, rev_match

def bipartite_matching(team0, team1, N):
    M = N
    assert N >= M, "智能体数量应大于或等于任务数量"

    distance_matrix = compute_distance_matrix(team0, team1)
    match, rev_match = simplified_matching(distance_matrix)

    # 计算总代价
    # total_cost = distance_matrix[match, jnp.arange(M)].sum()

    # 构建 edge_index 和 cost
    edge_index = jnp.array([match, jnp.arange(M)+N])
    # cost = distance_matrix[match, jnp.arange(M)]

    # 构建 agent_idx
    agent_idx = jnp.arange(N + M)
    agent_idx = agent_idx.at[N:].set(match)

    # 构建 task_idx
    task_idx = jnp.arange(N + M)
    task_idx = task_idx.at[:N].set(rev_match)

    return agent_idx, task_idx, edge_index


def construct_3d_basis_from_1_vector(u):
    # Z-axis vector
    e1 = jnp.array([0.0, 0.0, 1.0])
    e1 = jnp.broadcast_to(e1, u.shape)

    u2 = u - project_v2v(u, e1, axis=-1)
    e2 = normalize_vector(u2, axis=-1)
    e3 = jnp.cross(e1, e2, axis=-1)  # Orthogonal to both e1 and e2

    # Constructing basis matrix
    mat = jnp.stack([e2, e3, e1], axis=-1)  # (3, 3)
    return mat


def construct_3d_basis_from_1_vectors(v1):
    g = jnp.ones_like(v1)
    g = g.at[..., 2].set(0)
    v1 = g * v1
    # print("g",g)
    u1 = normalize_vector(v1, axis=-1)
    # print("u1",u1)

    u2 = jnp.zeros_like(u1)
    u2 = u2.at[..., 0].set(-u1[..., 1])
    u2 = u2.at[..., 1].set(u1[..., 0])
    # e2 = normalize_vector(u2, dim=-1)
    # print("u2",u2)

    # u3 = jnp.cross(u1, u2, axis=-1)    # (N, L, 3)
    u3 = jnp.zeros_like(u1)
    u3 = u3.at[..., 2].set(1)

    mat = jnp.stack([
        u1, u2, u3
    ], axis=-1)  # (N, L, 3, 3_index)
    return mat


def construct_3d_basis_from_2_vectors(v1, v2):
    e1 = normalize_vector(v1, axis=-1)

    u2 = v2 - project_v2v(v2, e1, axis=-1)
    e2 = normalize_vector(u2, axis=-1)

    e3 = jnp.cross(e1, e2, axis=-1)    # (3)

    mat = jnp.stack([e1, e2, e3], axis=-1)  # (3, 3)
    return mat


def normalize_vector(v, axis, eps=1e-6):
    return v / (jnp.linalg.norm(v, ord=2, axis=axis, keepdims=True) + eps)


def project_v2v(v, e, axis):
    """
    Description:
        Project vector `v` onto vector `e`.
    Args:
        v:  (N, L, 3).
        e:  (N, L, 3).
    """
    return (e * v).sum(axis=axis, keepdims=True) * e

class DistanceRBF:
    def __init__(self, num_channels=64, start=0.0, stop=2.0):
        self.num_channels = num_channels
        self.start = start
        self.stop = stop * 10
        self.params = self.initialize_params()

    def initialize_params(self):
        offset = jnp.linspace(self.start, self.stop, self.num_channels - 2)
        coeff = -0.5 / (offset[1] - offset[0])**2
        return {'offset': offset, 'coeff': coeff}

    def __call__(self, dist, dim):
        assert dist.shape[dim] == 1
        dist = dist * 10
        offset_shape = [1] * len(dist.shape)
        offset_shape[dim] = -1

        offset = self.params['offset'].reshape(offset_shape)
        coeff = self.params['coeff']

        overflow_symb = (dist >= self.stop).astype(jnp.float32)
        underflow_symb = (dist < self.start).astype(jnp.float32)
        y = dist - offset
        y = jnp.exp(coeff * jnp.square(y))
        return jnp.concatenate([underflow_symb, y, overflow_symb], axis=dim)



class BaseMLP(linen.Module):
    input_dim: int
    hidden_dim: int
    output_dim: int
    activation: ActivationFn = linen.relu
    kernel_init: Initializer = jax.nn.initializers.lecun_uniform()
    bias: bool = True
    residual: bool = False
    last_act: bool = False
    flat: bool = False
    kernel_init: Initializer = jax.nn.initializers.lecun_uniform()


    def setup(self):
        if self.flat:
            self.activation = linen.tanh
            self.hidden_dim = 4 * self.hidden_dim
        if self.residual:
            assert self.output_dim == self.input_dim
        layers = [
            linen.Dense(self.hidden_dim,
                        kernel_init=self.kernel_init,
                        use_bias=self.bias),
            self.activation,
            linen.Dense(self.hidden_dim,
                        kernel_init=self.kernel_init,
                        use_bias=self.bias),
            self.activation,
            linen.Dense(self.output_dim,
                        kernel_init=self.kernel_init,
                        use_bias=self.bias),
        ]
        if self.last_act:
            layers.append(self.activation)
        self.mlp = linen.Sequential(layers)

    
    def __call__(self, x):
        return self.mlp(x) + x if self.residual else self.mlp(x)
    

class SGNNMessagePassingLayer(linen.Module):
    hidden_dim: int
    activation: ActivationFn = linen.relu
    kernel_init: Initializer = jax.nn.initializers.lecun_uniform()
    vector_dim: int = 32
    # gravity_axis: Optional[int] = None
    

    def setup(self):
        self.net = BaseMLP(input_dim=0,
                           hidden_dim=self.hidden_dim,
                           output_dim=self.vector_dim * self.vector_dim + self.hidden_dim,
                           activation=self.activation,
                           residual=False,
                           last_act=False,
                           flat=False)
        self.self_net = BaseMLP(input_dim=0,
                                hidden_dim=self.hidden_dim,
                                output_dim=self.vector_dim * self.vector_dim + self.hidden_dim,
                                activation=self.activation,
                                residual=False,
                                last_act=False,
                                flat=False)
        self.embedding1 = linen.Dense(self.vector_dim,# if self.gravity_axis is None else 31,
                        kernel_init=self.kernel_init,
                        use_bias=False)
        self.embedding2 = linen.Dense(self.vector_dim,# if self.gravity_axis is None else 31,
                        kernel_init=self.kernel_init,
                        use_bias=False)
        
    
    def __call__(self, f, s, edge_index, edge_f=None, edge_s=None):
        if edge_index.shape[1] == 0:
            f_c, s_c = jnp.zeros_like(f), jnp.zeros_like(s)
        else:
            _f = jnp.concatenate((f[edge_index[0]], f[edge_index[1]]), axis=-1)
            if edge_f is not None:
                _f = jnp.concatenate((_f, edge_f), axis=-1)  # [M, 3, 2F+Fe]
            _s = jnp.concatenate((s[edge_index[0]], s[edge_index[1]]), axis=-1)
            if edge_s is not None:
                _s = jnp.concatenate((_s, edge_s), axis=-1)  # [M, 2S]
            _f = self.embedding1(_f)   # [M, 3, 31 or self.vector_dim]
            _f_T = jnp.swapaxes(_f, -1, -2)
            f2s = jnp.einsum('bij,bjk->bik', _f_T, _f)  # [M, vector_dim, vector_dim]
            f2s = f2s.reshape(f2s.shape[0], -1)  # [M, vector_dim*vector_dim]
            F_norm = jnp.linalg.norm(f2s, axis=-1, keepdims=True) + 1.0
            f2s = jnp.concatenate((f2s, _s), axis=-1)  # [M, vector_dim*vector_dim+2S+Se]
            c = self.net(f2s)  # [M, vector_dim*vector_dim+H]
            c = c / F_norm
            f_c, s_c = c[..., :-self.hidden_dim], c[..., -self.hidden_dim:]  # [M, vector_dim*vector_dim*F], [M, H]
            f_c = f_c.reshape(f_c.shape[0], self.vector_dim, -1)  # [M, vector_dim, vector_dim]
            f_c = jnp.einsum('bij,bjk->bik', _f, f_c)  # [M, 3, vector_dim]
            f_c = segment_mean(f_c, edge_index[0], num_segments=f.shape[0])  # [N, 3, vector_dim]
            s_c = segment_mean(s_c, edge_index[0], num_segments=f.shape[0])  # [N, H]
        # aggregate f_c and f
        temp_f = jnp.concatenate((f, f_c), axis=-1)  # [N, 3, 2vector_dim]
        temp_s = jnp.concatenate((s, s_c), axis=-1)  # [N, 2S]
        temp_f = self.embedding2(temp_f) # [N, 3, 31 or self.vector_dim]
        temp_f_T = jnp.swapaxes(temp_f, -1, -2) # [N, self.vector_dim, 3]
        temp_f2s = jnp.einsum('bij,bjk->bik', temp_f_T, temp_f)  # [N, self.vector_dim, self.vector_dim]
        temp_f2s = temp_f2s.reshape(temp_f2s.shape[0], -1)  # [N, self.vector_dim*self.vector_dim]
        F_norm = jnp.linalg.norm(temp_f2s, axis=-1, keepdims=True) + 1.0
        temp_f2s = jnp.concatenate((temp_f2s, temp_s), axis=-1)  # [N, self.vector_dim*self.vector_dim+2S]
        temp_c = self.self_net(temp_f2s)  # [N, 2F*F+H]
        temp_c = temp_c / F_norm
        temp_f_c, temp_s_c = temp_c[..., :-self.hidden_dim], temp_c[..., -self.hidden_dim:]  # [N, vector_dim*vector_dim], [N, H]
        temp_f_c = temp_f_c.reshape(temp_f_c.shape[0], self.vector_dim, -1)  # [N, self.vector_dim, vector_dim]
        temp_f_c = jnp.einsum('bij,bjk->bik', temp_f, temp_f_c)  # [N, 3, F] = [N, 3, self.vector_dim] * [N, self.vector_dim, vector_dim]
        f_out = temp_f_c + f
        s_out = temp_s_c + s
        return f_out, s_out


class SEMLP(linen.Module):
    hidden_dim: int
    activation: ActivationFn = linen.relu
    kernel_init: Initializer = jax.nn.initializers.lecun_uniform()
    vector_dim: int = 64
    # gravity_axis: Optional[int] = None
    

    def setup(self):
        self.self_net = BaseMLP(input_dim=0,
                                hidden_dim=self.hidden_dim,
                                output_dim=self.vector_dim * self.vector_dim + self.hidden_dim,
                                activation=self.activation,
                                residual=False,
                                last_act=False,
                                flat=False)
        
    
    def __call__(self, f, s):
        temp_f_T = jnp.swapaxes(f, -1, -2) # [self.vector_dim, 3]
        temp_f2s = jnp.einsum('ij,jk->ik', temp_f_T, f)  # [self.vector_dim, self.vector_dim]
        temp_f2s = temp_f2s.reshape(-1)  # [self.vector_dim*self.vector_dim]
        F_norm = jnp.linalg.norm(temp_f2s, axis=-1, keepdims=True) + 1.0
        temp_f2s = jnp.concatenate((temp_f2s, s), axis=-1)  # [self.vector_dim*self.vector_dim+S]
        temp_c = self.self_net(temp_f2s)  # [self.local_state_size*self.num_node]
        temp_c = temp_c / F_norm # [self.local_state_size*self.num_node]
        temp_f_c, temp_s_c = temp_c[..., :-self.hidden_dim], temp_c[..., -self.hidden_dim:]  # [=self.vector_dim * self.vector_dim], [H]
        temp_f_c = temp_f_c.reshape(self.vector_dim, -1)  # [self.vector_dim, self.vector_dim]
        temp_f_c = jnp.einsum('ij,jk->ik', f, temp_f_c)  # [N, 3, F] = [N, 3, self.vector_dim] * [N, self.vector_dim, F]
        f_out = temp_f_c + f
        s_out = temp_s_c + s
        return f_out, s_out
    


class HyperNetwork(linen.Module):
    """HyperNetwork for generating weights of QMix' mixing network."""
    hidden_dim: int
    output_dim: int
    init_scale: float

    @linen.compact
    def __call__(self, x):
        x = linen.Dense(self.hidden_dim, kernel_init=orthogonal(self.init_scale), bias_init=constant(0.))(x)
        x = linen.relu(x)
        x = linen.Dense(self.output_dim, kernel_init=orthogonal(self.init_scale), bias_init=constant(0.))(x)
        return x

class MixingNetwork(linen.Module):
    """
    Mixing network for projecting individual agent Q-values into Q_tot. Follows the original QMix implementation.
    """
    embedding_dim: int=64
    hypernet_hidden_dim: int=256
    init_scale: float=0.001

    @linen.compact
    def __call__(self, q_vals, states):
        
        n_agents = q_vals.shape[0]
        # q_vals = jnp.transpose(q_vals, (1, 2, 0)) # (time_steps, batch_size, n_agents)
        
        # hypernetwork
        w_1 = HyperNetwork(hidden_dim=self.hypernet_hidden_dim, output_dim=self.embedding_dim*n_agents, init_scale=self.init_scale)(states)
        b_1 = linen.Dense(self.embedding_dim, kernel_init=orthogonal(self.init_scale), bias_init=constant(0.))(states)
        w_2 = HyperNetwork(hidden_dim=self.hypernet_hidden_dim, output_dim=self.embedding_dim, init_scale=self.init_scale)(states)
        b_2 = HyperNetwork(hidden_dim=self.embedding_dim, output_dim=1, init_scale=self.init_scale)(states)
        
        # monotonicity and reshaping
        w_1 = jnp.abs(w_1.reshape(n_agents, self.embedding_dim))
        b_1 = b_1.reshape(1, self.embedding_dim)
        w_2 = jnp.abs(w_2.reshape(self.embedding_dim, 1))
        b_2 = b_2.reshape(1, 1)
    
        # mix
        hidden = linen.elu(jnp.matmul(q_vals[None, :], w_1) + b_1)
        q_tot  = jnp.matmul(hidden, w_2) + b_2
        
        return q_tot.squeeze() # (time_steps, batch_size)    

class SGNN(linen.Module):
    p_step: int = 2
    s_dim: int = 9
    hidden_dim: int = 256
    activation: ActivationFn = linen.relu
    kernel_init: Initializer = jax.nn.initializers.lecun_uniform()
    bias: bool = True
    cutoff: float = 0.10
    gravity_axis: Optional[int] = None
    local_state_size: int = 0
    torso: Optional[List[int]] = None
    num_node: int = 0
    edge_index: Optional[jnp.ndarray] = None
    edge_type: Optional[jnp.ndarray] = None
    obj_id: Optional[jnp.ndarray] = None
    out_dim: int = 1
    vector_dim: int = 32
    ctl_num: int = 2
    test: bool = False
    

    def setup(self):
        self.embedding_s = linen.Dense(2*self.vector_dim,
                        kernel_init=self.kernel_init,
                        use_bias=self.bias)
        self.embedding_f = linen.Dense(self.vector_dim,
                        kernel_init=self.kernel_init,
                        use_bias=False)
        
        # self.embedding_fout = linen.Dense(self.vector_dim//2,
        #         kernel_init=self.kernel_init,
        #         use_bias=False)
        # self.fusion = BaseMLP(input_dim=0,
        #             hidden_dim=self.hidden_dim,
        #             output_dim=self.vector_dim*2,
        #             activation=self.activation,
        #             residual=False,
        #             last_act=False,
        #             flat=False)
        
        
        self.object_message_passing = SGNNMessagePassingLayer(vector_dim=self.vector_dim, 
                                        hidden_dim=2*self.vector_dim, 
                                        activation=self.activation)

        self.dist_rbf = DistanceRBF(num_channels=64)


        if self.out_dim == 1:
            self.predictor = BaseMLP(input_dim=0,
                        hidden_dim=self.hidden_dim,
                        output_dim=1,
                        activation=self.activation,
                        residual=False,
                        last_act=False,
                        flat=False)
        else:
            # self.embedding_sp = linen.Dense(self.vector_dim,
            #                 kernel_init=self.kernel_init,
            #                 use_bias=self.bias)
            # self.embedding_fp = linen.Dense(self.vector_dim,
            #                 kernel_init=self.kernel_init,
            #                 use_bias=self.bias)
            
            self.embedding_u = linen.Dense(1,
                kernel_init=self.kernel_init,
                use_bias=False)
            
            out_dim = []
            for i in range(self.ctl_num):
                out_dim.append(self.torso[i+1]-self.torso[i]-1)
            predictor = []
            for i in range(self.ctl_num):
                predictor.append(BaseMLP(input_dim=0,
                        hidden_dim=self.hidden_dim,
                        output_dim=out_dim[i]*2,
                        activation=self.activation,
                        residual=False,
                        last_act=False,
                        flat=False))
            self.predictor = predictor

    
    def __call__(self, input):
        input = input.reshape(-1, self.local_state_size)

        x_p, v_p, a_p, h_p, yaw_normalized = input[..., :3], input[..., 3:6], input[..., 6:9], input[..., 9:-1], input[..., -1]  # [N, 3], [N, 3], [N, 3], [N, H]

        if self.out_dim == 1:
            x_m = x_p[self.torso, :]
            x_m = jnp.mean(x_m, axis=0, keepdims=True)  # [1, 3]
            x_p = x_p - x_m  # [N, 3]
            
            f_o = jnp.stack((x_p, v_p, a_p), axis=-1)
            f_o = f_o[self.torso, :, :] # [N_obj, 3, 3]
            s_o = h_p[self.torso, :] # [N_obj, H]


            if self.gravity_axis is not None:
                g_o = jnp.zeros_like(f_o)[..., 0]  # [N_obj, 3]
                g_o = g_o.at[..., self.gravity_axis].set(-9.8)
                f_o = jnp.concatenate((f_o, g_o[..., None]), axis=-1)  # [N_obj, 3, 3+1]
                s_o = jnp.concatenate((s_o, x_p[self.torso, self.gravity_axis][..., None]), axis=-1)  # [N_obj, H+1]
            
            edge_attr_inter_f = (f_o[self.edge_index[1],:,0] - f_o[self.edge_index[0],:,0])[..., None]  # [M_out=62, 3, 1]
            dist = jnp.linalg.norm(f_o[self.edge_index[1],:,0] - f_o[self.edge_index[0],:,0], axis=-1, keepdims=True)
            edge_attr_inter_s = self.dist_rbf(dist, dim=-1)  # [2, 64]
            edge_attr_inter_s = jnp.concatenate((edge_attr_inter_s, self.edge_type), axis=-1)  # [2, 65][..., None]

            f_o = self.embedding_f(f_o) # [N_obj, 3, vd]
            s_o = self.embedding_s(s_o)  # [N=32, H=256]
            for _ in range(self.p_step):
                f_o, s_o = self.object_message_passing(f_o, s_o, self.edge_index, edge_attr_inter_f, edge_attr_inter_s)  # [N_obj, 3, vector_dim], [N_obj, 2vector_dim]

            # f_o = self.embedding_fout(f_o)   # [N, 3, 31 or self.vector_dim]
            # f_o_T = jnp.swapaxes(f_o, -1, -2)
            # f2s = jnp.einsum('bij,bjk->bik', f_o_T, f_o)  # [N, vector_dim, vector_dim]
            # f2s = f2s.reshape(f2s.shape[0], -1)  # [N, vector_dim*vector_dim]
            # F_norm = jnp.linalg.norm(f2s, axis=-1, keepdims=True) + 1.0
            # f2s = jnp.concatenate((f2s, s_o), axis=-1)  # [N, vector_dim*vector_dim+2S+Se]
            # f2s = self.fusion(f2s)/F_norm  # [N, vector_dim]

            outs = self.predictor(s_o.reshape(-1))
            # u = f_o_
        else:
            torso_array = jnp.array(self.torso)
            x_o = x_p[self.torso, :]
            agent_idx, task_idx, edge_index_bi = bipartite_matching(x_o[:self.ctl_num], x_o[self.ctl_num:], self.ctl_num)


            x_m = x_p[self.torso[:self.ctl_num], :]
            x_p = x_p - x_m[agent_idx[self.obj_id]]  # [N, 3]

            x_p = x_p.at[torso_array[self.ctl_num:]].set(x_p[torso_array[self.ctl_num:]]/ (jnp.linalg.norm(x_p[torso_array[self.ctl_num:]], axis=-1, keepdims=True)+1e-6) )

            f_p = jnp.stack((x_p, v_p, a_p), axis=-1)
            f_o = f_p[self.torso, :, :] # [N_obj, 3, 3]
            s_o = h_p[self.torso, :] # [N_obj, H]

            if self.gravity_axis is not None:
                g_o = jnp.zeros_like(f_o)[..., 0]  # [N_obj, 3]
                g_o = g_o.at[..., self.gravity_axis].set(-9.8)
                f_o = jnp.concatenate((f_o, g_o[..., None]), axis=-1)  # [N_obj, 3, 3+1]
                s_o = jnp.concatenate((s_o, x_p[self.torso, self.gravity_axis][..., None]), axis=-1)  # [N_obj, H+1]
            
            edge_attr_inter_f = (f_o[edge_index_bi[1],:,0] - f_o[edge_index_bi[0],:,0])[..., None]  # [M_out=62, 3, 1]
            # dist = jnp.linalg.norm(f_o[edge_index_bi[1],:,0] - f_o[edge_index_bi[0],:,0], axis=-1, keepdims=True)
            # edge_attr_inter_s = self.dist_rbf(dist, dim=-1)  # [2, 64]

            f_o = self.embedding_f(f_o) # [N_obj, 3, vd]
            s_o = self.embedding_s(s_o)  # [N=32, H=256]
            f_o, s_o = self.object_message_passing(f_o, s_o, edge_index_bi, edge_attr_inter_f)#, edge_attr_inter_s)  # [N_obj, 3, vector_dim], [N_obj, 2vector_dim]

            u = self.embedding_u(f_o[:self.ctl_num])

            # f_o = self.embedding_fout(f_o)   # [N, 3, 31 or self.vector_dim]
            # f_o_T = jnp.swapaxes(f_o, -1, -2)
            # f2s = jnp.einsum('bij,bjk->bik', f_o_T, f_o)  # [N, vector_dim, vector_dim]
            # f2s = f2s.reshape(f2s.shape[0], -1)  # [N, vector_dim*vector_dim]
            # F_norm = jnp.linalg.norm(f2s, axis=-1, keepdims=True) + 1.0
            # f2s = jnp.concatenate((f2s, s_o), axis=-1)  # [N, vector_dim*vector_dim+2S+Se]
            # f2s = self.fusion(f2s)/F_norm  # [N, vector_dim]

            mat = construct_3d_basis_from_1_vectors(u[..., 0]) #[2,3,3]
            f_p = jnp.einsum('bij,bjk->bik', mat[agent_idx[self.obj_id]].transpose(0,2,1), f_p)  # [N,  3, K]

            # f_p_ = self.embedding_fp(f_p.reshape(f_p.shape[0],-1)) # [ N, vd]
            # s_p_ = self.embedding_sp(h_p)  # [N=32, H=256]
            ret = jnp.concatenate((f_p.reshape(f_p.shape[0],-1), h_p), axis=-1)  # [3*N=32*3+H] #s_o_[self.obj_id]
            ret_o = ret[self.torso, :]
            ret_o = ret_o[task_idx,:]
            outs = []


            for i in range(self.ctl_num):
                out = jnp.concatenate((ret[self.torso[i] : self.torso[i+1]], ret_o[i][None,...]), axis=0)
                out = jnp.concatenate((out.reshape(-1), s_o[i]))
                out = self.predictor[i](out)
                outs.append(out)

            outs = jnp.concatenate(([out.reshape(2,-1) for out in outs]), axis=-1)  #loc, scale = jnp.split(parameters, 2, axis=-1)

        # ret = self.embedding3(s)
        # ret = self.predictor(ret)

        if self.test:
            return outs.ravel(), jnp.swapaxes(u, -1, -2).ravel()
        else:
            return outs.ravel()
    



def make_sgnn_networks(
        obs_size: int,
        local_state_size: int,
        policy_params_size: int,
        torso: Optional[jnp.ndarray],
        num_node: int,
        ctl_num: int,
        obj_id: Optional[jnp.ndarray],
        edge_index: Optional[jnp.ndarray],
        edge_index_inner: Optional[jnp.ndarray],
        edge_type: Optional[jnp.ndarray],
        ) -> Tuple[networks.FeedForwardNetwork, networks.FeedForwardNetwork]:
    """Creates mlp models for policy and value functions,
    Args:
        policy_params_size: number of params that a policy network should generate
        obs_size: size of an observation
    Returns:
        a model for policy and a model for value function
    """
    
    dummy_obs = jnp.zeros((obs_size))
    print("SGNN_O_MLP_bipair")

    def policy_model_fn():
        module = SGNN(local_state_size = local_state_size, torso = torso, num_node = num_node, ctl_num=ctl_num, edge_index = edge_index, edge_type=edge_type, obj_id = obj_id, gravity_axis = 2, out_dim=policy_params_size)
        # def single_init(rng, single_obs):
        #     return module.init(rng, single_obs)
        # batch_init = vmap(single_init, in_axes=(None, 0))
        def single_apply(param, single_obs):
            return module.apply(param, single_obs)
        batch_apply = vmap(single_apply, in_axes=(None, 0))

        # print(dummy_obs.shape)
        model = networks.FeedForwardNetwork(
            init=lambda rng: module.init(rng, dummy_obs), apply=batch_apply)
        return model

    def value_model_fn():
        module = SGNN(local_state_size = local_state_size, torso = torso, num_node = num_node, ctl_num=ctl_num, edge_index = edge_index, edge_type=edge_type, obj_id = obj_id, gravity_axis = 2, out_dim=1)
        # def single_init(rng, single_obs):
        #     return module.init(rng, single_obs)
        # batch_init = vmap(single_init, in_axes=(None, 0))
        def single_apply(param, single_obs):
            return module.apply(param, single_obs)
        batch_apply = vmap(single_apply, in_axes=(None, 0))

        # print(dummy_obs.shape)
        model = networks.FeedForwardNetwork(
            init=lambda rng: module.init(rng, dummy_obs), apply=batch_apply)
        return model

    return policy_model_fn(), value_model_fn()


import unittest
import numpy as np

class ModelTest(unittest.TestCase):
    def dummy_sample(self, num_node):
        rng = self.key()
        f = e3nn.normal("3x1o", rng, (num_node,))
        s = e3nn.normal("10x0e", rng, (num_node,))
        dummy_obs = jnp.concatenate((f.array, s.array), axis=-1)
        dummy_obs = dummy_obs.ravel()
        return dummy_obs

       

    def key(self):
        return jax.random.PRNGKey(0)

    def assert_equivariant(self, model_apply, num_node, params):
        rng = self.key()

        f = e3nn.normal("3x1o", rng, (num_node,))
        s = e3nn.normal("10x0e", rng, (num_node,))

        def wrapper(f, s):
            dummy_obs = jnp.concatenate((f.array, s.array), axis=-1).ravel()
            x, y = model_apply(params, dummy_obs)
            return e3nn.IrrepsArray(f"{x.shape[0]}x0e", x), e3nn.IrrepsArray(f"{y.shape[0]//3}x1o", y)

        # # random rotation matrix
        # # R = -e3nn.rand_matrix(rng, ()) #reflection
        # R = e3nn.rand_matrix(rng, ())
        # # 假设 A 是你想要检查的矩阵
        # is_orthogonal = jnp.allclose(jnp.dot(R, R.T), jnp.eye(R.shape[0])) and jnp.allclose(jnp.dot(R.T, R), jnp.eye(R.shape[1]))
        # determinant = jnp.linalg.det(R)
        # print(is_orthogonal, determinant)
        # def random_rotation_matrix(rng, shape):
        #     # 生成一个随机矩阵
        #     A = jax.random.normal(rng, shape)
        #     # 使用 QR 分解得到正交矩阵 Q
        #     Q, _ = jnp.linalg.qr(A)
        #     # 计算 Q 的行列式
        #     det = jnp.linalg.det(Q)
        #     # 如果行列式为 -1，我们需要翻转 Q 的一个维度来得到一个旋转矩阵
        #     Q = jnp.where(det < 0, -Q, Q)
        #     return Q
        # 使用函数
        # R = random_rotation_matrix(rng, (3, 3))

        theta = random.uniform(rng, (), minval=0, maxval=2*jnp.pi)
        R = jnp.array([[jnp.cos(theta), -jnp.sin(theta), 0],
                    [jnp.sin(theta),  jnp.cos(theta), 0],
                    [0,               0,              1]])

        out1x, out1y = wrapper(f.transform_by_matrix(R), s.transform_by_matrix(R))
        out2x, out2y = wrapper(f, s)
        out2x = out2x.transform_by_matrix(R)
        out2y = out2y.transform_by_matrix(R)

        def assert_(x, y):
            self.assertTrue(
                np.isclose(x, y, atol=1e-5, rtol=1e-5).all(), "Not equivariant!"
            )

        jax.tree_util.tree_map(assert_, out1y, out2y)
        jax.tree_util.tree_map(assert_, out1x, out2x)
        

    def test_sgnn(self, local_state_size, torso, num_node, ctl_num, edge_index, edge_index_inner, edge_type, obj_id, policy_params_size):
        dummy_obs = self.dummy_sample(num_node)

        module = SGNN(local_state_size = local_state_size, torso = torso, num_node = num_node, ctl_num=ctl_num, edge_index = edge_index, edge_type=edge_type, obj_id = obj_id, gravity_axis = 2, out_dim=policy_params_size, test=True)
        model = networks.FeedForwardNetwork(
            init=lambda rng: module.init(rng, dummy_obs), apply=module.apply)
        
        params = model.init(self.key())
        num_params = sum(jnp.prod(jnp.array(param.shape)) for param in jax.tree_util.tree_leaves(params['params']))
        print(f"Total number of parameters: {num_params}")

        self.assert_equivariant(model.apply, num_node, params)

