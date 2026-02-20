
##### Helper funtions and classes for graph augmentation #####


import torch
from copy import deepcopy

import numpy as np
from torch_geometric.utils import remove_self_loops
from torch_geometric.data import Batch as GraphBatch
from torch_geometric.nn import global_mean_pool

# from .math3d import rot3d_matrix

def rot3d_matrix(angles : torch.Tensor) -> torch.tensor:
    """
    Compute the rotation matrix from Euler angles for rotations around the x, y, and z axes.
    Angles are expected in radians.

    Args:
    angles (torch.Tensor): shape (N,3) Rotation angles around the x, y, and z axes. in radians [0,2*pi]

    Returns:
    torch.Tensor: The Nx3x3 rotation matrix.
    """
    assert angles.shape[-1] == 3, "angles have spacial dimension of 3d "
    if angles.dim() == 1:
        angles = angles.view(1, 3) # add batch dimension
    assert angles.dim() == 2, "angles should be of shape (3,) or (N, 3)"

    # Precompute cosines and sines of the angles
    cos_angles = torch.cos(angles)
    sin_angles = torch.sin(angles)

    cos_x = cos_angles[:, 0]
    sin_x = sin_angles[:, 0]
    cos_y = cos_angles[:, 1]
    sin_y = sin_angles[:, 1]
    cos_z = cos_angles[:, 2]
    sin_z = sin_angles[:, 2]

    # Construct the rotation matrix
    x_row = torch.stack([cos_z * cos_y, cos_z * sin_y * sin_x - sin_z * cos_x, cos_z * sin_y * cos_x + sin_z * sin_x], dim=-1)
    y_row = torch.stack([sin_z * cos_y, sin_z * sin_y * sin_x + cos_z * cos_x, sin_z * sin_y * cos_x - cos_z * sin_x], dim=-1)
    z_row = torch.stack([-sin_y, cos_y * sin_x, cos_y * cos_x], dim=-1)

    rotation_matrix = torch.stack([x_row, y_row, z_row], dim=1)
    return rotation_matrix


def batch_com(pos, batch, nodewise=False):
    '''
    compute the center of mass for each graph in the batch
    Args:
        pos (torch.Tensor): The positions of the nodes in the batch.
        batch (torch.Tensor): The batch vector indicating the graph each node belongs to.
        if nodewise is True, reproject out the center of mass for each graph in shape of the position nodes
    Returns:
        torch.Tensor: The center of mass for each graph in the batch.
    '''
    # Compute the center of mass for each graph in the batch
    com = global_mean_pool(pos, batch) # get the center of mass for each graph in the batch
    if nodewise:
        # Reproject out the center of mass for each graph in the shape of the position nodes
        com = com[batch]
    return com

def batch_center(pos, batch):
    '''
    recenter the positions of each graph in the batch such that the center of mass is at the origin
    '''
    graph_com = global_mean_pool(pos, batch) # get the center of mass for each graph in the batch
    pos = pos - graph_com[batch] # subtract the center of mass from each node position
    return pos

def graph_meanvar_pool(x, batch):
    '''
    graph pooling operation that computes the mean and std of the node feature x
    '''
    mean_x = global_mean_pool(x, batch) #[batch]
    var_x = global_mean_pool((x - mean_x[batch])**2, batch) #[batch]
    return mean_x, var_x

def graph_poolsample(x, batch):
    '''
    use the mean and std of the node feature x to sample x values
    returns a sample the same shape as x
    '''
    mean_x, var_x = graph_meanvar_pool(x, batch)
    std_x = torch.sqrt(var_x)
    sample_x = mean_x + std_x * torch.randn_like(x)
    return sample_x


def batch_shear_from_plane_direction(a, n, m=1.0):
    """
    a: (B, 3) batch of unit vectors, shear direction lying in the plane
    n: (B, 3) batch of unit normals to the invariant plane (a·n == 0)
    m: scalar shear magnitude
    returns (B, 3, 3) batch of shear matrices S = I + m * a n^T
    """
    a = a / torch.linalg.norm(a, dim=-1, keepdim=True)
    n = n / torch.linalg.norm(n, dim=-1, keepdim=True)
    # (optional) enforce orthogonality
    n = n - (torch.sum(n * a, dim=-1, keepdim=True)) * a
    n = n / torch.linalg.norm(n, dim=-1, keepdim=True)
    # I = torch.eye(3, dtype=a.dtype, device=a.device).unsqueeze(0).expand(a.shape[0], -1, -1)
    return m * a[:, :, None] @ n[:, None, :]

def batch_rand_shear_matrix(batch_size, device):
    ''' 
    return a batch shear-transformed position updates 
        del_x = batch_rand_shear_update(pos_stack)

    '''
    B = batch_size
    a = torch.randn(B, 3)
    n = torch.randn(B, 3)
    shear_matrices = batch_shear_from_plane_direction(a, n)
    return shear_matrices.to(device)

def get_rand_shear(scale, pos, batch_idx, center_pos=False):
    '''
    get changes to positions to apply a shear
    new_pos = pos + pos_update

    args:
        scale (float): the magnitude of the shear
        pos (tensor: N,3): node positions from the batch
        batch_idx (tensor: N): the batch indices for each node
        center_pos (bool): whether to center the positions before applying the shear
    returns:
        pos_update (tensor: N, 3): the updated positions after applying the shear
        graph_shear (tensor: G, 3, 3): the shear matrices for each graph

    '''

    num_graphs = batch_idx.max().item() + 1
    graph_shear = batch_rand_shear_matrix(num_graphs, pos.device)

    node_shear = graph_shear[batch_idx]

    if center_pos:
        pos_update = (pos - batch_com(pos, batch_idx)[batch_idx]).unsqueeze(1) @ node_shear.transpose(1, 2)
    else:
        #assume each mol-graph is centered
        pos_update = pos.unsqueeze(1) @ node_shear.transpose(1, 2)
        # pos_update = torch.einsum('nd,n')

    pos_update = pos_update.squeeze(1) * scale  # Scale the shear transformation

    return pos_update, graph_shear




def batch_augment(batch,
                  rand_translate,
                  rand_rotate, 
                  pos_noise_scale=0.0,
                  pos_noise_ratio=0.0, 
                  pos_shear_scale=0.0, 
                  joint_mask = False, # mask z and pos jointly or independently,
                  z_mask_ratio=0.0, # if joint masking uses only noise ratio
                  z_mask_id = 0,
                  z_id_ignore_mask = [], # ids of z tokens that should never be masked, ie the cls token
                  center=True,
                  recenter=False,
                  device=None,
                  noise_type='gaussian', # or 'uniform'
                  ):
    
    #TODO: add option to turn off position masking 
    #TODO: or make position and id masking ratios independent

    '''
    Augment a batch of graphs by applying random translations, rotations, and positional noise.

    input batch must have keys 'pos', 'z', and 'batch'

    Args:
        batch (torch_geometric.data.Batch): The batch of graphs to augment.
        rand_translate (float): Scale in angstroms of the random translation (the standard deviation).
        rand_rotate (float): Scale in radians of the random rotation (the standard deviation).
        pos_noise (float): Scale in angstroms of the random jitter applied to atom positions (the standard deviation).
        mask_ratio (float): The ratio of nodes to mask. If 0, no masking is applied.
        joint_masking (bool): Whether to mask z and pos jointly or independently.
        center (bool): Whether to center the positions before applying augmentations.
        recenter (bool): Whether to recenter the positions after applying positional noise.
        device (torch.device, optional): The device to use for the batch. If None, use the device of the batch.
        return_aug (bool): Whether to return the augmentations applied.
    Returns:
        batch (torch_geometric.data.Batch): The augmented batch of graphs.
        aug_dict (dict): A dictionary containing the augmentations applied (if return_aug is True).

    '''
    
    assert isinstance(batch, GraphBatch), "batch should be a torch_geometric.data.Batch object"
    batch = deepcopy(batch) # make a copy of the batch to avoid modifying the original batch

    if device is None:
        device = batch.pos.device
    else:
        batch = batch.to(device)

    batch_size = batch.num_graphs# or len(batch) or len(batch.ptr)
    N_nodes = batch.num_nodes#len(batch.batch) # total number of nodes in the batch
    aug_dict = {} # recorde augmentations applied
   
    def center_positions(pos, batch):
        graph_com = global_mean_pool(pos, batch) # get the center of mass for each graph in the batch
        pos = pos - graph_com[batch] # subtract the center of mass from each node position
        return pos

    #### centering positions
    if center:
        ### centering before other augmentations prevents samples from being too far out of distribution
        batch.pos = center_positions(batch.pos, batch.batch)

    #### Apply random rotation and translation - assumed to be NON-corrupting
    if rand_rotate > 0:
        #### Add Rotational Noise - add before translational noise
        # create random rotation angles sampled from a uniform distribution
        random_angles = torch.rand(batch_size, 3, device=device)*rand_rotate
        rotmat = rot3d_matrix(random_angles) # get rotation matrix from angles
        # apply a random rotation to each graph in the batch
        batch.pos = torch.einsum('ijk,ik->ij', rotmat[batch.batch], batch.pos) # apply the rotation matrix to the positions
        aug_dict['rotate_angles'] = random_angles

    if rand_translate > 0:
        #### Add Translational Noise
        #create random positional jitter sampled from a normal distribution
        ### Gausanian noise
        # random_shift = torch.randn(batch_size, 3, device=device)*rand_translate
        
        ### Uniform noise
        random_shift = torch.rand(batch_size, 3, device=device)*rand_translate
        
        # apply a random shift to each graph in the batch
        batch.pos = batch.pos + random_shift[batch.batch] # get the random shift for each graph in the batch
        aug_dict['translate_noise'] = random_shift
    ######

    #record uncorrupted (but roto/trans augmented) targets
    aug_dict['true_pos'] = torch.clone(batch.pos) 
    aug_dict['true_z'] = torch.clone(batch.z)

    masked_batch = deepcopy(batch) # create a copy of the batch
    aug_dict['pos_noise_ratio'] = pos_noise_ratio
    
    if pos_shear_scale > 0.0:
        shear_update, g_shears = get_rand_shear(
                    scale=pos_shear_scale,
                    pos=batch.pos,
                    batch_idx=batch.batch,
                    center_pos=True)
        aug_dict['graph_shear'] = g_shears #[G, 3,3]
        aug_dict['shear_delta'] = shear_update #[N, 3]
        masked_batch.pos = masked_batch.pos + shear_update

    #corrupt positions with noise
    if pos_noise_ratio > 0 and pos_noise_scale > 0:    
        #### Add per note positional noise
        #create random positional jitter sampled from a normal distribution
        assert noise_type == 'gaussian', "noise_type must be 'uniform' or 'gaussian'"
        
        if noise_type == 'uniform':
            pos_noise = torch.rand(N_nodes, 3, device=device)*pos_noise_scale
        else: # default to gaussian noise
            pos_noise = torch.randn(N_nodes, 3, device=device)*pos_noise_scale

        #center noise around 0 for each graph in the batch
        noise_com = global_mean_pool(pos_noise, batch.batch)
        pos_noise = pos_noise - noise_com[batch.batch]

        #create a noise mask based on the pos_noise_ratio
        noise_mask = torch.rand(N_nodes, device=device) < pos_noise_ratio # 1s where noise is applied, 0s where no noise is applied
        noise_mask = noise_mask.to(device) # move to the same device as the batch 

        #zero out the noise where the mask is False
        pos_noise = pos_noise * noise_mask.unsqueeze(-1) # unsqueeze to match the shape of pos_noise

        #apply noise to positions
        masked_batch.pos = masked_batch.pos + pos_noise # add the random jitter to the positions
        
        aug_dict['pos_noise'] = pos_noise
        aug_dict['target_noise'] = pos_noise #/pos_noise_scale # unscale the target noise for prediction

        aug_dict['mask_pos'] = noise_mask # record the mask for positions
       
        #### recentering positions after positional noise 
        if recenter: # noise addition slightly shifts the center of mass
            masked_batch.pos = center_positions(masked_batch.pos, masked_batch.batch)
            if rand_translate > 0:
                # if we added translational noise, we need to recenter again
                masked_batch.pos += aug_dict['translate_noise'][masked_batch.batch] # add the translational noise back to the positions
    else:
        aug_dict['mask_pos'] = torch.zeros(masked_batch.num_nodes, device=masked_batch.pos.device, dtype=torch.bool)

    if joint_mask:
        # if joint masking is used, we mask z and pos jointly
        z_mask_ratio = pos_noise_ratio

    # if pos_mask_ratio > 0:
    #     #create nodewise mask [B, N]
    #     mask_pos = torch.rand(batch.num_nodes) < pos_mask_ratio
    #     mask_pos = mask_pos.to(batch.pos.device) # move to the same device as the batch
    #     aug_dict['mask_pos'] = mask_pos

    #     #record uncorrupted targets
    #     pos = masked_batch.pos 
        
    #     #sample random positions based on each graphs mean and variance
    #     mean_pos = global_mean_pool(pos, batch.batch)[batch.batch]
    #     var_pos = global_mean_pool((pos - mean_pos)**2, batch.batch)[batch.batch]
    #     std_pos = torch.sqrt(var_pos)
    #     sample_pos = mean_pos + std_pos * torch.randn_like(pos) # draw random positions from normal distribution
    #     masked_batch.pos[mask_pos] = sample_pos[mask_pos]
    # else:
    #     aug_dict['mask_pos'] = torch.zeros(batch.num_nodes, device=batch.pos.device, dtype=torch.bool)

    aug_dict['z_mask_ratio'] = z_mask_ratio
    if z_mask_ratio > 0:
        if joint_mask:
            mask_z = aug_dict['mask_pos'] # use the same mask as for positions
        else:
            mask_z = torch.rand(masked_batch.num_nodes) < z_mask_ratio
            mask_z = mask_z.to(masked_batch.z.device)
        
        #exclude z tokens that should not be masked
        if z_id_ignore_mask is not None:
            if isinstance(z_id_ignore_mask, int):
                z_id_ignore_mask = [z_id_ignore_mask]
            
            if len(z_id_ignore_mask) > 0:
                mask_z[torch.isin(masked_batch.z, torch.tensor(z_id_ignore_mask, device=masked_batch.z.device))] = False
                aug_dict['mask_z_ignore'] = z_id_ignore_mask

        aug_dict['mask_z'] = mask_z
        masked_batch.z[mask_z] = z_mask_id # replace atom mask token
    else:
        aug_dict['mask_z'] = torch.zeros(batch.num_nodes, device=batch.z.device, dtype=torch.bool)
    
    aug_dict['mask_any'] = aug_dict['mask_pos'] | aug_dict['mask_z'] # combine the masks

    return batch, masked_batch, aug_dict

    # if mask_ratio > 0:
    #     #TODO: 
    #     # Explore independant z and pos masking vs joint masking
    #     # this would use 2 masks

    #     #create nodewise mask [B, N]
    #     mask_pos = torch.rand(batch.num_nodes) < mask_ratio
    #     mask_pos = mask_pos.to(batch.pos.device) # move to the same device as the batch
    #     if joint_masking:
    #         mask_z = mask_pos
    #     else:
    #         mask_z = torch.rand(batch.num_nodes) < mask_ratio
    #         mask_z = mask_z.to(batch.z.device)

    #     aug_dict['mask_pos'] = mask_pos
    #     aug_dict['mask_z'] = mask_z

    #     mask_any = mask_pos | mask_z # combine the masks
    #     aug_dict['mask'] = mask_any
        
    #     # mask_ids = torch.rand(batch.num_nodes) < mask_ratio
    #     # mask_ids = mask_ids.to(batch.pos.device) # move to the same device as the batch
    #     # aug_dict['mask'] = mask_ids

    #     #record uncorrupted targets
    #     pos = torch.clone(batch.pos) # original positions
        
    #     #sample random positions based on each graphs mean and variance
    #     mean_pos = global_mean_pool(pos, batch.batch)[batch.batch]
    #     var_pos = global_mean_pool((pos - mean_pos)**2, batch.batch)[batch.batch]
    #     std_pos = torch.sqrt(var_pos)
    #     sample_pos = mean_pos + std_pos * torch.randn_like(pos) # draw random positions from normal distribution

    #     #apply masking
    #     masked_batch = deepcopy(batch) # create a copy of the batch 
    #     masked_batch.pos[mask_pos] = sample_pos[mask_pos] # replace node positions with random noise positions
    #     masked_batch.z[mask_z] = z_mask_id # replace atom mask token

    #     return batch, masked_batch, aug_dict
    # else:
    #     aug_dict['mask'] = torch.zeros(batch.num_nodes, device=batch.z.device, dtype=torch.bool)
    #     aug_dict['mask_pos'] = torch.zeros(batch.num_nodes, device=batch.pos.device, dtype=torch.bool)
    #     aug_dict['mask_z'] = torch.zeros(batch.num_nodes, device=batch.z.device, dtype=torch.bool)
    #     return batch, batch, aug_dict
    

class GraphAugmenter:
    '''
    A class to augment a batch of graphs by applying random translations, rotations, and positional noise.
    Args:
        rand_translate (float): Scale in angstroms of the random translation (the standard deviation).
        rand_rotate (float): Scale in radians of the random rotation (the standard deviation).
        pos_noise (float): Scale in angstroms of the random jitter applied to atom positions (the standard deviation).
        center (bool): Whether to center the positions before applying augmentations.
        recenter (bool): Whether to recenter the positions after applying positional noise.
        device (torch.device, optional): The device to use for the batch. If None, use the device of the batch.
    '''
    def __init__(self, 
                 rand_translate=0.0,
                 rand_rotate=0.0,
                 pos_noise_scale=0.0,
                 pos_noise_ratio=0.0, 
                 pos_shear_scale=0.0,
                 noise_type='gaussian', # or 'uniform'
                 joint_mask = False, # whether to mask noised positions in z
                 z_mask_ratio=0.0,
                 z_mask_id=0, # id of the mask token for z
                 z_id_ignore_mask = [],
                 center=True,
                 recenter=False,
                 device=None):
        
        self.rand_translate = rand_translate
        self.rand_rotate = rand_rotate

        self.pos_noise_scale = pos_noise_scale
        self.pos_noise_ratio = pos_noise_ratio # ratio of nodes to mask in positions
        self.noise_type = noise_type
        assert noise_type in ['gaussian', 'uniform'], "noise_type must be 'gaussian' or 'uniform'"

        self.pos_shear_scale = pos_shear_scale # scale of the shear transformation

        self.joint_mask = joint_mask
        self.z_mask_ratio = z_mask_ratio if not joint_mask else pos_noise_ratio # ratio of nodes to mask in z
        
        self.z_mask_id = z_mask_id # id of the mask token for z
        self.z_id_ignore_mask = z_id_ignore_mask # ids of z tokens that should never be masked, ie the cls token

        self.center = center
        self.recenter = recenter
        self.device = device # default device to use
    
    def augment(self, batch, device=None,):
        if device is None:
            device = self.device
            
        return batch_augment(batch,
                             self.rand_translate,
                             self.rand_rotate,

                             pos_noise_scale=self.pos_noise_scale,
                             pos_noise_ratio=self.pos_noise_ratio, 
                             pos_shear_scale=self.pos_shear_scale, 
                             noise_type=self.noise_type,
                             joint_mask=self.joint_mask,

                             z_mask_ratio=self.z_mask_ratio,
                             z_id_ignore_mask=self.z_id_ignore_mask,
                             z_mask_id=self.z_mask_id,

                             center=self.center,
                             recenter=self.recenter,
                             device=device,
                            )
    
    def __call__(self, batch, device=None):
        return self.augment(batch, device=device)
    
    def __repr__(self):
        return f"GraphAugmenter(rand_translate={self.rand_translate}, rand_rotate={self.rand_rotate}, pos_noise={self.pos_noise}, center={self.center}, recenter={self.recenter})"
    

######## graph-edge formulation functions ########

@torch.no_grad()
def distance_graph(pos, cutoff, include_self_loops=False, 
                   add_cls_node=False, 
                   cls_issource = True, # whether to create edges from cls node to all other nodes
                   cls_istarget = True, # whether to create edges from all other nodes to cls node
                   cls_source_dist = 0.0, # distance b/w cls node as source and all other nodes as target 
                   cls_target_dist = 0.0, # distance b/w all other nodes as source and cls node as target
                   clsgraph_issource = True, # whether to create edges cls->nodes in cls graph
                   clsgraph_istarget = True, # whether to create edges nodes->cls in cls graph
                   ):
    '''
    Create a distance graph from the positions of nodes in a graph data object.
    
    args:   
        pos (torch.Tensor): Node positions of shape (N, 3)
        cutoff (float): Distance threshold for edge creation.
            Edges will be created between nodes that are within this distance.
            if set to None, a fully connected graph will be created.
        include_self_loops (bool): Whether to include self-loops in the graph.
            If True, edges from a node to itself will be included. this includes the cls node as well.  
        add_cls_node (bool): Whether to add a class node or not.
            if a cls node is added, it will be added at the center of mass (mean of all positions)
            in the zeroth position of pos.
        cls_issource (bool): Whether the class node is a source for messages to other nodes in the main graph.
            if True, edges are created from class node to other nodes in the main graph, 
            ie, messages will sent from class node to other nodes. nodes are updated by cls node in the main graph.
        cls_istarget (bool): Whether the class node is a target for messages from other nodes.
            if True, edges are created from other nodes to class node in the main graph, 
            ie, messages will sent from other nodes to class node. cls node is updated by other nodes.
        cls_source_dist (float): Constant Distance of class node to all other nodes as source.
            If set to None, global distance to each node will be used. cls edges are subject to cutoff distance.
        cls_target_dist (float): Constant Distance of all other nodes to class node as target.
            If set to None, global distance to each node will be used. cls edges are subject to cutoff distance.
        clsgraph_issource (bool): Whether the class node is a source for messages to other nodes in the cls graph.
        clsgraph_istarget (bool): Whether the class node is a target for messages from other nodes in the cls graph.
        
    returns:
        pos (torch.Tensor): Updated node positions including class node if added (N,3 or N+1,3)
        edge_dist (torch.Tensor): Edge distances of shape (E,1)
        edge_index (torch.Tensor): Edge index of the graph (2, E)
        cls_edge_dist (torch.Tensor): Edge distances for class node connections (E_cls, 1)
            includes edges distances between cls node and all other nodes
        cls_edge_index (torch.Tensor): Edge index for class node connections (2, E_cls)
            the cls graph includes edges between the cls node and all other nodes
            construction does not use the cutoff distance. 
    
    '''
    device = pos.device
    if add_cls_node:
        #positions must be of shape (N, 3) or (...,N,3)
        com = pos.mean(dim=-2, keepdim=True)  # Compute center of mass
        cls_node = com
        pos = torch.cat([cls_node, pos], dim=0)  # Concatenate class node to positions

    N = pos.size(-2) #number of nodes

    # Create edge index based on distance threshold
    if cutoff is None:
        cutoff = 1  # Fully connected graph if cutoff is None
        dist_matrix = torch.zeros((N, N), device=device)  # Fully connected graph
    else:
        # Compute pairwise distances
        dist_matrix = torch.cdist(pos, pos)

    # Add class node connections
    if add_cls_node:
        #Alter distance matrix to set cls node distances
        if cls_issource: # if nodes recieved message from cls node
            if cls_source_dist is not None:
                dist_matrix[0, 1:] = cls_source_dist
        else: # if cls node is not a source for messages passing to other nodes
            dist_matrix[0, 1:] = float('inf')  # No connections from cls node to other nodes
        if cls_istarget: # if nodes send message to cls node
            if cls_target_dist is not None:
                dist_matrix[1:, 0] = cls_target_dist
        else: # if cls node is not a target for messages passing from other nodes
            dist_matrix[1:, 0] = float('inf')  # No connections from other nodes to cls node
    
    #mask diagoonal elements with inf to remove self-loops
    if not include_self_loops:
        dist_matrix.fill_diagonal_(float('inf'))

    row, col = torch.where(dist_matrix < cutoff)

    #create edge index for whole graph
    edge_dist = dist_matrix[row, col].unsqueeze(-1)  # Get distances for edges
    edge_index = torch.stack([row, col], dim=0)

    if add_cls_node:
        cls_node_idx = 0
        other_nodes_idx = np.arange(1, N, dtype=np.long)  # Other nodes indices
        edges = []
        if include_self_loops:  # Add self-loops for class node if required
            edges.append((cls_node_idx, cls_node_idx))  # Class node self-loop
        if clsgraph_issource:  # Class node as source
            edges.extend([(cls_node_idx, idx) for idx in other_nodes_idx])  # Class node to other nodes
        if clsgraph_istarget:  # Class node as target
            edges.extend([(idx, cls_node_idx) for idx in other_nodes_idx])  # Other nodes to class node
        cls_edge_index = torch.tensor(edges, dtype=torch.long, device=device).t()  # Create edge index for class node connections
        cls_edge_dist = dist_matrix[cls_edge_index[0], cls_edge_index[1]].unsqueeze(-1)  # Get distances for cls edges
        
        return pos, edge_dist, edge_index, cls_edge_dist, cls_edge_index
    else:
        return pos, edge_dist, edge_index, None, None
    

class DistanceGraph(object):
    '''
    a class wrapper for the distance graph function
    creates a distance graph from the positions of nodes in a graph data object.
    also includes the option to add a class node (cls node) to the graph.

    '''
    def __init__(self, 
                 cutoff=10.0, 
                 include_self_loops=False, 
                 add_cls_node=False, 
                 cls_node_id = 0, # id of the class token, if added
                 cls_issource=True, 
                 cls_istarget=True, 
                 cls_source_dist=0.0, 
                 cls_target_dist=0.0,
                 clsgraph_issource=True,
                 clsgraph_istarget=True
                 ):
        self.cutoff = cutoff
        self.include_self_loops = include_self_loops
        self.add_cls_node = add_cls_node
        self.cls_node_id = cls_node_id  # id of the class token, if added
        self.cls_issource = cls_issource
        self.cls_istarget = cls_istarget
        self.cls_source_dist = cls_source_dist
        self.cls_target_dist = cls_target_dist
        self.clsgraph_issource = clsgraph_issource
        self.clsgraph_istarget = clsgraph_istarget
    
    def __call__(self, data):
        """
        Create a distance graph from the positions of nodes in a graph data object.
        Args:
            data (torch_geometric.data.Data): The graph data object containing node positions.
        Returns:
            torch_geometric.data.Data: The updated graph data object with edge index and edge attributes.
        """
        pos = data.pos
        pos, edge_dist, edge_index, cls_edge_dist, cls_edge_index = distance_graph(
            pos,
            self.cutoff,
            include_self_loops=self.include_self_loops,
            add_cls_node=self.add_cls_node,
            cls_issource=self.cls_issource,
            cls_istarget=self.cls_istarget,
            cls_source_dist=self.cls_source_dist,
            cls_target_dist=self.cls_target_dist,
            clsgraph_issource=self.clsgraph_issource,
            clsgraph_istarget=self.clsgraph_istarget
        )
        
        #update data object 
        data.pos = pos
        data.edge_index = edge_index
        data.edge_dist = edge_dist
        
        if self.add_cls_node:
            data.cls_edge_index = cls_edge_index
            data.cls_edge_dist = cls_edge_dist
            # Add class token to z
            new_z = torch.cat([torch.tensor([self.cls_node_id], device=data.z.device), data.z], dim=0)  
            data.z = new_z
            #check if data contains key 'x
            if hasattr(data, 'x'):
                # Add class token to x
                new_x = torch.cat([data.x.new_zeros(1, data.x.size(-1)), data.x], dim=0)
                data.x = new_x

        return data



class CompleteGraph(object):
    """
    This transform adds all pairwise edges into the edge index per data sample, 
    then removes self loops, i.e. it builds a fully connected or complete graph

    simple and faster than using the distance graph function with a large cutoff
    """
    def __call__(self, data):
        device = data.edge_index.device

        row = torch.arange(data.num_nodes, dtype=torch.long, device=device)
        col = torch.arange(data.num_nodes, dtype=torch.long, device=device)

        row = row.view(-1, 1).repeat(1, data.num_nodes).view(-1)
        col = col.repeat(data.num_nodes)
        edge_index = torch.stack([row, col], dim=0)

        edge_attr = None
        if data.edge_attr is not None:
            idx = data.edge_index[0] * data.num_nodes + data.edge_index[1]
            size = list(data.edge_attr.size())
            size[0] = data.num_nodes * data.num_nodes
            edge_attr = data.edge_attr.new_zeros(size)
            edge_attr[idx] = data.edge_attr

        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
        data.edge_attr = edge_attr
        data.edge_index = edge_index

        return data
    