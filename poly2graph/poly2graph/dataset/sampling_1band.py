import numpy as np
import itertools, h5py, pickle, os
from .. import spectral_graph, auto_Emaxes, contract_close_nodes

from numpy.typing import ArrayLike
from typing import Optional, Union, List

### Uniform (skewed) Data Sampling ###

def dim_samples_step(samples_per_dim=7, dim=4, c_max=1.2):
    values = list(np.linspace(-c_max, c_max, samples_per_dim))
    combinations = list(itertools.product(values, repeat=dim))
    return combinations

def generate_full_coefficients(free_coeff: ArrayLike) -> ArrayLike:
    # convert free coefficients to full coefficients
    free_coeff = np.asarray(free_coeff)
    dim = free_coeff.shape[1]
    if dim == 4:
        coeff_matrix = np.array([(1, x[0], x[1], 0, x[2], x[3], 1) for x in free_coeff])
    elif dim == 6:
        coeff_matrix = np.array([(1, x[0], x[1], x[2], 0, x[3], x[4], x[5], 1) for x in free_coeff])
    elif dim == 7:
        coeff_matrix = np.array([(x[0], x[1], x[2], x[3], 0, x[4], x[5], x[6], 1) for x in free_coeff])
        coeff_matrix = coeff_matrix[np.abs(coeff_matrix[:, :4]).sum(-1) > 0]
    else:
        raise ValueError(f"free_coeff with dim={dim} is not implemented. Only dim=4, 6, 7 are supported.")
    return coeff_matrix


### Balanced Data Sampling ###

def class_samples_rand(binary_mask, num_samples, c_max=1):
    num_non_zero = np.sum(binary_mask)
    samples = np.random.uniform(-c_max, c_max, (num_samples, num_non_zero))
    vectors = np.zeros((num_samples, len(binary_mask)))
    vectors[:, binary_mask == 1] = samples
    return vectors

def class_samples_step(binary_mask, num_samples, c_max=1):
    num_non_zero = np.sum(binary_mask)
    num_per_class = np.round(num_samples**(1/num_non_zero)).astype(int)
    samp_col = np.linspace(-c_max, c_max, num_per_class)
    samples = np.array(list(itertools.product(samp_col, repeat=num_non_zero)))
    vectors = np.zeros((samples.shape[0], len(binary_mask)))
    vectors[:, binary_mask == 1] = samples
    return vectors

def generate_coefficients_balanced(samples_per_class=2000, dim=4, c_max=1.2, method='rand'):
    indices = []
    for k in range(1, dim+1):  # from 1 to 6 non-zero elements
        indices.extend(itertools.combinations(range(dim), k))
    
    if method == 'rand': class_gen = class_samples_rand
    if method == 'step': class_gen = class_samples_step

    all_vectors = []
    for idx_tuple in indices:
        binary_mask = np.zeros(dim, dtype=int)
        binary_mask[list(idx_tuple)] = 1
        class_vectors = class_gen(binary_mask, samples_per_class, c_max)
        all_vectors.append(class_vectors)
    
    vectors = np.vstack(all_vectors)
    if method == 'rand': vectors = np.vstack([vectors, np.zeros(dim)])

    return np.unique(vectors, axis=0)



### Generate Dataset ###

def generate_dataset(
    file_name_prefix: str,
    coeff_matrix: Union[ArrayLike, List],
    labels: Union[ArrayLike, List],
    Elen: int,
    contract_threshold: Optional[int]=None,
    num_partition: Optional[int]=1,
    auto_Emaxes_kwargs: Optional[dict]={}
) -> None:
    
    ### convert labels to coefficients
    # labels = np.asarray(labels)
    # coeff_matrix = np.asarray(coeff_matrix)
    assert len(coeff_matrix) == len(labels), "Number of coefficients and labels do not match."

    # Partition the coeff_matrix
    total_coeff = len(coeff_matrix)
    partition_size = total_coeff // num_partition
    
    for partition in range(num_partition):
        start_index = partition * partition_size
        if partition == num_partition - 1:  # Handle the last partition
            end_index = total_coeff
        else:
            end_index = (partition + 1) * partition_size

        partition_coeff_matrix = coeff_matrix[start_index:end_index]
        
        # Prepare data structures
        graphs = []
        for c in partition_coeff_matrix:
            Eauto = auto_Emaxes(c, **auto_Emaxes_kwargs)
            graph = spectral_graph(c, Emax=Eauto, Elen=Elen, 
                              contract_threshold=contract_threshold,
                              Potential_feature=True,
                              DOS_feature=True,
                              s2g_kwargs={},
                              PosGoL_kwargs={'ksizes': [11]}) 
            graphs.append(graph)
            print(f'Partition {partition + 1} / {num_partition}: {len(graphs)} / {end_index - start_index}', end='\r')
        
        # Serialize graphs using pickle
        serialized_graphs = [pickle.dumps(graph) for graph in graphs]
        
        # Save the subset of the dataset
        file_name = f"{file_name_prefix}_part_{partition + 1}.h5"
        with h5py.File(file_name, 'w') as f:
            print(f'Saving partition {partition + 1} / {num_partition} (graphs) ...'+' '*20, end='\r')
            f.create_dataset('graphs', data=np.string_(serialized_graphs))
            print(f'Saving partition {partition + 1} / {num_partition} (labels) ...'+' '*20, end='\r')
            f.create_dataset('labels', data=labels[start_index:end_index])
        print(f'Partition {partition + 1} / {num_partition} Done!'+ ' ' * 20)


def load_dataset(file_name_prefix, num_partition=None):
    if num_partition == None:
        if file_name_prefix[-3:] != '.h5':
            file_name = f"{file_name_prefix}.h5"
        else: file_name = file_name_prefix
        
        with h5py.File(file_name, 'r') as f:
            serialized_graphs = f['graphs'][:]
            labels = f['labels'][:]
        
        graphs = [pickle.loads(graph.tobytes()) for graph in serialized_graphs]
        return graphs, labels

    else:
        all_graphs = []
        all_labels = []
        for partition_index in range(1, num_partition + 1):
            partition_file_name = f"{file_name_prefix}_part_{partition_index}.h5"
            with h5py.File(partition_file_name, 'r') as f:
                serialized_graphs = f['graphs'][:]
                labels = f['labels'][:]
            
            graphs = [pickle.loads(graph.tobytes()) for graph in serialized_graphs]
            all_graphs.extend(graphs)
            all_labels.extend(labels)
        
        return all_graphs, all_labels


if __name__ == '__main__':
    if not os.path.exists('./Datasets'):
        os.makedirs('./Datasets')
    # generate_dataset('./Datasets/dataset_graph_dim6', samples_per_dim=7, dim=6, c_max=1.2, Elen=900, num_partition=10)
    free_coeff = dim_samples_step(samples_per_dim=7, dim=6, c_max=1.2)
    coeff = generate_full_coefficients(free_coeff)
    generate_dataset('./Datasets/dataset_graph_dim6', coeff, Elen=512, contract_threshold=14, num_partition=10)





# # Generate to a single file
# def generate_dataset_graph(file_name, samples_per_dim=7, dim=4, c_max=1.2, Elen=256):
#     # Generate all combinations of coefficients
#     values = list(np.linspace(-c_max, c_max, samples_per_dim))
#     combinations = list(itertools.product(values, repeat=dim))
    
#     # Prepare data structures
#     graphs = []
#     labels = []
    
#     for comb in combinations:
#         if dim == 4:
#             c = [1, comb[0], comb[1], 0, comb[2], comb[3], 1]
#         if dim == 6:
#             c = [1, comb[0], comb[1], comb[2], 0, comb[3], comb[4], comb[5], 1]
#         Eauto = auto_Emaxes(c, N=40, pbc=False)
#         graph = spectral_graph(c, Emax=Eauto, Elen=Elen)
#         graphs.append(graph)
#         labels.append(comb)
    
#     labels = np.array(labels)
    
#     # Serialize graphs using pickle
#     serialized_graphs = [pickle.dumps(graph) for graph in graphs]
    
#     # Save the dataset
#     with h5py.File(file_name, 'w') as f:
#         f.create_dataset('graphs', data=np.string_(serialized_graphs))
#         f.create_dataset('labels', data=labels)

# def load_dataset_graph(file_name):
#     with h5py.File(file_name, 'r') as f:
#         serialized_graphs = f['graphs'][:]
#         labels = f['labels'][:]

#     graphs = [pickle.loads(graph.tobytes()) for graph in serialized_graphs]
#     return graphs, labels