import numpy as np

def sort_eigen_values_vectors(eigen_values, eigen_vectors):
    sorted_indexes = eigen_values.argsort()
    eigen_values = eigen_values[sorted_indexes]
    eigen_vectors = eigen_vectors[:, sorted_indexes]
    
    return eigen_values, eigen_vectors

def print_eigen_values_vectors(eigen_values, eigen_vectors):
    for i in range(eigen_values.shape[0]):
        print(f'Autovalor[{i}] = {eigen_values[i]}')
    
    print()

    for i in range(eigen_vectors.shape[1]):
        print(f'Autovetor[{i}] = {eigen_vectors[:, i].T}')

    