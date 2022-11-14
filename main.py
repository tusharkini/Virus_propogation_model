import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
from operator import itemgetter

random.seed(100)

#Given values for the model to run on
beta1 = 0.2
delta1 = 0.7
Cvpm1 = beta1 / delta1
beta2 = 0.01
delta2 = 0.6
Cvpm2 = beta2 / delta2

def compute_largest_eigen_value(adjacency_matrix):
    eigen_value, eigen_vector = np.linalg.eig(adjacency_matrix)
    #eig_set = [(eigenvalue[i], eigenvector[i]) for i in range(len(eigenvalue))]
    eigen_value = sorted(eigen_value, reverse=True)
    #take only the real part of the complex eigen value
    return eigen_value[0].real

def static_delta(eigen_value, delta,title):
    print(f"Computing minimum transmission probability with delta = {delta}")
    # creating the test data
    x = np.linspace(0, 1,1000)
    y=(eigen_value*x)/delta

    # rendering the chart
    plt.plot(x,y)
    plt.title(title)
    plt.xlabel('Minimum transmission probability Beta')
    plt.ylabel('Effective Strength s')
    plt.axhline(y=1, linestyle='--', linewidth=1, color='r')
    plt.savefig(f'./results/{title}.png', bbox_inches='tight')
    plt.close()
    # for there to be epidemic we would have s>1
    #thus we get min beta with s==1 so beta will be greater than this value
    min_beta = (1*delta)/eigen_value
    return min_beta


def static_beta(eigen_value, beta, title):
    # creating the test data
    x = np.linspace(0, 1,1000)
    y=(eigen_value*beta)/x

    # rendering the chart
    plt.plot(x,y)
    plt.title(title)
    plt.xlabel('Maximum healing probability Delta')
    plt.ylabel('Effective Strength s')
    plt.axhline(y=1, linestyle='--', linewidth=1, color='r')
    plt.savefig(f'./results/{title}.png', bbox_inches='tight')
    plt.close()
    # for there to be epidemic we would have s>1
    #thus we get min beta with s==1 so beta will be greater than this value
    max_delta = (eigen_value*beta)/1.0
    if max_delta >= 1.0 :
        return 1.0
    else:
        return max_delta

def main():
    # Read the data from file and add edges to empty graph
    g = nx.Graph()
    file = './datasets/static.network'
    with open(file) as f:
        next(f)
        for line in f:
            line = line.split()
            g.add_edge(int(line[0]), int(line[1]))
    
    # creating a list of lists for adjacency matrix representation
    total_node = nx.number_of_nodes(g)
    adjacency_matrix = []
    for i in range(total_node):
        row = []
        for j in range(total_node):
            row.append(0)
        adjacency_matrix.append(row)
    # creating an adjacency matrix for anundirected graph
    for i in nx.edges(g):
        adjacency_matrix[i[0]][i[1]] = 1
        adjacency_matrix[i[1]][i[0]] = 1
    
    largest_eigenvalue = compute_largest_eigen_value(adjacency_matrix)
    s1 = largest_eigenvalue * Cvpm1
    s2 = largest_eigenvalue * Cvpm2
    print(f"Effective strengh with beta = {beta1}, delta = {delta1} is: {s1}")
    print(f"Effective strengh with beta = {beta2}, delta = {delta2} is: {s2}")
    
    # b. Fix delta and evaluate beta affecting the effective strength
    # delta1 results
    fixed_delta1_min_beta = static_delta(
        largest_eigenvalue, delta1, f'Strength vs varying beta with delta={delta1}')
    print(f"Minimum transmission probability: {fixed_delta1_min_beta} with fixed delta={delta1}")
    # delta2 results
    fixed_delta2_min_beta = static_delta(
        largest_eigenvalue, delta2, f'Strength vs varying beta with delta={delta2}')
    print(f"Minimum transmission probability: {fixed_delta2_min_beta} with fixed delta={delta2}")

    # c. Fix beta and evaluate delta affecting the effective strength
    # beta1 results
    fixed_beta1_max_delta = static_beta(
        largest_eigenvalue, beta1, f'Strength vs varying delta with beta={beta1}')
    print(f"Maximum healing probability: {fixed_beta1_max_delta} with fixed beta={beta1}")
    # beta2 results
    fixed_beta2_max_delta = static_beta(
        largest_eigenvalue, beta2, f'Strength vs varying delta with beta={beta2}')
    print(f"Maximum healing probability: {fixed_beta2_max_delta} with fixed beta={beta2}")
    pass


if __name__ == "__main__":
    main()