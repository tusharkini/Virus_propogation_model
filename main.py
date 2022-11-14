import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
from operator import itemgetter
from copy import copy, deepcopy

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
    eigen_value = sorted(eigen_value, reverse=True)
    #take only the real part of the complex eigen value
    return eigen_value[0].real

def static_delta(eigen_value, delta,title):
    print(f"Computing minimum transmission probability with delta = {delta}")
    # creating the test data
    x = np.linspace(0.001, 1,1000)
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
    x = np.linspace(0.001, 1,1000)
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
    #thus we get max delta with s==1 so delta will be lesser than this value
    max_delta = (eigen_value*beta)/1.0
    if max_delta >= 1.0 :
        return 1.0
    else:
        return max_delta


def policy_A(adjacency_matrix, g, k):
    immunized = set()
    total_node = nx.number_of_nodes(g)
    while len(immunized) < k:
        # get a random node from the network
        r = random.randint(0, total_node-1)
        if r not in immunized:
            immunized.add(r)
    # vaccinate the chosed nodes, no edge so cannot infect anyone
    for node in immunized:
        for i in range(len(adjacency_matrix[node])):
            adjacency_matrix[node][i] = 0
            adjacency_matrix[i][node] = 0
    return adjacency_matrix

def policy_B(adjacency_matrix, g, k):
    immunized = set()
    # sort the nodes based on degree of node
    degree = sorted(list(g.degree()), key=lambda x: x[1], reverse=1)
    for i in range(k):
        immunized.add(degree[i][0])

    for node in immunized:
        # vaccinate the chosen nodes, no edge so cannot infect anyone
        for i in range(len(adjacency_matrix[node])):
            adjacency_matrix[node][i] = 0
            adjacency_matrix[i][node] = 0
    return adjacency_matrix

def policy_C(g, k):
    immunized = set()
    total_node = nx.number_of_nodes(g)
    while len(immunized) < k:
        # get the node with the highest degree
        degree = sorted(list(g.degree()), key=lambda x: x[1], reverse=1)
        immunized.add(degree[0][0])
        #remove the node from the graph and iterate again
        g.remove_node(degree[0][0])

    # create empty adjacency matrix
    new_adjacency_matrix = [[0 for _ in range(total_node)] for _ in range(total_node)]
    # populate the adjacency matrix
    for i in nx.edges(g):
        new_adjacency_matrix[i[0]][i[1]] = 1
        new_adjacency_matrix[i[1]][i[0]] = 1
    return new_adjacency_matrix


def policy_D(adjacency_matrix, g, k):
    total_node = nx.number_of_nodes(g)
    value, vector = np.linalg.eig(adjacency_matrix)
    eigen_set = [(value[i], vector[i]) for i in range(len(value))]
    eigen_set = sorted(eigen_set, key=lambda x: x[0], reverse=1)
    #get the eigen vetcor for corresponding largest eigen value
    largest_vector = eigen_set[0][1]
    largest_vector = np.absolute(largest_vector)
    # get the first k largest value in the eigen vector and get the corresponding node
    target = [u[0] for u in sorted(enumerate(largest_vector), reverse=True, key=itemgetter(1))[:k]]
    for i in target:
        g.remove_node(i)
    # create empty adjacency matrix
    new_adjacency_matrix = []
    for i in range(total_node):
        row = []
        for j in range(total_node):
            row.append(0)
        new_adjacency_matrix.append(row)
    # populate the adjacency matrix
    for i in nx.edges(g):
        new_adjacency_matrix[i[0]][i[1]] = 1
        new_adjacency_matrix[i[1]][i[0]] = 1
    return new_adjacency_matrix

def find_k_vaccine(adjacency_matrix, g, policy):
    strengths = []
    number_vaccines = []
    step = 50
    #when flag set to 0 we have found min k for which strength <=1
    flag = 1
    while flag:
        number_vaccines.append(
            step if not number_vaccines else number_vaccines[-1] + step)
        k = number_vaccines[-1]
        if policy == 'A':
            new_adjacency = policy_A(adjacency_matrix.copy(), g.copy(), k)
        elif policy == 'B':
            new_adjacency = policy_B(adjacency_matrix.copy(), g.copy(), k)
        elif policy == 'C':
            new_adjacency = policy_C(g.copy(), k)
        elif policy == 'D':
            new_adjacency = policy_D(adjacency_matrix.copy(), g.copy(), k)
        largest_eigen = compute_largest_eigen_value(new_adjacency)
        strength = largest_eigen * Cvpm1
        if strength < 1:
            flag = 0
        strengths.append(strength)
    plt.plot(number_vaccines, strengths)
    title = f"Strength vs varying k with beta={beta1} and delta={delta1} with Policy {policy}"
    plt.title(title)
    plt.xlabel('Number of vaccines k ')
    plt.ylabel('Effective Strength s ')
    plt.axhline(y=1, linestyle='--', linewidth=1, color='r')
    plt.savefig(f'./results/{title}.png', bbox_inches='tight')
    plt.close()
    return number_vaccines[-1]

def main():
    # Read the data from file and add edges to empty graph
    print('OPTION 1 PART 1')
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
    ''''
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
    print('OPTION 1 PART 2')


    print('OPTION 1 PART 3')
    # Implements immunization policies
    # Calculate the effective strength (s) of the virus on the immunized contact network
    print("Immunization with Policy A")
    adjacency_matrix_A = policy_A(adjacency_matrix.copy(), g.copy(), k=200)
    largest_eigenvalue_A = compute_largest_eigen_value(adjacency_matrix_A)
    strength_A = largest_eigenvalue_A * Cvpm1
    print(f"Effective strengh of Policy A with beta = {beta1}, delta = {delta1} is: {strength_A}")

    print("Immunization with Policy B")
    adjacency_matrix_B = policy_B(adjacency_matrix.copy(), g.copy(), k=200)
    largest_eigenvalue_B = compute_largest_eigen_value(adjacency_matrix_B)
    strength_B = largest_eigenvalue_B * Cvpm1
    print(f"Effective strengh of Policy B with beta = {beta1}, delta = {delta1} is: {strength_B}")
'''
    print("Immunization with Policy C")
    adjacency_matrix_C = policy_C(g.copy(), k=200)
    largest_eigenvalue_C = compute_largest_eigen_value(adjacency_matrix_C)
    strength_C = largest_eigenvalue_C * Cvpm1
    print(f"Effective strengh of Policy C with beta = {beta1}, delta = {delta1} is: {strength_C}")

    print("Immunization with Policy D")
    adjacency_matrix_D = policy_D(adjacency_matrix.copy(), g.copy(), k=200)
    largest_eigenvalue_D = compute_largest_eigen_value(adjacency_matrix_D)
    strength_D = largest_eigenvalue_D * Cvpm1
    print(f"Effective strengh of Policy D with beta = {beta1}, delta = {delta1} is: {strength_D}")
    
    # Find minimum number of vaccines required k
    print("Finding minimum number of vaccine k for policy A")
    min_number_vaccines_A = find_k_vaccine(adjacency_matrix, g, 'A')
    print(f'Minimum vaccines required for policy A with beta={beta1}, delta={delta1} is: {min_number_vaccines_A}')

    print("Finding minimum number of vaccine k for policy B")
    min_number_vaccines_B = find_k_vaccine(adjacency_matrix, g, 'B')
    print(f'Minimum vaccines required for policy A with beta={beta1}, delta={delta1} is: {min_number_vaccines_B}')

    print("Finding minimum number of vaccine k for policy C")
    min_number_vaccines_C = find_k_vaccine(adjacency_matrix, g, 'C')
    print(f'Minimum vaccines required for policy C with beta={beta1}, delta={delta1} is: {min_number_vaccines_C}')

    print("Finding minimum number of vaccine k for policy D")
    min_number_vaccines_D = find_k_vaccine(adjacency_matrix, g, 'D')
    print(f'Minimum vaccines required for policy D with beta={beta1}, delta={delta1} is: {min_number_vaccines_D}')

if __name__ == "__main__":
    main()