# Synchronisation-on-Networks
Finalised: The code I created for my Project III at Durham University about the synchronisation of ODEs on networks.

# ODEs-on-Networks
The code I created for my Project III at Durham University about ODEs on networks. My default matplotlib style is seaborn.

The first file is entitled 'Gossip - Solves ODEs on Networks'. This python script solves ODEs given any network, and functions f(x_i) and g(x_i, x_j). It solves them in the case where the x_i are 1-D (i.e not vectors). Mathematically, given a user specified network, the program solves the set of coupled ODEs given by: \frac{dx_i}{d t}=f(x_i)+\sum_{j=1}^{n} A_{ij} g(x_i,x_j), where i = 1,..., n; the number of nodes in the network, and A_{i j} are the elements of the adjacency matrix. We specify the specific network as well as the functions f and g that we use in the 'gossip example', seen in the corrosponding part of the essay.

The program 'Gossip - Solves ODEs on Networks' only works when we have one variable per node. The second file 'Solving x-Coupled Rossler Oscillators on a Network' is an example of solving a system of ODEs on a network with multiple variables per node. It solves coupled Rossler Oscillators in the case where we have x-coupling of the form x_i - x_j. The code can solve the coupled Rossler system on any network. It makes three 2-D plots showing the x, y, and z components each against time. Finally, it creates a 3-D plot showing the evolution of all three components, (x,y,z), of each node simulataneously.

The third program is entitled 'The Average Eigen-Ratio for Connected Watts Strogatz Networks n, 2m are 100, 4'. It creates the figures showing the (averaged) eigen-ratio for connected watts strogatz small worlds with varying rewiring probability. It then plots p_min, the minimum rewiring probability such that the averaged eigenratio of the resulting small world is half that of the original RL(100, 4) network.  

The fourth program is entitled 'P_Min Heatmap for Connected_Watts_Strogatz Small World Model'. It is similiar to the third program, expect we just plot the p_min for varying n and m on a heatmap.

The fifth program is entitled 'Histograms of adj bound errors' and calculates the errors in the first two bounds on the adjacency matrix in my essay. The code calculates the 100,000 errors for each bound and plots the results on two seperate subplots. We use an Erdos-Renyi random network with n = 100 nodes and connection probability, p = 1/3 as an example.  


