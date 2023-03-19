import numpy as np
from scipy.integrate import odeint
import networkx as nx
import matplotlib.pyplot as plt

# Define the function that represents the ODE system
def ode_system(x, t, adj_matrix, f, g):
    # Get the number of nodes in the graph
    n = len(x)

    # Initialize the derivatives vector
    dxdt = np.zeros(n)

    # Iterate over each node in the graph
    for i in range(n):
        # Calculate the sum of the coupling functions over all neighbours
        sum_contributions = 0
        for j in range(n):
            sum_contributions += adj_matrix[i][j] * g(x[i], x[j])

        # Calculate the intrinic derivative of node i
        dxdt[i] = f(x[i]) + sum_contributions

    return dxdt

#We define the network as an adjacency matrix.
#Alternatively, define a NetworkX graph and then calculate the adjacency matrix using:
#adj_matrix = nx.adjacency_matrix(G).toarray()
adj_matrix = np.array([
   [0, 0, 0, 0, 1, 0, 0],\
   [0, 0, 0, 0, 0, 0, 1],\
   [0, 0, 0, 0, 0, 0, 1],\
   [0, 0, 0, 0, 1, 0, 0],\
   [1, 0, 0, 1, 0, 1, 0],\
   [0, 0, 0, 0, 1, 0, 1],\
   [0, 1, 1, 0, 0, 1, 0]
   ])

#Calculate the key network information (the eigenvalues)
G = nx.from_numpy_matrix(adj_matrix)
lambda_max_reciprical = 1/max(nx.laplacian_spectrum(G))
kappa_plus_reciprical = 1/max(nx.adjacency_spectrum(G))
kappa_minus_reciprical = 1/min(nx.adjacency_spectrum(G))

# Define the functions f and g
def f(x):
    a = 1
    return a-x

def g(x_i, x_j):
    b = 0.5 
    g = 1 
    return -b/(g+x_i**2) + b/(g+x_j**2)

# Define the initial state vector
x1_0 = 1.5
x2_0 = 0.5
x3_0 = 1
x4_0 = 1.05
x5_0 = 0.75
x6_0 = 1
x7_0 = 1.25
x0 = np.array([x1_0, x2_0, x3_0, x4_0, x5_0, x6_0, x7_0])

# Define the time points at which to solve the ODEs
t_max = 100
num_points = 1000
t = np.linspace(0, t_max, num=num_points)

# Solve the system of ODEs
sol = odeint(ode_system, x0, t, args=(adj_matrix, f, g))

x1_sol = sol.T[0]
x2_sol = sol.T[1]
x3_sol = sol.T[2]
x4_sol = sol.T[3]
x5_sol = sol.T[4]
x6_sol = sol.T[5]
x7_sol = sol.T[6]
######################## PALATINATE COLOUR PALATE ################: 
plt.plot(t, x1_sol, label = r'$x_1$', color= '#68246d', linewidth= 3)
plt.plot(t, x2_sol, label = r'$x_2$', color = '#6d6824', linewidth= 3)
plt.plot(t, x3_sol, label = r'$x_3$', color = '#246d68', linewidth= 3)

############################ CAMBRIDGE BLUE COLOUR PALATE (for poster) ###################### 
# plt.plot(x, y1_sol, label = r'$x_1$', color= '#007234', linewidth= 3)
# plt.plot(x, y2_sol, label = r'$x_2$', color = '#003e72', linewidth= 3)
# plt.plot(x, y3_sol, label = r'$x_3$', color = '#72003e', linewidth= 3)
# plt.plot(x, y4_sol, label = r'$x_4$', color= '#726d00', linewidth= 3)
# plt.plot(x, y5_sol, label = r'$x_5$', color = '#72003e', linewidth= 3)
# plt.plot(x, y6_sol, label = r'$x_6$', color = '#72003e', linewidth= 3)
# plt.plot(x, y7_sol, label = r'$x_7$', color= '#371900', linewidth= 3)


plt.legend(loc="best", prop={'size': 12})
plt.xlabel(r'Time')
plt.margins(x=0)
#plt.yticks([0.90,0.95,1.00,1.05,1.10]) #only include for g = 1.1 plot
plt.show()