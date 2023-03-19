import numpy as np
from scipy.integrate import odeint
import networkx as nx
import matplotlib.pyplot as plt

#since each rossler oscillator is a 3-component vector variable, with the first component
#called the `x-component', the notation can be confusing. Please refer to Appendix for...
#full explanation in detail.

#Before reading the code, we define the notation we use throughout:
#bold_x = (x,y,z) refers to the vector variable of a oscillator at some unspecified node.
#bold_X refers to the array that stores all the bold_x, i.e bold_X = (bold_x_1, bold_x_2,..., bold_x_n) = (x_1, y_1, z_1, x_2, y_2, z_2,..., x_n, y_n, z_n)
#bold_X_0 is the inital condition for bold_X

#Makes matplotlib use latex font as default
plt.rcParams['text.usetex'] = True
#Use seaborn style. Use 'print(plt.style.available)' to show all alternative styles
plt.style.use('seaborn')


# Set parameter values to those found in original paper
a = 0.2
b = 0.2
c = 5.7


# Define the Rossler intrinsic dynamics of bold_x, the vector variable at a single node
def f(bold_x, t):
    #break apart bold_x into its elements
    x = bold_x[0]
    y = bold_x[1]
    z = bold_x[2]
    
    #intrinsic derivative for x component
    x_dot = -y - z
    
    #intrinsic derivative for y component
    y_dot = x + a*y
    
    #intrinsic derivative for z component
    z_dot = b + z*(x - c)
   
    
    bold_x_dot = np.array([x_dot, y_dot, z_dot])

    return bold_x_dot


#Define whatever graph you want to use:
G = nx.complete_graph(3)
adj_matrix = nx.adjacency_matrix(G).toarray() #can use change to sparse if using really large network
n = G.number_of_nodes()

#Set the initial conditions for all oscillators
bold_X_0 = np.zeros(3*n)
for i in range(n):
    #Set the x-component initial variables
    bold_X_0[i*3] = -10 + 10*i
    
    #Set the y-component initial variables
    bold_X_0[i*3+1] = -1 + 1*i 
    
    #Set the z-component initial variables
    bold_X_0[i*3+2] = 1 + 4*i

print(bold_X_0)
#calculates derivatives at all nodes and then returns them all as a vector
def bold_X_dot_func(bold_X, t): 
    #Define the coupling strength
    sigma = 1
    
    #Initialise a vector of size 3*n to store derivatives
    bold_X_dot = np.zeros_like(bold_X)

    for i in range(n):
        
        #Set  intrinsic dynamics for i_th node
        bold_X_dot[i*3:i*3+3] = f(bold_X[i*3:i*3+3], t)
        
        #Add coupling dynamics
        for j in range(n):
                #we use x-coupling i.e we add on sigma*sum_j(A_{ij}*(x_i - x_j)) to x_i_dot
                coupling_j = (adj_matrix[i][j])*(bold_X[j*3] - bold_X[i*3])
                bold_X_dot[i*3] += sigma*coupling_j

    return bold_X_dot


t_max = 20
t_step = 0.01
t = np.arange(0, t_max, t_step)

# Solve the system of ODEs
bold_X_SOL = odeint(bold_X_dot_func, bold_X_0, t)


# Plot x-component solutions in 2D
fig, ax = plt.subplots()
for i in range(n):
    ax.plot(t, bold_X_SOL[:, i*3], label=f'Node {i+1}', linewidth= 4)
ax.set_xlabel(r'Time', fontsize= 40)
ax.set_ylabel(r'$x$', fontsize= 50)
plt.margins(x=0)
plt.yticks([-10, -5, 0.0, 5.0, 10], fontsize=30)
plt.xticks([0, 5, 10, 15, 20], fontsize=30)
ax.legend(fontsize=20, frameon = True, framealpha = 0.9)
plt.show()