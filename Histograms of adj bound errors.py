import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import math
import pylab 
import scipy.stats as stats


#the following function will be useful to get an appropriate upper y-axis limit
def round_up_to_nearest_n(num, n):
    return math.ceil(num / n) * n

#given a network G2, calculate the relevant bounds on the adjacency matrix as well as
#its maximum eigenvalue
 
def EIGEN_BOUND(G2):
    n = G2.number_of_nodes()
    
    #find kappa_1
    eigen_A = nx.adjacency_spectrum(G2)
    max_eigen_A = max(eigen_A)
    
    #calculate statistics for bounds
    avg_degree = sum(np.array([G2.degree[i] for i in range(n)]))/n
    avg_square_degree = sum(np.array([G2.degree[i]**2 for i in range(n)]))/n    
    return (max_eigen_A,  avg_degree,  avg_square_degree**0.5)


#initialize lists
eigen_lst = []
first_bnd_lst = []
second_bnd_lst = []

#now calculate the errors to be used in the histogram plot
for i in range(1,100001):
    
        #define the network
        G2 = nx.erdos_renyi_graph(100, 0.333333, seed=None, directed=False)
        
        #add G2's kappa_1 to the list
        eigen_lst = eigen_lst + [ EIGEN_BOUND(G2)[0] ]
        
        #Add the bounds to the respective lists
        first_bnd_lst = first_bnd_lst + [ EIGEN_BOUND(G2)[1] ]
        second_bnd_lst = second_bnd_lst + [ EIGEN_BOUND(G2)[2] ]


#Calculate the differences between the lower bounds and kappa_1
x_1 = np.array(eigen_lst) - np.array(first_bnd_lst)
x_2 = np.array(eigen_lst) - np.array(second_bnd_lst)

#obtain the edges of the bins
bins=np.histogram(np.hstack((x_1,x_2)), bins=1000)[1]
y_x1, x_x1, _ = plt.hist(x_1, bins)
y_x2, x_x2, _ = plt.hist(x_2, bins)

#make the top of the y axis 100 higher than the maximum bin height
y_lim = round_up_to_nearest_n(max(y_x1.max(), y_x2.max()), 100)


#Define layout/size of plot
plt.rcParams["figure.figsize"] = [6.50, 3.50]
plt.rcParams["figure.autolayout"] = True

#Split the plot into two subplots to have histograms side by side
ax1 = plt.subplot(1, 2, 1)

#Define layout of first histogram 'Errors in <k> bound'
ax1.set_xlim(left=0, right = 1)
ax1.set_ylim(bottom=0, top= y_lim )
ax1.hist(x_1, bins, color = '#68246d', edgecolor='none')
ax1.set_xlabel(r'Error in $\langle k \rangle$ Bound')

#Define layout of second histogram, using the same axis as the first
ax2 = plt.subplot(1, 2, 2, sharey=ax1, sharex=ax1)
ax2.hist(x_2, bins, color = "#68246d", edgecolor='none')
ax2.set_xlabel(r'Error in $\sqrt{\langle k^2 \rangle}$ Bound')
plt.show()

#Now plot normal distribution probability plots
stats.probplot(x_1, dist="norm", plot=pylab)
stats.probplot(x_2, dist="norm", plot=pylab)
pylab.show()
