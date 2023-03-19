import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#since python does not allow us to write 2m (and `two_m' is clunky), we use k in place of 2m
def p_min_func(n,k):
    p = -0.001
    R_L = nx.connected_watts_strogatz_graph(n, k, 0)
    eigen_L = nx.laplacian_spectrum(R_L)
    ratio_OG = max(eigen_L)/eigen_L[1]
    mean = 10000000000
    
    #While loop to find p_min
    while mean > ratio_OG/2:
        eigen_ratio_list = []
        p = p + 0.001
        
        #prevent floating point errors
        if p > 1:
            p = 1
            
        #now calculate num_simulation_per_p small worlds and take the average of their eigen-ratios
        for j in range(1000):
            small_world = nx.connected_watts_strogatz_graph(n, k, p) # k < log n works well, see reference [8]
            
            #calculate eigen-ratio
            eigen_L = nx.laplacian_spectrum(small_world)
            ratio = max(eigen_L)/eigen_L[1]
            
            #add to list
            eigen_ratio_list = eigen_ratio_list + [ratio]
            
        #calculate mean and then go back to start of while loop
        eigen_ratio_list = np.array(eigen_ratio_list)
        mean = np.mean(eigen_ratio_list)
    return p
     
nkp_array = []
for n in range(100,900,100):
    #useful to estimate overall run time
    print(n)
    for k in range(4, 12, 2):
        nkp_entry = [n,k, p_min_func(n,k)]
        nkp_array = nkp_array + [nkp_entry]
    

#Create a DataFrame from the array
nkp_array = np.array(nkp_array) 
df = pd.DataFrame(nkp_array, columns=['n', 'k', 'p-min'])

#df = df[12:33] #this just selects n from 400 to 800, used to create second heat map
df['n'] = (df['n']/100)
df['n'] = df['n'].apply(lambda x: str(int(x)))
df['k'] = df['k'].apply(lambda x: str(int(x)))

#pivot the data frame so it can be fed into seaborn.heatmap() later on
pivot = df.pivot('n', 'k', 'p-min')

#reorder the colums as desired 
cols = pivot.columns.tolist()
cols[-1], cols[1] = cols[1], cols[-1]
pivot = pivot[cols]


#use seaborn to create heatmap
sns.set(font_scale = 1.5)
ax = sns.heatmap(pivot, vmin=0, vmax=0.05, linecolor='white', linewidth = 0.5) #linewidths=0.5
plt.title(r'\textbf{Heat Map of} $\mathbf{p_{{min}}}$', fontsize = 20)
ax.set_xlabel(r'$\mathbf{2m}$', fontsize = 20)
ax.set_ylabel(r'$\mathbf{n \times 10^{-2}}$', fontsize = 20)
plt.show()




