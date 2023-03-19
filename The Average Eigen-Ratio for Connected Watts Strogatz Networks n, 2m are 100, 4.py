import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#create a list to store the mean eigen-ratios
y = []

#caculate ratio_OG - the original eigen-ratio of the RL(100, 4) - so we can see when it halves
R_L = nx.connected_watts_strogatz_graph(100, k = 4, p = 0)
eigen_L = nx.laplacian_spectrum(R_L)
ratio_OG = max(eigen_L)/eigen_L[1]


# define the incredments we will increase p by and the number of small-worlds calculated per value of p 
increment = 0.01
num_simulation_per_p = 1000 #Set this to 1 for first plot without any averaging 

#initialize p and p_min, the minimum p such that the averaged eigen-ratio...
#is half its original value

p = -increment
p_min = 10

#loop over values of p ranging from 0 to 1 with increments 0.01 
for i in range(101):
      p = p + increment
      
      #initialize the list that stores the eigen-ratios for one specific p
      eigen_ratio_list = []
      
      #prevents floating point errors
      if p > 1:
          p = 1 
          
      #now calculate num_simulation_per_p small worlds and take the average of their eigen-ratios
      for j in range(num_simulation_per_p):
          small_world = nx.connected_watts_strogatz_graph(100, 4, p) # k < log n works well, see reference [8]
          
          #calculate eigen-ratio
          eigen_L = nx.laplacian_spectrum(small_world)
          ratio = max(eigen_L)/eigen_L[1]
          
          #add to list
          eigen_ratio_list = eigen_ratio_list + [ratio]
      
      #calculate mean eigen-ratio for each p
      eigen_ratio_list = np.array(eigen_ratio_list)
      mean = np.mean(eigen_ratio_list)
      
      #store it for plotting later
      y = y + [mean]
      
      #store the first p such that the eigen-ratio halves, this corrosponds to the green point on the plot later on
      if mean < ratio_OG/2 and p < p_min:
          p_min = p
          p_min_ratio = np.mean(eigen_ratio_list)
          
      #this is useful to track the run time of the program
      print(i)

#create plot
p_0_to_1 = np.linspace(0,1, num = 101)
plt.plot(p_0_to_1, y, color = '#68246d', linewidth = 3)

#Improve the plots readability and add green marker at p_min
plt.plot([p_min], [p_min_ratio], marker="o", markersize=10, markeredgecolor="#286830", markerfacecolor="#286830")
plt.text(1.1*p_min,1.1*p_min_ratio,r'$(%.2f, %.2f)$' % (p_min, p_min_ratio),horizontalalignment= 'left', fontsize = 14)
plt.ylabel(r'Eigenratio, $\frac{\lambda_n}{\lambda_2}$', fontsize = 40)
plt.xlabel(r'p', fontsize = 40)
plt.yticks(fontsize = 20)
plt.xticks(fontsize = 20)
plt.margins(x=0)
plt.show()