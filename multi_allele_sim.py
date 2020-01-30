from stepping_stone_funcs import*
from tqdm import tqdm
from itertools import product
import numpy as np
import datetime

start = datetime.datetime.now()
K_space = np.array([100,500])
mu_space = np.array([.001,.005])
alphas = np.array([3,5])

reps= 5

params_out = ['reps='  +str(reps),'K='+str(K_space),'mu='+str(mu_space),'alphas='+str(alphas) ]

r=.1

results = []

for rep, K,mu,a in tqdm(product(range(reps), K_space,mu_space, alphas)):
    L = fix_time(K,10,r,a,mu,3/K,False)
    results.append([L])
    
    
    
    np.save('multi_allele_data_%s.npy' % start,np.array([results,params_out]))
    
np.save('multi_allele_data_%s.npy' % start,np.array([results,params_out]))

    
    
    
            





    
    
  
    
    
    
    
            


