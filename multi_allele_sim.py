from stepping_stone_funcs import*
from tqdm import tqdm
from itertools import product
import numpy as np
import datetime

start = datetime.datetime.now()
K_space = np.array([100,450,800])
mu_space = np.array([.0001,.0005,.001])
alphas = np.array([1.5,2,5])

reps= 5

params_out = ['reps='  +str(reps),'K='+str(K_space),'mu='+str(mu_space),'alphas='+str(alphas) ]

r=.1

results = []

for rep, K,mu,a in tqdm(product(range(reps), K_space,mu_space, alphas)):
    L,g = fix_time(K,r,a,mu,1/K,dist_key = 'normal')
    tip = np.where(L[:,0]==K)[0][0]
    clones =np.sum(L[:,1:]>0,axis=1)[:tip]
    results.append([clones,g])
    
    
    
    np.save('multi_allele_data_%s.npy' % start,np.array([results,params_out]))
    
np.save('multi_allele_data_%s.npy' % start,np.array([results,params_out]))

    
    
    
            





    
    
  
    
    
    
    
            


