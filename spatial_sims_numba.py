
import numpy as np
from sim_funcs_vectorize_n import*
from numba import prange
import itertools
from datetime import datetime
from scipy.integrate import odeint
from itertools import product

##start time
start = datetime.now()

mut_rates = np.array([.001],dtype=np.dtype('float64'))
pop_size = np.array([1000],dtype=np.dtype('int64'))
alphas = np.array([1.05],dtype=np.dtype('float64'))
#mut_rates = np.array([.01,.1])
#pop_size = np.array([100,200])
#delta=np.array([.00001,.0001])


trials = 1

#empty list to save
results = np.empty((len(pop_size),trials,len(mut_rates),len(alphas)))
inits = []
for K in pop_size:
    inits.append(initialize(K,1,.01)[0])
#inits=np.array(inits)  
iters = np.array(list(itertools.product(np.arange(len(mut_rates)),
                                        np.arange(len(pop_size)),
                                        np.arange(len(alphas)),np.arange(trials))))


@njit(
    parallel=True,
    fastmath=True
    )
def spatial_sims(mut_rates,pop_size,alphas,trials,iters,inits,start):
    results_main = np.zeros((len(iters),100),dtype=np.int64)
    results_sec = np.zeros((len(iters),9),dtype=np.float64)
    r = .1
    
    for i in prange(len(iters)):

        
        L, g_rates,t,arise_time, mut_events= fix_time_spatial(pop_size[iters[i][1]],r,alphas[iters[i][2]],
            mut_rates[iters[i][0]],inits[iters[i][1]],.05, True)

        clones_per_deme = np.sum(L[L[:,0]<K,2:]>0,axis=1)
        results_main[i,:len(clones_per_deme)] = np.sum(L[L[:,0]<K,2:]>0,axis=1)
        results_sec[i,:] = np.array([pop_size[iters[i][1]],mut_rates[iters[i][0]],r,
            alphas[iters[i][2]],len(g_rates)-2,t,arise_time,mut_events,iters[i][3]])

        #results_arr = np.array([results,start,['pop_size','mutation rate','wt growth rate',
        #    'alpha','deme_allele_count','fitness','fix time','est time',
        #    'tot_mut_events','sample numbers']])

        #np.save('results/spatial_results_%s.npy' % start, results_arr)




    return results_main, results_sec
                #convert results list to array, with the start time for record keeping
      
    



results_main, results_sec = spatial_sims(mut_rates,pop_size,alphas,trials,iters,inits,str(start))
#convert results list to array, with the start time for record keeping
results_arr = np.array([results_main,results_sec,start,['deme_allele_count','pop_size','mutation rate','wt growth rate',
    'alpha','surv_muts','fix time','est time',
    'tot_mut_events','sample numbers']])

np.save('results/spatial_results_%s.npy' % start, results_arr)
          
            
            
            
        
    