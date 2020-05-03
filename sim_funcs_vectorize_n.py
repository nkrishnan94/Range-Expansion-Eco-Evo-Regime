#from sim_funcs_1 import*
from numba import jit,njit
import numpy as np
from numpy.random import choice
from numba import prange
import random
import matplotlib.pyplot as plt
from scipy.integrate import odeint 
from numba import jit,njit
##standing wave solution of Fisher wave


def standing_wave(y0,x,D,rw):
    w = y0[0]      ##initial value for wave profile at x =0, i.e. w(x=0)
    z = y0[1]      ##initial value for rate of change of profile w.r.t. position x , at x=0 i.e. dw/dx(x=0)
    dwdx = z
    dzdx =(-2*((rw*D)**.5)*dwdx -w*rw*(1-w))/D
    
    return [dwdx,dzdx]
    
    return [dwdx,dzdx]
def initialize(K,n_allele,mu):
    ## generate standing wave
    stand = odeint(standing_wave,[1,-(2*2**.5)/K],np.arange(70),args=(2*2**.5,1))[:,0]
    
    ## cuttoff non integer cell density based off of carry capacity K
    w_0 = (K*stand).astype(int)
    w_0 = w_0[w_0>1]
    
    ## subtract wild type cells from carrying capacity to get 'emopty' particle count
    L = np.vstack(((K-w_0),w_0)).T
    L = np.append(L,np.zeros((len(w_0),n_allele-1)),axis=1)

    ##array strucutre of an empty deme to be added on as needed 
    L_empty= np.append([K],np.zeros(n_allele,dtype=int))

    ## add on some number of empty demes
    for i in range(500):
        L= np.append(L,[L_empty],axis=0)
        
    return L.astype(int), L_empty


@njit
def true_bincount(arr,minlength):
    binned = np.bincount(arr)
    return np.append(binned,np.zeros(minlength - len(binned),dtype=np.int64))



@njit
def advindexer(arr,index, to_index):
    arr[index] = to_index
    return arr
    


@njit
def update(L, ## population
    L_empty, ## empty deme structure
    P, ## porbability matrix for mutation
    K, # population size
    g_rates,
    r,
    alpha,
    mu): ##mutation rate
        #L_tip = np.where(L[:,0]!=K)[0][-1]
    n_allele = len(g_rates) -1
    alleles=np.arange(len(g_rates))
    rands = np.random.random((3,K))
    n_demes = np.where(L[:,0]!=K)[0][-1] +2
    deme_seeds = np.random.randint(0,n_demes,(3,K))
    neighbs = np.append(np.expand_dims(deme_seeds[0,:],0),
           np.expand_dims((((rands[0,:]<.5) & (deme_seeds[0,:]) !=0) | (deme_seeds[0,:] == (n_demes -1)) *1) *-2+1 +deme_seeds[0,:],0),axis=0).T

    neighb_counts = np.bincount(neighbs.flatten(),
                                #np.max(neighbs)
                               )
    mig_picks = np.zeros((K,2),dtype=np.int64)
    cnt = 0
    for i in np.unique(neighbs.flatten()):
        to_ind = np.random.choice(np.repeat(alleles,L[i]),neighb_counts[i],replace=False)
        #to_ind = np.repeat(alleles,L[i])[np.searchsorted(np.arange(0,1,1/(K-1)), rands[2,cnt (cnt+neighb_counts[i])],side="right")]
        inds = np.where(neighbs==i)
        #mig_picks[np.where(neighbs==i)[0],:].take(np.where(neighbs==i)[1]+np.arange(0,len(np.where(neighbs==i)[1])*2,2)) = np.random.choice(np.repeat(alleles,L[i]),neighb_counts[i],replace=False)
        for ind in range(len(inds[0])):
            mig_picks[inds[0][ind],inds[1][ind]] = to_ind[ind] 
        cnt+=1    

    for i in np.unique(neighbs.flatten()):
        #L[i] 
        #L[i]+= neighbs[np.where(neighbs==i)[0],:].take(np.where(neighbs==i)[1]+np.arange(0,len(np.where(neighbs==i)[1])*2,2))
        inds = np.where(neighbs==i)
        L[i] -= true_bincount( mig_picks[inds[0],:].take(inds[1]+np.arange(0,len(inds[1])*2,2)),
                                                          n_allele+1)
        L[i] += true_bincount( mig_picks[inds[0],:].take((inds[1]==0)*1+np.arange(0,len(inds[1])*2,2)),
                              n_allele+1) 
    dup_counts = np.bincount(deme_seeds[1,:],
                               #np.max(deme_seeds[1,:])
                              )
    dup_picks = np.zeros((K,2),dtype=np.int64)

    for i in np.unique(deme_seeds[1,:].flatten()):
        dup_picks[np.where(deme_seeds[1,:]==i)] = np.random.choice(np.repeat(alleles,L[i]),(dup_counts[i],2),replace=False)

    for i in np.unique(deme_seeds[1,:]):
        dup_inds = np.where(deme_seeds[1,:]==i)
        pairs =  dup_picks[np.where(deme_seeds[1,:]==i)].T
        dup_bool =(P[:,pairs[1]].T.take(pairs[0])>rands[0].take(dup_inds[0]))
        L[i] +=true_bincount(pairs[:,dup_bool].take(np.arange(len(pairs[:,dup_bool][0]))),n_allele+1) -  true_bincount(pairs[:,dup_bool].take(len(pairs[:,dup_bool][0])+np.arange(len(pairs[:,dup_bool][0]))) ,n_allele+1)
   
    mut_counts = np.bincount(deme_seeds[2,:], 
                               #np.max(deme_seeds[2,:])
                              )
    mut_picks = np.zeros(K,dtype=np.int64)

    for i in np.unique(deme_seeds[2,:].flatten()):
        mut_picks[np.where(deme_seeds[2,:]==i)] =np.random.choice(np.repeat(alleles,L[i]),(mut_counts[i]),replace=False)


    mt_cnt = 0
    for i in np.unique(deme_seeds[2,:]):


        to_mut = mut_picks[np.where(deme_seeds[2,:]==i)]
        mut_bool =(to_mut != 0 ) &(g_rates[to_mut] ==r ) & (mu>rands[1,:].take(np.where(deme_seeds[2,:]==i)[0]))


        ##remove original cell and add mutated cell to cell counts
        s_new = np.random.normal(r*alpha,.001,np.sum(mut_bool))
        if s_new.size > 0:
            for s in s_new:

                g_rates = np.sort(np.append(g_rates,np.asarray(s)))
                s_pos = np.where(g_rates==s)[0][0]
                #print(s_pos)
                L = np.concatenate((L[:,:(s_pos)].T, np.expand_dims(np.zeros(len(L),dtype=np.dtype( np.int64)),0), L[:,(s_pos):].T)).T
                #L = np.concatenate((L[:,:(s_pos)].T,np.expand_dims(np.zeros(len(L)).astype(int),0),L[:,(s_pos):].T)).T 




                P = np.ones((len(g_rates),len(g_rates)))

                P[0,:] = 1 - g_rates
                #alleles = np.arange(n_allele+1)
                L_empty = np.array([K]+[0]*(len(g_rates)-1))
                L[i,1] -=1
                L[i,s_pos] +=1
                mt_cnt+=1
    #shift = 0
    #while L[0,0]<int(.02*K):
    #    L=L[1:,:]
    #    shift+=1

    #for i in range(shift):
    #    L=np.append(L,np.expand_dims(L_empty,0),axis=0)
    return L, L_empty, g_rates, P,mt_cnt

@njit
def run_stepping_stone(n_gen,## nunmber of gnerations
                K, ## population size
                r,
                alpha,## fitness landscape (Growthrates of each genotype (should be <<1))
                mu,
                L_init,prune):
    


    ##initialize probability matrix
    L_empty = L_init[-1]
    L = L_init
    g_rates = np.array([0,r])
    n_allele = len(g_rates)-1
    P = np.ones((n_allele+1,n_allele+1))
    P[0,:] = 1 - g_rates
    ## list of allele number - pre-established so array doesnt have to be regenerated

    #if track:
    #L_history = []
    gr_hist = []
    #    L_history = [L]
    mut_events=0
    scount = 0
    for t in range(n_gen):
        #if scount == 0:
        #    mu_t =0
        #else:
        #    mu_t = mu
        L, L_empty, g_rates, P,new_muts =update(L,L_empty,P,K,g_rates,r,alpha,mu)
        mut_events += new_muts
        gr_hist.append(g_rates)
        
        if prune:
            non_empty = np.sum(L,axis=0)!=0
            L= L[:,non_empty]
            L_empty = L_empty[non_empty]
            g_rates = g_rates[non_empty]

        
        while L[0,0]<1:
            L=L[1:,:]
            L=np.append(L,np.expand_dims(L_empty,0),axis=0) 
            scount+=1

         

    #    if track:
        #L_history.append(L)
        
    #if track:
    #    return L_history, g_rates
    #else:
    return  L, g_rates,gr_hist,scount

@njit
def fix_time_spatial(K, ## population size
                     r,
                     alpha,## fitness landscape (Growthrates of each genotype (should be <<1))
                     mu,
                     L_init,
                     thresh,
                     prune):  ## mutation rate
    
    
    
    
    L_empty = L_init[-1]
    L = np.copy(L_init)
    g_rates = np.array([0,r])
    n_allele = len(g_rates)-1
    P = np.ones((n_allele+1,n_allele+1))
    P[0,:] = 1 - g_rates
    
    fixed=False
    muts_check = False
    del_check =False
    del_fix = False
    arise_time =0 
    fixed=False
    mut_events = 0
    t = 0
    scount = 0
    L_i= np.copy(L_init)
    while not fixed:
        #if scount < 10:
        #    mu_t =0
        #    L_i = np.copy(L)
        #else:
        #    mu_t = mu
        L, L_empty, g_rates, P,new_muts =update(L,L_empty,P,K,g_rates,r,alpha,mu)
        mut_events += new_muts
        
        if prune:
            non_empty = np.sum(L,axis=0)!=0
            L = L[:,non_empty]
            L_empty = L_empty[non_empty]
            g_rates = g_rates[non_empty]
        

        while L[0,0]<1:
            L=L[1:,:]
            L=np.append(L,np.expand_dims(L_empty,0),axis=0)
            scount+=1
            
        if len(g_rates)>2 ==True and not muts_check:
            ## record time
            arise_time = t
            muts_check = True
        if not len(g_rates)>2:
            muts_check = False
            arise_time = 0

        ## check if fixed
        n_demes = np.where(L[:,0]!=K)[0][-1] +2
        wt_ind = np.where(g_rates==r)[0][0]
        #fix_bools = L[:(n_demes-2),wt_ind] < np.asarray(thresh*np.sum(L[:(n_demes-2),1:],axis=1),dtype=np.dtype('int64'))
        fixed = np.all((L[L[:,0]!=K,wt_ind])/(K-L[L[:,0]!=K,0]) < thresh)
        #fixed = np.all(fix_bools)
        #fixed = np.sum(L[:,1:n_allele])<(thresh*init_pop)

        t+=1


    return L, L_i, g_rates,t,arise_time, mut_events, scount