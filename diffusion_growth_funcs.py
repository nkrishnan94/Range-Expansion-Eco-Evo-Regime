import numpy as np
from numpy.random import choice

import random
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import glob
import os

import numpy as np
from numpy.random import choice
import random
from scipy.integrate import odeint


##standing wave solution of Fisher wave
def standing_wave(y0,x,D,rw):
    w = y0[0]      ##initial value for wave profile at x =0, i.e. w(x=0)
    z = y0[1]      ##initial value for rate of change of profile w.r.t. position x , at x=0 i.e. dw/dx(x=0)
    dwdx = z
    dzdx =(-2*((rw*D)**.5)*dwdx -w*rw*(1-w))/D
    
    return [dwdx,dzdx]




def fit_dist(r,alpha,dist_key):
    dist_dict = {'exponential':np.random.exponential,
                 'gamma':np.random.gamma,
                 'normal':np.random.normal}
    func = dist_dict[dist_key]
    s_bar  = r*func(alpha)
    return s_bar

##standing wave solution of Fisher wave
def standing_wave(y0,x,D,rw):
    w = y0[0]      ##initial value for wave profile at x =0, i.e. w(x=0)
    z = y0[1]      ##initial value for rate of change of profile w.r.t. position x , at x=0 i.e. dw/dx(x=0)
    dwdx = z
    dzdx =(-2*((rw*D)**.5)*dwdx -w*rw*(1-w))/D
    
    return [dwdx,dzdx]


###given a list of demes (indexes), pick two neighobiring ones at random
def rand_neighbors(demes):
    ind_1 = np.random.choice(demes)
    left = demes[:ind_1][-1:]
    right = demes[ind_1:][1:2]
    neighb = np.append(left,right).flatten()
    ind_2=choice(neighb)
    neigh = [ind_1,ind_2]
    neigh.sort()
    return np.array(neigh)

## covert a list of cell counts for each particle type to an array of 
#all particles represented with their type represented by an itenged
def counts_to_cells(counts,n_allele):
    cells = np.unique(cells,return_counts=True)
    return cells

## covert an array of all cells and their type to a list of counts for each cell type 
def cells_to_counts(cell_types,g_rates):
    g_r = np.unique(cell_types,return_counts=True)[0]
    raw_counts = np.unique(cell_types,return_counts=True)[1]
    cell_counts = []
    for i in g_rates:
        if any(g_r == i):
            cell_counts.append(raw_counts[np.where(g_r == i)[0][0]])
        else:
            cell_counts.append(0)
    cell_counts = np.array(cell_counts)

    return cell_counts
    
    
## from a list 2d array of the cell list from two neighboring demes
##pick two cells to be swapped at random and return resulting cell list
def migration(cell_counts,g_rates,K,r):
    picks = []
    cell_types = [] 
    for i in [0,1]:
        pick_ind=choice(np.arange(K))
        picks.append(np.arange(K) == pick_ind)
        cell_types.append(np.repeat( g_rates, cell_counts[i].astype(int)))
    picks = np.array(picks)
    keep =  ~np.array(picks)
    


    cell_types[0]= np.append(cell_types[0][keep[0]], [cell_types[1][picks[1]]])
    cell_types[1]= np.append(cell_types[1][keep[1]], [cell_types[0][picks[0]]])
    return np.array(cell_types)

## from the cell list from the chosen d0eme
##pick *-\\\\\\\\\\\\\\\
#wo cells and replace one with a duplicate of the other according
##to transition matrix
def duplication(cell_counts,g_rates,K,r):
    pick_ind=choice(np.arange(K),2,replace= False)
    cell_types = np.repeat(g_rates, cell_counts.astype(int))
    picks = np.array([np.arange(K) == pick_ind[i] for i in [0,1]])
    r= np.random.random()
    P=prob_mat(g_rates)
    #print(cell_types)
    if P[tuple([np.where(cell_types[ind] == g_rates)[0] for ind in pick_ind])]> r:
        cell_types[pick_ind[1]] =cell_types[pick_ind[0]]
    return cell_types
    
 ## from the cell list from the chosen deme
##pick a cell and mutate it with probability according to mutation rate
def mutation(mu,alpha,dist_key, cell_counts,g_rates,K,r):

    cell_types = np.repeat(g_rates,cell_counts.astype(int))
    pick_ind=choice(np.arange(K),np.ones(K)/K)
    p= np.random.random()
    s_pos=0
    if mu>p:
        if cell_types[pick_ind] == r :
            s_new = fit_dist(r,alpha,dist_key)
            cell_types[pick_ind] = s_new
            g_rates = np.sort(np.append(g_rates,[cell_types[pick_ind]]))
            s_pos = np.where(g_rates==s_new)[0][0]
    return cell_types, g_rates,s_pos

## perform stepping stone alogrithm (migration, duplication, mutation) for each step
## and return new simulation box, recenter

def recenter(L, g_rates, K):
    shift = 0
    L_empty= np.append([K],np.zeros(len(g_rates)-1,dtype=float))
    while L[0,0]<int(.02*K):
        L=L[1:,:]
        shift+=1
    for i in range(shift):
        L=np.append(L,[L_empty],axis=0)
    return L


def update(L,g_rates,dist_key,K,r,alpha,mu):
    #demes = np.arange(len(L))
    L_tip = np.where(L[:,0]!=K)[0][-1]
    demes = np.arange(L_tip+2)
    #migration
    neighbors = rand_neighbors(demes)
    demes = np.arange(L_tip+2)
    #migration
    neighb = rand_neighbors(demes)

    L[neighb]= migration(L[neighb],n_allele,K)


    #duplication
    dup_deme = choice(demes,np.ones(len(demes))/len(demes))
    L[dup_deme] = duplication(L[dup_deme],K,P,n_allele)

    ##mutation
    mut_deme = choice(demes)
    cells, g_rates,s_pos = mutation(mu,alpha,dist_key, L[mut_deme],g_rates,K,r)
    counts = cells_to_counts(cells,g_rates)
    for i in range(len(g_rates)-len(L[0])):
        L = np.append(np.append(L[:,:(s_pos)],
                                np.vstack(np.zeros(len(L))),axis=1),L[:,(s_pos):],axis=1)
        #print(counts)
        #print(L[mut_deme])
        #print(L)
    L[mut_deme]=counts
        
    return L,g_rates


def initialize(K,r,alpha,mu):
    stand = odeint(standing_wave,[1,-(2*2**.5)/K],np.arange(70),args=(2*2**.5,1))[:,0]
    w_0 = (K*stand).astype(int)
    w_0 = w_0[w_0>1]
    L = np.vstack(((K-w_0),w_0)).T
    g_rates = np.array([0,r])

    ##initialize array
    L_empty= np.append([K],np.zeros(len(g_rates)-1))

    for i in range(70):
        L= np.append(L,[L_empty],axis=0)
    return L.astype(int), g_rates

def prob_mat(g_rates):
    P = np.ones((len(g_rates),len(g_rates)))
    P[0,:] = 1-g_rates
    return P 

def prune(L,g_rates):
    
    full_bools = np.where(np.sum(L,axis=0)!=0)[0][-1]
    L=L[:,:full_bools]
    g_rates = g_rates[:full_bools]
    return L,g_rates

## run one dimensional stepping stone for a given number of possible beneficil mutations, and generations   
def run_stepping_stone(n_gen,K,r,alpha,mu,dist_key = 'exponential'):
    func_args = [K,r,alpha,mu]
    ##initialize probability matrix
    c = 1
    move=10
    L, g_rates = initialize(*func_args)
    L_history=[L]
    #begin evolution
    count = 0
    for t in range(n_gen):
        for dt in range(K):
            L,g_rates = update(L,g_rates,dist_key,*func_args)
            
            L= recenter(L,g_rates,K)
        
        #L,g_rates =prune(L,g_rates)
        L_history.append(L)
        

        count+=1
    return L_history, g_rates


## run a two allele 1d stepping stone simulation, recording mutant establishment and fixation along simulation box
def fix_time(K,r,alpha,mu,thresh,dist_key , track):
    func_args = [K,r,alpha,mu]
    ##initialize probability matrix
    c = 1
    move=10
    L, g_rates = initialize(*func_args)
    L_history=[L]
    #begin evolution
    count = 0
    #begin evolution
    fixed=False
    t = 0

    fix_times = np.zeros(len(L))
    est_times = np.zeros(len(L))
    fb_1, eb_1 = (np.zeros(len(L)) == 1),(np.zeros(len(L)) == 1) 
    while not fixed:
        L,g_rates = update(L,g_rates,dist_key,*func_args)
        L= recenter(L,g_rates,K)
        wt_ind = np.where(g_rates==r)[0][0]
        fix_bools = L[:,wt_ind] < int(thresh*K)
        #est_bools = L[:,2] > int(1/(r*alpha))
        fixed= all(fix_bools)

        fix_times  = [max(i,time) for i, time in zip((fix_bools^fb_1)*t,fix_times)]
            #est_times  = [max(i,time) for i, time in zip((est_bools^eb_1)*t,est_times)]
        fb_1= fix_bools
        t+=1
    return L,fix_times,g

      

        