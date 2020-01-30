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
    cells = np.repeat(np.arange(n_allele+1),counts)
    return cells

## covert an array of all cells and their type to a list of counts for each cell type 
def cells_to_counts(cells,n_allele):
    counts = np.bincount(cells, minlength=n_allele+1)
    return counts
    
    
## from a list 2d array of the cell list from two neighboring demes
##pick two cells to be swapped at random and return resulting cell list
def migration(cells,K):
    cells_1 = cells[0]
    cells_2 = cells[1]
    pick_ind=choice(np.arange(K),2,replace= True)
    picks = np.array([np.arange(K) == pick_ind[i] for i in [0,1]])
    keep =  ~picks
    cells_1 = np.append(cells_1[keep[0]], cells_2[picks[1]])
    cells_2 = np.append(cells_2[keep[1]], cells_1[picks[0]])
    return np.array([cells_1,cells_2])

## from the cell list from the chosen deme
##pick two cells and replace one with a duplicate of the other according
##to transition matrix
def duplication(cells,K,P):
    pick_ind=choice(np.arange(K),2,replace= False)
    picks = np.array([np.arange(K) == pick_ind[i] for i in [0,1]])
    r= np.random.random()
    if P[tuple(cells[pick_ind])]> r:
        cells[pick_ind[1]] =cells[pick_ind[0]]
    return cells
    
 ## from the cell list from the chosen deme
##pick a cell and mutate it with probability according to mutation rate
def mutation(cells,mu,K,n_allele):
    pick_ind=choice(np.arange(K))
    r= np.random.random()
    if mu>r:
        if cells[pick_ind] != n_allele and cells[pick_ind] !=0:
            cells[pick_ind] = cells[pick_ind] +1
    return cells

## perform stepping stone alogrithm (migration, duplication, mutation) for each step
## and return new simulation box, recenter

def recenter(L,L_empty,K):
    shift = 0
    while L[0,0]<int(.02*K):
        L=L[1:,:]
        shift+=1
    for i in range(shift):
        L=np.append(L,[L_empty],axis=0)
    return L


def update(L,L_empty,P,K,n_allele,r,alpha,mu):
    
    demes = np.arange(len(L))
    #migration
    neighb = rand_neighbors(demes)
    cells = np.array(list(map(counts_to_cells, L[neighb],2*[n_allele] )))
    cells = migration(cells,K)
    counts =  np.array(list(map(cells_to_counts, cells,2*[n_allele] )))
    L[neighb] = counts

    #duplication
    dup_deme = choice(demes)
    cells = counts_to_cells(L[dup_deme],n_allele)
    cells = duplication(cells,K,P)
    counts = cells_to_counts(cells, n_allele)
    L[dup_deme] = counts

    ##mutation
    mut_deme = choice(demes)
    cells = counts_to_cells(L[mut_deme],n_allele)
    cells = mutation(cells,mu,K,n_allele)
    counts = cells_to_counts(cells, n_allele)

    L[mut_deme] = counts
    
    

    return L
def initialize(K,n_allele,r,alpha,mu):
    stand = odeint(standing_wave,[1,-(2*2**.5)/K],np.arange(70),args=(2*2**.5,1))[:,0]
    w_0 = (K*stand).astype(int)
    w_0 = w_0[w_0>1]

    
    ##initialize array
    L_empty= np.append([K],np.zeros(n_allele,dtype=int))
    L0 = np.zeros((len(w_0),n_allele+1),dtype=int)
    L0[:,1] = w_0
    L0[:,0] = K-w_0
    L= L0
    for i in range(50):
        L= np.append(L,[L_empty],axis=0)
    return L, L_empty


## run one dimensional stepping stone for a given number of possible beneficil mutations, and generations   
def run_stepping_stone(n_gen,K,n_allele,r,alpha,mu):
    func_args = [K,n_allele,r,alpha,mu]
    ##initialize probability matrix
    P = np.ones((n_allele+1,n_allele+1))
    P[0,1:] = 1-r*(alpha**np.arange(n_allele))
    c = 1
    move=10
    L, L_empty = initialize(*func_args)
    
    L_history=[L]
    #begin evolution
    count = 0
    for t in range(n_gen):
        for dt in range(K):
            L = update(L,L_empty,P,*func_args)
            L = recenter(L,L_empty,K)
        L_history = np.append(L_history,[L],axis=0)

        count+=1
    return L_history


## run a two allele 1d stepping stone simulation, recording mutant establishment and fixation along simulation box
def fix_time(K,n_allele,r,alpha,mu,thresh, track):
    func_args = [K,n_allele,r,alpha,mu]
    ##initialize probability matrix
    P = np.ones((n_allele+1,n_allele+1))
    P[0,1:] = 1-r*(alpha**np.arange(n_allele))
    
    L,L_empty = initialize(*func_args)
    L_history=[L]
    #begin evolution
    fixed=False
    t = 0
    if track:
        fix_times = np.zeros(len(L))
        est_times = np.zeros(len(L))
        fb_1, eb_1 = (np.zeros(len(L)) == 1),(np.zeros(len(L)) == 1) 
    while not fixed:
        L = update(L,L_empty,P,*func_args)
        L = recenter(L,L_empty,K)
        fix_bools = L[:,1] < int(thresh*K)
        est_bools = L[:,2] > int(1/(r*alpha))
        fixed= all(fix_bools)
        if track:
            fix_times  = [max(i,time) for i, time in zip((fix_bools^fb_1)*t,fix_times)]
            est_times  = [max(i,time) for i, time in zip((est_bools^eb_1)*t,est_times)]
            fb_1,eb_1 = fix_bools,est_bools
        t+=1
    if not track:
        return L
    else: 

        fix_times,est_times


