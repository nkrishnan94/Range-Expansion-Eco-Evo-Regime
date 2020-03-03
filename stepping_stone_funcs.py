import numpy as np
from numpy.random import choice

import random

from scipy.integrate import odeint

def choice(options,probs):
    x = np.random.rand()
    cum = 0
    for i,p in enumerate(probs):
        cum += p
        ##sum of probability must be 1
        if x < cum:
            break
    return options[i]


##generate a standing wave solution of fisher equations - i.e. an 'established' wave front

def standing_wave(y0,x,D,rw):
    w = y0[0]      ##initial value for wave profile at x =0, i.e. w(x=0)
    z = y0[1]      ##initial value for rate of change of profile w.r.t. position x , at x=0 i.e. dw/dx(x=0)
    dwdx = z
    dzdx =(-2*((rw*D)**.5)*dwdx -w*rw*(1-w))/D ## fisher equation in comoving frame
    
    return [dwdx,dzdx]

### given a deme return that deme and a random neighbor

def rand_neighbors(n_demes):
    ind_1 = np.random.randint(n_demes)
    if ind_1 == 0:
        ind_2 =1
    else:
        if ind_1 == n_demes:
            ind_2 = n_demes-1
        else:
            if np.random.randn()>.5 and ind_1 !=0:
                ind_2 = ind_1-1
            else:
                ind_2 = ind_1+1
    return np.array([ind_1,ind_2])



def fit_dist(r,alpha,dist_key):
    dist_dict = {'exponential':np.random.exponential,
                 'gamma':np.random.gamma,
                 'normal':np.random.normal}
    func = dist_dict[dist_key]
    if func ==np.random.normal:
        s_bar  = r*func(alpha,.01)
    else:
        s_bar  = r*func(alpha)
    return s_bar




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

def migration(cell_counts,g_rates,K):
    empty_cnt_1,empty_cnt_2 = np.zeros(len(g_rates)),np.zeros(len(g_rates))
    try:
     
        chosen_1 = choice(g_rates, cell_counts[0]/K)
        chosen_2 = choice(g_rates, cell_counts[1]/K)
    except:
        chosen_1 = choice(g_rates, cell_counts[0]/np.sum(cell_counts[0]))
        chosen_2 = choice(g_rates, cell_counts[1]/np.sum(cell_counts[0]))
    chosen_1 = np.where(chosen_1==g_rates)[0][0]
    chosen_2 = np.where(chosen_2==g_rates)[0][0]
    #print(np.where(chosen_1==g_rates))
    empty_cnt_1[chosen_1] =1
    empty_cnt_2[chosen_2] =1
    cell_counts[0]=cell_counts[0]- empty_cnt_1 + empty_cnt_2
    cell_counts[1]= cell_counts[1]+ empty_cnt_1 - empty_cnt_2
    return cell_counts
## from the cell list from the chosen d0eme
##pick *-\\\\\\\\\\\\\\\
#wo cells and replace one with a duplicate of the other according
##to transition matrix

def duplication(cell_counts,g_rates,K):
    picks = []
    for i in range(2):
        try:
            picks.append(choice(g_rates,
                              cell_counts/K))
        except:
            picks.append(choice(g_rates,
                              cell_counts/np.sum(cell_counts)))
            

    empty_cnt_1, empty_cnt_2 = np.zeros(len(g_rates)),np.zeros(len(g_rates))
    picks[0] = np.where(picks[0]==g_rates)[0][0]
    picks[1] = np.where(picks[1]==g_rates)[0][0]
    empty_cnt_1[picks[0]] =1
    empty_cnt_2[picks[1]] =1
    r= np.random.random()
    P=prob_mat(g_rates)
    #print(cell_types)
    if P[tuple(picks)]> r:
        cell_counts = cell_counts + empty_cnt_1 - empty_cnt_2
    return cell_counts


    
 ## from the cell list from the chosen deme
##pick a cell and mutate it with probability according to mutation rate

    

def mutation(mu,alpha,dist_key, cell_counts,g_rates,K,r):

    cell_types = np.repeat(g_rates,cell_counts.astype(int))
    pick_ind=choice(np.arange(K), np.ones(K)/K)
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
    

    #migration
    n_demes =  np.where(L[:,0]!=K)[0][-1]+2
    #migration
    neighb = rand_neighbors(n_demes)

    L[neighb]= migration(L[neighb],g_rates,K)


    #duplication
    dup_deme = np.random.randint(n_demes)
    L[dup_deme] = duplication(L[dup_deme],g_rates,K)

    ##mutation
    mut_deme = np.random.randint(n_demes)
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
def fix_time(K,r,alpha,mu,thresh,dist_key):
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

        #fix_times  = [max(i,time) for i, time in zip((fix_bools^fb_1)*t,fix_times)]
            #est_times  = [max(i,time) for i, time in zip((est_bools^eb_1)*t,est_times)]
        fb_1= fix_bools
        t+=1
    return L,g_rates

      

        