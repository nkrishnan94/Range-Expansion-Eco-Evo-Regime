import numpy as np
from numpy.random import choice
import random
import matplotlib.pyplot as plt
from scipy.integrate import odeint 
from numba import jit,njit
##standing wave solution of Fisher wave


def standing_wave(y0,x,D,rw):
    w = y0[0]      ##initial value for wave profile at x =0, i.e. w(x=0)
    z = y0[1]      ##initial value for rate of change of profile w.r.t. position x , at x=0 i.e. dw/dx(x=0)
    dwdx = z
    dzdx =(-2*((rw*D)**.5)*dwdx -w*rw*(1-w))/D ## fisher equation in comoving frame
    
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



## given  ab array with counts and an array withprobabilities return index from first array
# faster than np.random.choice for smallish arrays
@njit
def choice(options,probs):
    x = np.random.rand()
    cum = 0
    for i,p in enumerate(probs):
        cum += p
        ##sum of probability must be 1
        if x < cum:
            break
    return options[i]


@njit
def stepping_stone_nonvec(n_gen,## nunmber of gnerations
                K, ## population size
                r,
                alpha,## fitness landscape (Growthrates of each genotype (should be <<1))
                mu,
                L_init):  ## mutation rate
    

    ##initialize probability matrix
    #L = L_init
    L_empty = L_init[-1]
    L=L_init
    g_rates = np.array([0,r])
    n_allele = len(g_rates)-1
    P = np.ones((n_allele+1,n_allele+1))
    P[0,:] = 1 - g_rates
    
    ## list of allele number - pre-established so array doesnt have to be regenerated
    alleles = np.arange(n_allele+1)
    
    #slots for picked alleles each iteration to be stored, so array doesnt havent to be regerated each time
    picks = np.array([0,0,0,0])
    
    #sstore trace history
    #L_history= np.expand_dims(L,0)

    #for i in range(n_gen-1):
    #    L_history = np.concatenate((L_history, np.expand_dims(np.zeros(L.shape),0)))
    
    delta_x = 0
    for t in range(n_gen):
        for dt in range(K):
            
            #number of demes with a non empty particle (+2)
            n_demes = np.where(L[:,0]!=K)[0][-1] +2

            #pick adjacent demes tobe swapped, pick demes for duplication and mutation
            ind_1 = np.random.randint(0, n_demes)
            if (np.random.random() < .5 and ind_1 != 0) or ind_1 == n_demes - 1:
                ind_2 = ind_1 - 1
            else:
                ind_2 = ind_1 + 1
            neighb = np.array([ind_1, ind_2])
            dup_deme, mut_deme = np.random.randint(0,n_demes,2)


            #dmigration: pick two cells from each of the selected demes, and swap them
            for i in range(2):
                picks[i] = choice(alleles, L[neighb][i]/K)
                
            for inds in [[0,1],[1,0]]:
                L[neighb[inds[0]],picks[inds[0]]] -= 1
                L[neighb[inds[0]],picks[inds[1]]] += 1


            #duplication: pick two cells from the selected deme and echange first with copy of second according to
            #3 probability matrix
            for i in range(2,4):
                picks[i] = choice(alleles,L[dup_deme]/K)

            if P[picks[2],picks[3]] > np.random.random():
                L[dup_deme,picks[2]] += 1
                L[dup_deme,picks[3]] -= 1



            ##mutation
            mut_deme = np.random.randint(n_demes)
            picks[4] = choice(alleles,L[mut_deme]/K)
            #picks.append(choice(alleles,L[mut_deme]/K))
            if mu>np.random.random() and ( picks[4]>0):
                #3 only particles that are not empty spaces and are not the 'peak' in the landscape strucutre can mutate
                #if picks[4] != n_allele and picks[4] != 0:
                    ## mutate cell and fromat in terms of cell counts i.e. [empty cell count,...,chosen cell count]

                ##remove original cell and add mutated cell to cell counts
                #s_0 = g_rates[picks[4]]
                s_new = np.random.normal(r*alpha**picks[4],.001)
                g_rates = np.sort(np.append(g_rates,np.asarray(s_new)))
                s_pos = np.where(g_rates==s_new)[0][0]
                #print(s_pos)
                L = np.concatenate((L[:,:(s_pos)].T, np.expand_dims(np.zeros(len(L),dtype=np.dtype( np.int64)),0), L[:,(s_pos):].T)).T
                #L = np.concatenate((L[:,:(s_pos)].T,np.expand_dims(np.zeros(len(L)).astype(int),0),L[:,(s_pos):].T)).T 
                

                n_allele = len(g_rates)-1

                P = np.ones((n_allele+1,n_allele+1))
                
                P[0,:] = 1 - g_rates
                alleles = np.arange(n_allele+1)
                L_empty = np.array([K]+[0]*n_allele)
                L[mut_deme,picks[4]] -=1
                L[mut_deme,s_pos] +=1
                
                
        non_empty = np.sum(L,axis=0)!=0
        L= L[:,non_empty]
        L_empty = L_empty[non_empty]
        g_rates = g_rates[non_empty]

        ##track how many demes are to be omitted
        shift = 0
        while L[0,0]<1:
            L=L[1:,:]
            shift+=1

        delta_x+=shift
        #if L[0,0]<int(.02*K):
        #    shift = np.where(L[:,0]<int(.02*K))[-1][0]
        #    L = L[shift:,:]
        for i in range(shift):
            L=np.append(L,np.expand_dims(L_empty,0),axis=0)


        #L_history = np.concatenate((L_history, np.expand_dims(L,0)),axis=0)
        #L_history[t] = L
        
    return L,g_rates,delta_x


@njit
def fix_time_spatial(K, ## population size
                     r,
                     alpha,## fitness landscape (Growthrates of each genotype (should be <<1))
                     mu,
                     L_init,
                     thresh):  ## mutation rate
    

    ##initialize probability matrix
    #L = L_init
    L_empty = L_init[-1]
    L=L_init
    g_rates = np.array([0,r])
    n_allele = len(g_rates)-1
    P = np.ones((n_allele+1,n_allele+1))
    P[0,:] = 1 - g_rates
    
    ## list of allele number - pre-established so array doesnt have to be regenerated
    alleles = np.arange(n_allele+1)
    
    #slots for picked alleles each iteration to be stored, so array doesnt havent to be regerated each time
    picks = np.array([0,0,0,0])
    
    #sstore trace history
    #L_history= np.expand_dims(L,0)

    #for i in range(n_gen-1):
    #    L_history = np.concatenate((L_history, np.expand_dims(np.zeros(L.shape),0)))

    fixed=False
    muts_check = False
    del_check =False
    del_fix = False
    arise_time =0 
    t = 0

    while not fixed:
        
        #number of demes with a non empty particle (+2)
        n_demes = np.where(L[:,0]!=K)[0][-1] +2

        #pick adjacent demes tobe swapped, pick demes for duplication and mutation
        ind_1 = np.random.randint(0, n_demes)
        if (np.random.random() < .5 and ind_1 != 0) or ind_1 == n_demes - 1:
            ind_2 = ind_1 - 1
        else:
            ind_2 = ind_1 + 1
        neighb = np.array([ind_1, ind_2])
        dup_deme, mut_deme = np.random.randint(0,n_demes,2)


        #dmigration: pick two cells from each of the selected demes, and swap them
        for i in range(2):
            picks[i] = choice(alleles, L[neighb][i]/K)

        for inds in [[0,1],[1,0]]:
            L[neighb[inds[0]],picks[inds[0]]] -= 1
            L[neighb[inds[0]],picks[inds[1]]] += 1


        #duplication: pick two cells from the selected deme and echange first with copy of second according to
        #3 probability matrix
        for i in range(2,4):
            picks[i] = choice(alleles,L[dup_deme]/K)

        if P[picks[2],picks[3]] > np.random.random():
            L[dup_deme,picks[2]] += 1
            L[dup_deme,picks[3]] -= 1



        ##mutation
        mut_deme = np.random.randint(n_demes)
        picks[4] = choice(alleles,L[mut_deme]/K)
        #picks.append(choice(alleles,L[mut_deme]/K))
        if (mu>np.random.random()) and( picks[4]>0):
            #3 only particles that are not empty spaces and are not the 'peak' in the landscape strucutre can mutate
            #if picks[4] != n_allele and picks[4] != 0:
                ## mutate cell and fromat in terms of cell counts i.e. [empty cell count,...,chosen cell count]

            ##remove original cell and add mutated cell to cell counts

            s_new = np.random.normal(r*alpha**picks[4],.001)
            g_rates = np.sort(np.append(g_rates,np.asarray(s_new)))
            s_pos = np.where(g_rates==s_new)[0][0]
            #print(s_pos)
            L = np.concatenate((L[:,:(s_pos)].T, np.expand_dims(np.zeros(len(L),dtype=np.dtype( np.int64)),0), L[:,(s_pos):].T)).T
            #L = np.concatenate((L[:,:(s_pos)].T,np.expand_dims(np.zeros(len(L)).astype(int),0),L[:,(s_pos):].T)).T 


            n_allele = len(g_rates)-1

            P = np.ones((n_allele+1,n_allele+1))

            P[0,:] = 1 - g_rates
            alleles = np.arange(n_allele+1)
            L_empty = np.array([K]+[0]*n_allele)
            L[mut_deme,picks[4]] -=1
            L[mut_deme,s_pos] +=1

        ##track how many demes are to be omitted

        non_empty = np.sum(L,axis=0)!=0
        L = L[:,non_empty]
        L_empty = L_empty[non_empty]
        g_rates = g_rates[non_empty]
        

        while L[0,0]<1:
            L=L[1:,:]
            L=np.append(L,np.expand_dims(L_empty,0),axis=0)
      
            
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
        t+=1
        #L_history = np.concatenate((L_history, np.expand_dims(L,0)),axis=0)
        #L_history[t] = L
    return L,g_rates,t,arise_time, 


## run simulation until a fixation event occurs (fixation to some threshold 'thresh')





#Run the automaton
#Implements cell division. The division rates are based on the experimental data


    
