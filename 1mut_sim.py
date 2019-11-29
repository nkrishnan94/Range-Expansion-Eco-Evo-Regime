from evo_diffeqs import*
from tqdm import tqdm

##simulating fixation of single mutant
#parameter values
mu_range =1/(10**np.array([6,5,4,3]))
s_range = [.4,.7,1]
K_range=10**np.array( [2.9,3.2])
D_range=[1]
params = [mu_range,s_range,K_range,D_range]

#record data
data_exp= []

for mu in tqdm(mu_range):
    for s in s_range:
        for K in K_range:
            for D in D_range:
                print(mu,s,K)
                ##initial standing wave solution



                r = .1
                x_init=100
                dx =.5*(r)**.5
                x = np.linspace(0,x_init-dx,x_init/dx)
                b0= 1 
                db0 = -.005
                stand = odeint(standing_wave,[b0,db0],x,args=(1,1))[:,0]
                stand[stand<0]=0
                L_f = np.where(stand*K<1)[0][0]



                R=400
                dR =.5*(r)**.5
                R_space = np.linspace(0,R-dR,R/dR)


                #initial conditions

                b0= np.append(stand,np.zeros(len(R_space)-len(x)))
                b1 = np.zeros(len(R_space))
                b = np.array([b0,b1])
                dx=dR ##spatial resolution to be numerically solvedequation through time
               
                dt= 1/K
                t=0
                ##initialize arrays to record fixation times
                fb_1 =np.zeros(L_f)==1
                eb_1 =np.zeros(L_f)==1
                fix_times=[]
                est_times=[]
                fixed = False

                ##simulation until fixaiton along entire fron
                while not fixed:
                    #solve stochastich Fisher equation for on time step
                    db = sFisher_1max(b,R_space,dt,mu,s,r,K,D)
                    ##add new alleles to b vector
                    b_1 = b+dt*db  ##compute new population density per Euler-Murayama method
                    for i in range(len(b_1)):
                        b_1[i,(b_1[i]/(K*mu))<(1*10**-7)]=0
                                ##remove any hanging empty, vectors to keep it resonable size
                    ##record population
                    b = b_1

                    ### position where total population density goes to 0, or the "front end"
                    btot = np.sum(b,axis=0)
                    f_end = max(np.where((btot*K)<1)[0][0],L_f)


                    ##check where the wild type wave is exteinct along the front
                    front_bools = np.array((b[0][f_end-L_f:f_end]*K)<1)
                    est_bools = np.array((b[1][f_end-L_f:f_end]*K)>1)
                    fix_bools=front_bools*est_bools
                    ##record the time of any new fixations
                    fix_times.append((fb_1^fix_bools)*t)
                    fb_1 = fix_bools
                    fixed = all(fix_bools)
                    
                    ##check where the wild type wave is extinct along the front
                    est_bools = (b[1][f_end-L_f:f_end]*K)>1*mu
                    
                    est_times.append((eb_1^est_bools)*t)
                    eb_1 = est_bools
                    t+=1


                data_exp.append([np.array(fix_times)[:,:].max(axis=0)*dt,
                               np.array(est_times)[:,:].max(axis=0)*dt])
                np.save('data_exp.npy', np.array(data_exp))
                
np.save('data_exp.npy', np.array(data_exp))