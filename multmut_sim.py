from evo_diffeqs import*
from tqdm import tqdm
##simulating evolution of multiple competing mutant waves
##parameter values
mu_range =1/(10**np.array([2,2.15,2.3]))
s_range = [3,4,5]
K_range=10**np.array( [3.1,3.3])
D_range=[1]
params = [mu_range,s_range,K_range,D_range]

#record data

reps = 5
#record data
CII_exp= []
for r in tqdm(range(reps)):
    for mu in mu_range:
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
                    db0 = -.05
                    stand = odeint(standing_wave,[b0,db0],x,args=(1,1))[:,0]
                    stand[stand<0]=0
                    L_f = np.where(stand*K<1)[0][0]



                    R=300
                    dR =.5*(r)**.5
                    R_space = np.linspace(0,R-dR,R/dR)
                    b = np.append(stand,np.zeros(len(R_space)-len(x)))
                    

                    b = [b]
                    dx=dR ##spatial resolution to be numerically solvedequation through time
                    dt =1/K
                    t=0

                    ##simulation until fixaiton along entire fron
                    ##initialize arrays to record fixation times
                    fb_1 =np.zeros(L_f)==1
                    eb_1 =np.zeros(L_f)==1
                    fix_times=[]
                    est_times=[]
                    fixed = False
                    
                    ##simulation until fixaiton along entire fron
                    while not fixed:
                        #solve stochastich Fisher equation for on time step
                        db = sFisher(b,R_space,mu,s,r,K,D)
                        ##add new alleles to b vector
                        for i in range(len(db)-len(b)):
                
                             b.append(np.zeros(len(R_space)))
           # print(len(np.array(b)[0]))
                        b_1 = np.array(b)+dt*db  ##compute new population density per Euler-Murayama method
                        for i in range(len(b_1)):
                            b_1[i,(b_1[i])<(1/(K*100))]=0
            ##remove any hanging empty, vectors to keep it resonable size
                        while  np.all(b_1[-1]==0):
                            b_1 = b_1[:-1]
                                    ##remove any hanging empty, vectors to keep it resonable size
                        ##record population
                        b = b_1

                        ### position where total population density goes to 0, or the "front end"
                        btot = np.sum(b,axis=0)
                        f_end = max(np.where(btot*K<1)[0][0],L_f)


                        ##check where the wild type wave is exteinct along the front
                        front_bools = np.array((b[0][f_end-L_f:f_end]*K)<1)


                        ##record the time of any new fixations
                        fix_times.append((fb_1^front_bools)*t)
                        fb_1 = front_bools
                        fixed = all(front_bools)
                        
                        ##check where the wild type wave is extinct along the front
                        t+=1
                        if f_end > (4*L_f):
                            for i in range(len(b)):
                                b[i] = np.append(b[i][2*L_f:], np.zeros(2*L_f))
                        b =b.tolist()

                    ##record polygenecity of final population density through time

                    btot = np.sum(b,axis=0)
                    f_end = max(np.where(btot*K<1)[0][0],L_f)
                    b=np.array(b)
                    poly_bools = np.sum(b[:,f_end-L_f:f_end]*K>1,axis=0)
                    
                    CII_exp.append(poly_bools)
                    np.save('CII_exp.npy', np.array(CII_exp))
                    ##simulation until fixaiton along entire fron

    np.save('CII_exp.npy', np.array(CII_exp))