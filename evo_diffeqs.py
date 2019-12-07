import numpy as np
from numpy import gradient as grad
from scipy.integrate import odeint
import scipy.integrate as integrate
from scipy.interpolate import interp1d




##standing wave solution of Fisher wave
def standing_wave(y0,x,D,rw):
    w = y0[0]      ##initial value for wave profile at x =0, i.e. w(x=0)
    z = y0[1]      ##initial value for rate of change of profile w.r.t. position x , at x=0 i.e. dw/dx(x=0)
    dwdx = z
    dzdx =(-2*((rw*D)**.5)*dwdx -w*rw*(1-w))/D
    
    return [dwdx,dzdx]


###diffusion equation for surfing probability
def surf_prob(y0,x,rw,rm,D,bfunc):
    u = y0[0]  ##initial value for probability at x =0, i.e. u(x=0)
    z = y0[1]  ##initial value for rate of change of profile w.r.t. position x , at x=0 i.e. du/dx(x=0)
    dudx = z
    dzdx =(-(2*((D*rw)**.5))*dudx+ u*(rm)*(1-bfunc(x))-u**2)/D
    return [dudx,dzdx]

##solv surfing probabiligty using ODEint
def surf_prob_solve(b,rw,rm,D, du0 ):
    ##generate a continuous function from initial standing wave
    bfunc = interp1d(range(len(b)),b, bounds_error=False, fill_value="extrapolate")
    
    ##find range of positions to integrate over
    x = np.linspace(0,len(b)-1,len(b))
    ##integarate
    u_x =odeint(surf_prob,[0,du0],x,args = (rw,rm,D,bfunc))
    return u_x

###stochastic differential equation for evolution of single mutant 
def sFisher_1max(b,R_space,dt,mu,s,r,K,D):
    a=s+1 ### a = rm/rw
    dbdt=[]  ### empty list to store change over time for eache allele
    ## iterate through each allele "i"
    for i in range(len(b)):
        bi = b[i]
        btot = np.sum(b,axis=0) #total population density w.r.t. x 
        dbidt= []  ###empty list to store each expression of SDE
        dbidt.append(D*grad(grad(bi,np.diff(R_space)[0]),np.diff(R_space)[0])) #diffusion
        dbidt.append((a**i) * bi*(1-btot)) ##growth
        ##noise as a Moran process: sqrt(Db(1-b)) where D = 2/NT_g where T_g is generation time or 1/r
        dbidt.append((((2*r*(a**i)*bi*[max(0,j) for j in (1-btot)])/K*(r)**.5)**.5) * np.random.normal(0,1,len(bi))) 
        #dbidt.append( (b[0]*mu) *(2*(i==1)-1)) ##mutation 
        #dbidt.append(b[0]*mu*(2*(i==1)-1)) ##mutation 
        dbidt.append(np.random.poisson(b[0]*mu)*(2*(i==1)-1)) ##mutation 
        #dbidt.append((1/K)*np.random.poisson(b[0]*mu*dt)*(2*(i==1)-1)) ##mutation 
        dbdt.append(np.sum(dbidt,axis=0).tolist())##change in  allale population due to diffusion, growth, genetic, drift, and mutation
        
    return np.array(dbdt)



def sFisher(b,R_space,mu,s,r,K,D):
    a = 1+s
    dbdt=[]
    ## chane in mutation from previous allele to current allel, beings at 0
    dbmu_ = np.zeros(len(b[0]))
    ## iterate through each allele "i"
    for i in range(len(b)):
        bi = np.array(b[i])
        btot = np.sum(b,axis=0) #total population density w.r.t. x 
        dbidt = []
        ##Stochastic fisher equation
        #dbidt = D*grad(grad(bi,dx),dx) + ((a)**i) * bi*(1-btot) +(((2*r*bi*[max(0,j) for j in (1-btot)])/K)**.5) * np.random.normal(0,1,len(bi))
        dbidt.append(D*grad(grad(bi,np.diff(R_space)[0]),np.diff(R_space)[0]))
        dbidt.append(((a)**i) * bi*(1-btot))
        
        dbidt.append((((2*r*(a**i)*bi*[max(0,j) for j in (1-btot)])/K*(r)**.5)**.5) * np.random.normal(0,1,len(bi))) 
        dbmu = - np.random.poisson(bi*mu)
        dbidt.append(dbmu) #beneficial mutation rate, "out" of current allele
        dbidt.append(dbmu_)
        dbdt.append(np.sum(dbidt,axis=0).tolist())
        #dbdt.append(dbidt+dbmu+dbmu_)##change in  allale population due to diffusion, growth, genetic, drift, and mutation
        dbmu_ = -dbmu #"int to next allele" of current allele
    dbdt.append(dbmu_) ## mutation into next allele, for which population density vector doesnt exist yet
    return np.array(dbdt)

def sFisher_solve(t,R,dR,b,func_args):
    dx=dR ##spatial resolution to be numerically solved
    K=func_args[-2] ##compact support, 1/K - if K*mu<1 this will prevent any mutant growth
    dt = t[1]-t[0] ## time step inferred from time vector
    hist = [] ##to record population density vectors at each time step
    hist.append(b)
    ## solve fisher equation through time
    for t_step in t:
        #solve stochastich Fisher equation for on time step
        db = sFisher(b,dx,*func_args)
        ##add new alleles to b vector
        for i in range(len(db)-len(b)):
            
            b.append(np.zeros(R))
       # print(len(np.array(b)[0]))
        b_1 = np.array(b)+dt*db  ##compute new population density per Euler-Murayama method
        for i in range(len(b_1)):
            b_1[i,b_1[i]<(10**-10)] =0 
        ##remove any hanging empty, vectors to keep it resonable size
        while  np.all(b_1[-1]==0):
            b_1 = b_1[:-1]
        ##record population
        hist.append(b_1)
        b = b_1.tolist()
        
    return hist

