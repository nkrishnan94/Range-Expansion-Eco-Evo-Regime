import numpy as np
from numpy import gradient as grad
from scipy.integrate import odeint
import scipy.integrate as integrate
from scipy.interpolate import interp1d



##
def standing_wave(y0,x,D,rw):
    u = y0[0]
    z = y0[1]
    dudx = z
    dzdx =(-2*((rw*D)**.5)*dudx -u*rw*(1-u))/D
    return [dudx,dzdx]

###'simplest' cases
def simple_fisher(u,t,D, r,dt):
    ##classic 1D fisher equation with diffusion constant D, growth rate r, population density u 
    ### population density u  is scaled by carrying capacity
    dudt = D*grad(grad(u,dt),dt) + r* u*(1-u)
    return dudt


def surf_prob(y0,x,rw,rm,D,bfunc):
    u = y0[0]
    z = y0[1]
    
    
    
    dudx = z
    dzdx =(-(2*((D*rw)**.5))*dudx+ u*(rm)*(1-bfunc(x))-u**2)/D

    return [dudx,dzdx]



def surf_prob_solve(b,rw,rm,D, du0 ):
    bfunc = interp1d(range(len(b)),b, bounds_error=False, fill_value="extrapolate")
    x = np.linspace(0,len(b)-1,len(b))
    s =odeint(surf_prob,[0,du0],x,args = (rw,rm,D,bfunc))
    return s





    ##functions for modified Croze et al. chemotaxis equations, with stochastic growth 


#scaled diffusion constant as a function of agar concentration
def M_A(c,c_0,c_1,R_div):
    
    M = (1+np.exp((c-c_1)/c_0))**-1
    M/R_div
    return M

#scaled advection constant as a function of agar concentration
def X_A(c,c_0,c_1,I):
    
    X = I*((np.exp((c-c_1)/c_0))*((1+np.exp((c-c_1)/c_0))**-2))
    return X


#####mean cell growth for all cells at each position determined by random number from binomial distribution 
###centered around birth/death probabilities and size as cell number at give position
###whole numbers of cells determined from cell density wrt position scaled CC - the reciprocal of which
### is density corresponding to one cell
def cell_growth(b0,birth,death, CC):
    pop_arr = np.round((np.array(b0)*CC)).astype(int)
    b_rate = birth/(birth+death)
    births = np.random.binomial(pop_arr,b_rate)*(birth+death)
    deaths = np.random.binomial(pop_arr,1-b_rate)*(birth+death)
    return (births-deaths)/CC
    
#####mean cell growth for all cells at each position determined by random number from binomial distribution 
###centered around mutation rate and size as cell number at give position. pop size determined as for cell growth func    
def cell_mut(b0,mut_rate,CC):
    pop_arr = (np.array(b0)*CC).astype(int)
    muts = np.random.binomial(pop_arr,mut_rate)

    return muts/CC



def model_sde_5mut(z,t,R,R_div,c,c_0,c_1,delta,I,k_x,k_s,N,H,f,mu):
    ##wt, mutant populations, and substrate given as one length 4*R array due to odeint constraints
    ind =R*R_div
    b = z[0:ind]
    bm = z[ind:2*ind]
    bm2 = z[2*ind:3*ind]
    bm3 = z[3*ind:4*ind]
    bm4 = z[5*ind:6*ind]
    bm5 = z[6*ind:7*ind]
    s = z[7*ind:]
    
    ##compact support heaviside function height
    CC = (3.5*10**8 ) 
    
    
    dbdt = M_A(c,c_0,c_1,R_div)*grad(grad(b)) -delta*X_A(c,c_0,c_1,I)* grad(b *(k_x/(s+k_x)**2)*grad(s))+ cell_growth(b,rw*(s/(k_s+s)),rw*(b+bm+bm2+bm3+bm4+bm5),CC) - cell_mut(b,mu,CC)
    dbmdt = M_A(c,c_0,c_1,R_div)*grad(grad(bm)) -delta*X_A(c,c_0,c_1,I)* grad(bm *(k_x/(s+k_x)**2)*grad(s))+cell_growth(bm,rw*a*(s/(k_s+s)),rw*a*(b+bm+bm2+bm3+bm4+bm5),CC) + cell_mut(b,mu,CC) - cell_mut(bm,mu,CC)
    dbm2dt = M_A(c,c_0,c_1,R_div)*grad(grad(bm2)) -delta*X_A(c,c_0,c_1,I)* grad(bm2 *(k_x/(s+k_x)**2)*grad(s))+cell_growth(bm2,rw*a*a*(s/(k_s+s)),rw*a*a*(b+bm+bm2+bm3+bm4+bm5),CC) + cell_mut(bm,mu,CC)-cell_mut(bm2,mu,CC)
    dbm3dt = M_A(c,c_0,c_1,R_div)*grad(grad(bm3)) -delta*X_A(c,c_0,c_1,I)* grad(bm3 *(k_x/(s+k_x)**2)*grad(s))+cell_growth(bm3,rw*a*a*a*(s/(k_s+s)),rw*a*a*a*(b+bm+bm2+bm3+bm4+bm5),CC) + cell_mut(bm2,mu,CC) - cell_mut(bm3,mu,CC)
    dbm4dt = M_A(c,c_0,c_1,R_div)*grad(grad(bm4)) -delta*X_A(c,c_0,c_1,I)* grad(bm4 *(k_x/(s+k_x)**2)*grad(s))+cell_growth(bm4,rw*a*a*a*a*(s/(k_s+s)),(b+bm+bm2+bm3+bm4+bm5)*a*a*a*a*rw,CC) + cell_mut(bm3,mu,CC) -cell_mut(bm4,mu,CC)
    dbm5dt = M_A(c,c_0,c_1,R_div)*grad(grad(bm5)) -delta*X_A(c,c_0,c_1,I)* grad(bm5 *(k_x/(s+k_x)**2)*grad(s))+cell_growth(bm5,rw*a*a*a*a*a*(s/(k_s+s)),(b+bm+bm2+bm3+bm4+bm5)*rw*a*a*a*a*a,CC) + cell_mut(bm4,mu,CC) -cell_mut(bm5,mu,CC)
                                                
    dsdt = N*grad(grad(s)) - H*(s/(s+k_s))*(rw*b+rw*a*bm+rw*a*a*bm2)+rw*a*a*a*bm3+rw*a*a*a*a*f*bm4+rw*a*a*a*a*a*bm5
    
    dzdt = [dbdt,dbmdt,dbm2dt,dbm3dt,dbm4dt,dbm5dt,dsdt]
    return np.concatenate(dzdt)

### Deterministic Chemotaxis per Croze et al. with stochastic growth
def model_sde_2mut(z,t,R,R_div,c,c_0,c_1,delta,I,k_x,k_s,N,H,rw,a,mu,K):
    ##wt, mutant populations, and substrate given as one length 4*R array due to odeint constraints
    z[z<0] = 0
    ind =R*R_div
    b = z[0:ind]
    bm = z[ind:2*ind]
    bm2 = z[2*ind:3*ind]
    s = z[3*ind:]
    
    ##compact support heaviside function height
    CC = (3.5*10**8 ) 
    
    
    dbdt = M_A(c,c_0,c_1,R_div)*grad(grad(b)) -delta*X_A(c,c_0,c_1,I)* grad(b *(k_x/(s+k_x)**2)*grad(s))+ b*rw*((s/(k_s+s)) -b-bm-bm2) -b*mu +np.random.normal(np.zeros(len(b)),abs(b*abs(((2*b*(1-b-bm-bm2))/K))**.5))


    dbmdt = M_A(c,c_0,c_1,R_div)*grad(grad(bm)) -delta*X_A(c,c_0,c_1,I)* grad(bm *(k_x/(s+k_x)**2)*grad(s))+bm*rw*a*((s/(k_s+s)) -b-bm-bm2) +b*mu-bm*mu + np.random.normal(np.zeros(len(bm)),abs(bm*abs(((2*bm*(1-b-bm-bm2))/K))**.5))
    dbm2dt = M_A(c,c_0,c_1,R_div)*grad(grad(bm2)) -delta*X_A(c,c_0,c_1,I)* grad(bm2 *(k_x/(s+k_x)**2)*grad(s))+bm*rw*a*a*((s/(k_s+s)) -b-bm-bm2) +bm*mu  + np.random.normal(np.zeros(len(bm2)),abs(bm2*abs(((2*bm2*(1-b-bm-bm2))/K))**.5))
                                                
    dsdt = N*grad(grad(s)) - H*(s/(s+k_s))*(rw*b+rw*a**bm+rw*a*a*bm2)
    
    dzdt = [dbdt,dbmdt,dbm2dt,dsdt]
    return np.concatenate(dzdt)


def model_sde_1mut(z,t,R,R_div,c,c_0,c_1,delta,I,k_x,k_s,N,H,rw,a,mu,K):
    ##wt, mutant populations, and substrate given as one length 4*R array due to odeint constraints
    z[z<0] = 0
    ind =(R-1)*R_div+1
    b = z[0:ind]
    bm = z[ind:2*ind]
    s = z[2*ind:]
    
    ##compact support heaviside function height
    CC = (3.5*10**8 ) 
    
    
    dbdt = M_A(c,c_0,c_1,R_div)*grad(grad(b)) -delta*X_A(c,c_0,c_1,I)* grad(b *(k_x/(s+k_x)**2)*grad(s))+ b*rw*((s/(k_s+s)) -b-bm) -b*mu + np.random.normal(np.zeros(len(b)),abs(b*abs(((2*b*(1-b-bm))/K))**.5))

    dbmdt = M_A(c,c_0,c_1,R_div)*grad(grad(bm)) -delta*X_A(c,c_0,c_1,I)* grad(bm *(k_x/(s+k_x)**2)*grad(s))+bm*rw*a*((s/(k_s+s)) -b-bm) +b*mu-bm*mu + np.random.normal(np.zeros(len(bm)),abs(b*abs(((2*bm*(1-b-bm))/K))**.5))
                                                
    dsdt = N*grad(grad(s)) - H*(s/(s+k_s))*(rw*b+a*rw*bm)
    
    dzdt = [dbdt,dbmdt,dsdt]
    return np.concatenate(dzdt)


####functions to solve above sde



def sde_model_solve_5mut(R, R_div,c,c_0,c_1,delta,I,K_x, K_s,N, H,rw,a,mu,b0,s0,t):
    N/R_div

    bm0,bm20,bm30,bm40,bm50 = 5*[np.zeros((R-1)*R_div+1)]
    z  = np.concatenate([b0,bm0,bm20,bm30,bm40,bm50,s0])
    results = []
    results.append(z)
    dt = t[1]-t[0]
    for i in t:
        results.append(results[-1] +dt*model_sde_2mut(results[-1], 0,R,R_div,c,c_0,c_1,delta,I,K_x,K_s,N,H,rw,a,mu))
        results[-1][results[-1]< 0]= 0  
        
    return results



def sde_model_solve_2mut(R, R_div,c,c_0,c_1,delta,I,K_x, K_s,N, H,rw,a,mu,b0,s0,t,L_f,K):

    N/R_div

    bm0,bm20 = 2*[np.zeros((R-1)*R_div+1)]
    z  = np.concatenate([b0,bm0,bm20,s0])
    results = []
    results.append(z)
    dt = t[1]-t[0]

    for i in t:
        btot = results[-1][0:R]+results[-1][R:2*R]+results[-1][2*R:3*R]
        if len(btot[btot*K>1])< L_f:
            mut_rate = 0
        else:
            mut_rate = mu

        results.append(results[-1] +dt*model_sde_2mut(results[-1], 0,R,R_div,c,c_0,c_1,delta,I,K_x,K_s,N,H,rw,a,mut_rate,K))
        results[-1][results[-1]< 0]= 0  
        
    return results


def sde_model_solve_1mut(R, R_div,c,c_0,c_1,delta,I,K_x, K_s,N, H,rw,a,mu,b0,s0,t,L_f,K):
    N/R_div

    bm0 = np.zeros((R-1)*R_div+1)
    z  = np.concatenate([b0,bm0,s0])
    results = []
    results.append(z)
    dt = t[1]-t[0]
    for i in t:
        btot = results[-1][0:R]+results[-1][R:2*R]+results[-1][2*R:3*R]

        results.append(results[-1] +dt*model_sde_1mut(results[-1], 0,R,R_div,c,c_0,c_1,delta,I,K_x,K_s,N,H,rw,a,mu,K))
        results[-1][results[-1]< 0]= 0
        
    return results

