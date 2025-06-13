# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 10:07:40 2024

@author: acimino001

solves the transient flow inside a circular tube of radius r0 

 /////////////////////////////////
         ^                        
 ^r      |                         
 |       |r0
 |-->x   |



//////////////////////////////////

du/dt=nu d^2u/dr^2+ nu 1/r du/dr -1/rho dp/dx

dp/dx is  an arbitrary function of time and space, 


BCS u(r=r0, t)=0

    du/dr(r=0, t)=0
    u1(r,0)=-u_ss


Solution is obtained using separation of variables

u(r, t)=V_j(r)T_j(t)

and the pressure term is implemented using modal decomposition


dpdx = g_j(t)*V_j(y)

////////////////////////////////////////////////////////////////////////
naming convention for variables and functions

_s means symbolic variable
_n means numerical variable
_v means vector of variables or numpy array

"""



import numpy as np
import math as m
import scipy.integrate as integrate
import scipy.optimize as opt
from scipy.special import jv
import gen_unst_poiseuille_aux_axisym as auxaxisym






""" symbolic solution
proposing a solution of the form 

u(t,r)=T(t)*V(r)

we propose 

V(r)= A_i* Jo(lambda_i * r/ r0) since it satisfies the BCs
where J0 is bessel's function of the first kind

and arbitrary IC

u(0, y)=u_0(y)

"""



import sympy as sym


#symbolic indep variables, constants and params (all positive)
r_s,t_s,s,  r0_s, lambda_s, Omega_s, nu_s= sym.symbols('r t  s  r0 λ_i Ω ν', positive=True, real= True)
#symbolic coefficients and functions
A_s,V_s,T_s=sym.symbols('A V T' , nonzero=True)
C_1_s,g_s, dpdx_s, u_s, K_s, u_0s, u_max_s= sym.symbols(' C_1 g dpdx u K u_0 u_max') # sybolic functions and coefficients
#integer symbolic variables
n_s, k_s = sym.symbols('n k', integer = True)


""" verification cases"""
#########  test function for a known case -dpdx=K*cos(omega*t), as a function of s, an integration
# variable

# u_0s= sym.re(sym.I*K_s/Omega_s*( sym.besselj(0, r_s*sym.sqrt(-sym.I* Omega_s/nu_s ))/sym.besselj(0, r0_s*sym.sqrt(-sym.I * Omega_s/nu_s))-1))# u_0s=0
# dpdx_s=K_s*sym.cos(Omega_s*s)#*sym.exp(-5*s) # pressure gradient function


##########    test function for a known case : sudden impulse in dpdx  #######
### dpdx=K  constant, complete solution (transient +steady state)

# u_0s=0
# dpdx_s=K_s

##############5     transient solution for the starting flow
# u_0s=-u_max_s*(1-(r_s/r0_s)**2)
# dpdx_s=0

####### pressure gradient with sin^2  and zero IC( no analytical solution), compared with fluent's numerical solution
u_0s= 0
dpdx_s=K_s*sym.sin(Omega_s*s)**2#*sym.exp(-5*s) # pressure gradient function


print("initial condition")
sym.pprint(sym.simplify(u_0s), num_columns=300)

""" spatial modes and modal decomposition of the pressure gradient function"""
#k=0

#modes for the spatial solution

V_s=A_s*sym.besselj(0,lambda_s*r_s/r0_s) #proposed spatial solution


#orthogonality factor 
Orth_factor=sym.integrate(r_s*V_s**2,(r_s, 0, r0_s))
# application of boundary condition u(r=r0)=0, which implies sym.besselj(0, lambda_s )=0
Orth_factor=Orth_factor.subs(sym.besselj(0, lambda_s ), 0)

print("Orth factor")
sym.pprint(Orth_factor)

#☻transformed pressure gradient function for modal decomposition
print("g_s")
g_s=1/Orth_factor*sym.integrate(r_s *dpdx_s*V_s, (r_s, 0, r0_s))
# *r0_s**2/nu_s
sym.pprint(g_s , num_columns=200)
"""temporal solution of the variable separation ( for zero IC) """


def ex(e,x):
    
    """ function that expands an expression only wrt to a given factor"""
    mults = [i for i in e.atoms(sym.Mul) if not i.has(x)] #searchs for evey factor which contains the variable x
    reps = dict(zip(mults,[sym.Dummy() for i in mults])) #creates a dictionary with temporary symbolic variables as items
    # zip is like a cartesian product of two iterables
    return sym.expand(e.xreplace(reps)).subs([(v,k) for k,v in reps.items()])




#T_s=sym.exp(-nu_s*lambda_s**2*t_s)*sym.integrate(g_s*sym.exp(nu_s*lambda_s**2*s), (s, 0, t_s))
T_s=sym.integrate(g_s*sym.exp(-nu_s*(lambda_s/r0_s)**2*(t_s-s)), (s, 0, t_s))
T_s=sym.simplify(T_s)
# T_s=sym.expand_power_exp(T_s)


# procedure to eliminate terms with positive exponentials
#that  produce overflows during the numerical evaluation


#terms_exp=  [i for i in T_s.atoms(sym.Mul) if  i.has(sym.exp)] # 

exp0= list(T_s.atoms(sym.exp)) #list with all exponentials in the expression

args_exps=[i.args[0] for i in exp0] #extracts the argument of every exponential

exp_min=sym.Min(*args_exps) #searches for the smallest exponent
T_s=ex(T_s,sym.exp(exp_min)) #expand all terms wrt to the smallest exponent
T_s=sym.powsimp(T_s, deep=True)
"""temporal  solution without initial condition
"""
print("T_s wo IC")
sym.pprint(T_s , num_columns=200)
#T_s=sym.simplify(T_s)

# hardcoded solution for oscillating flow
#T_s=h_s**2/A_s*(-nu_s*sym.pi**2*(k_s + 1/2)**2*sym.exp(-nu_s*sym.pi**2*t_s*(2*k_s + 1)**2/(4*h_s**2)) + nu_s*sym.pi**2*(k_s + 1/2)**2*sym.cos(Omega_s*t_s) + h_s**2*Omega_s*sym.sin(Omega_s*t_s))/(nu_s**2*(k_s + 1/2)**4*sym.pi**4 + Omega_s**2*h_s**4)

"""constant C_1 for a given initial condition"""

C_1_s=sym.re( 1/Orth_factor*sym.integrate(r_s*u_0s*V_s, (r_s, 0, r0_s)))
print("done")

# substitution of constants ( uncomment if running the pulsed solution)
# C_1_s = C_1_s.subs ( J_1i, sym.sin(B_1i*h_s)**2*sym.sinh(B_1i*h_s)**2+sym.cos(B_1i*h_s)**2*sym.cosh(B_1i*h_s)**2)
# C_1_s = C_1_s.subs ( B_1i , sym.sqrt(Omega_s/2/nu_s))
C_1_s=sym.simplify(C_1_s)

print("C_1")
sym.pprint((C_1_s), num_columns=300)

"""complete temporal  solution
initial condition and temporal variation of the pressure gradient
"""

T_s= T_s+ C_1_s*sym.exp(-nu_s*(lambda_s/r0_s)**2*(t_s))
print("T_s")
sym.pprint(T_s, num_columns=500)
#T_s=T_s.subs(lambda_s, sym.pi*(2*k_s+1)/2/h_s)

""" complete solution"""

u_s=T_s*V_s#.subs(lambda_s, sym.pi*(2*k_s+1)/2/h_s)
u_s=sym.simplify(u_s)
print("u_s")


sym.pprint(u_s, num_columns=500)

""" numerical calculations"""



nt=10 # number of terms of the solution  series

""" numerical values of variables for  evaluation"""



Deltap_n=-50 #pa
L_n=1 #[m]
r0_n=10e-3
mu_n=1.00e-3
rho_n=1000


dpdx_n=Deltap_n/L_n



# parameters for the oscillating pressure gradient
Period=30 ## Period of the pressure pulse [s]
Omega_n=2*np.pi/Period # # frequency of the pressure pulse [1/s]
K_n= -dpdx_n/rho_n#  amplitude of the pressure oscilation


nu_n=mu_n/rho_n

#max velocity (at r=0, t->infinity) for the starting flow
u_max_n=r0_n**2/4/mu_n*K_n*rho_n #dpdx_n

#tau=np.log(2)/(np.pi/2/h_n)**2/nu_n



# number of space and time discretizations
nDr=11
nDt=121
tf=2*Period


nterms=10
# #arrays of discretized space and time

t_v=np.linspace(0, tf, nDt)

r_v=np.linspace(0, r0_n, nDr)

u_sol=np.zeros((nDr, nDt))
Q_sol=np.zeros(( nDt))
# u_ss_sol=u_ss(y_v)


lambda_v=[None]*nterms # empty list of roots of J0

for i in range(nterms):
    
    # first guess for the ith root
    lambda_0v=(4*(i+1)-1)*np.pi/4
    lambda_v[i]=opt.fsolve(lambda x: jv(0, x), lambda_0v)[0]

#transient solution for a starting flow with a step pressure gradient
u_num_trans=sym.lambdify([r_s,t_s,K_s,k_s, r0_s, nu_s, u_max_s, Omega_s, lambda_s], u_s)
for i in range(nDt):
#     Q_sol[i]=rho*2*integrate.quad(lambda x: u(x, i*tf/nDt), 0,h)[0]
    for j in range(nDr):
        u_sol[j,i]=0#u_ss(y_v[j])
        for k in range(nterms):
            u_sol[j,i]= u_sol[j,i]+u_num_trans(r_v[j], t_v[i] ,K_n ,k, r0_n, nu_n, u_max_n, Omega_n, lambda_v[k])

Q_sol_gen=np.zeros(( nDt))
for i in range(nDt):
    Q_sol_gen[i] = rho_n*2*np.pi*integrate.trapezoid((u_sol[:,i]*r_v[:]), x=r_v)

# """ trajectory determination """

# def u_tray(t,x):
#     """ form of the velocity eq to be used with the ode solver"""
#     return u(y_0, t)

# x_tray=integrate.solve_ivp(u_tray,[0 ,tf], [0] )
""" data from fluent """

import os 

# filename = "fluent_files/start/ax/mass_flow_start_ax_rfile.out"
# filename = "fluent_files/pulsed/ax/mass_flow_axisym_pulsed_rfile.out"
filename = "fluent_files/sinsq/axisym/mass_flow_sinsq_ax_rfile.out"


if not os.path.isfile(filename):
    print('File does not exist.')

else:
# try:
    # Open the file and read its content.
    with open(filename) as f:
        content = f.read().splitlines()
    fluent_massflow=np.zeros((len(content)-3, 2))

    # Display the file's content line by line.
    for i in range(3,len(content)):
        values=content[i].split()
        fluent_massflow[i-3,:] = [float(values[2]), float(values[1])]
# except:
#     print('File does not exist.')

#filename = "fluent_files/v_prof_start_t01"

def fluent_velprof(file):
    """opens a profile file from fluent with velocity profiles and converts 
    them into a numpy array"""
    if not os.path.isfile(file):
        print('File does not exist.')

    else:
        # try:
            # Open the file and read its content.
        with open(file) as f:
            content = f.read().splitlines()
    
            values_vt0=[] #list with values of the lines in the file

    # search the file's content line by line, not considering the first 3 lines.
            for i in range(4,17):
                values_vt0.append(content[i].split())
            for i in range(20,33):
                values_vt0.append(content[i].split())
            for i in range(36,49):
                values_vt0.append(content[i].split())
                #a  numpy array for every line where there is data storaged
            # print(values_vt0)
            fluent_vel=np.zeros((12, 2, 3))
            data_buffer1=np.zeros((12, 2))
            data_buffer2=np.zeros((12, 2))
            data_buffer3=np.zeros((12, 2))
                # data_buffer4=np.zeros((11, 2))
                # fluent_vel_x01=np.zeros((11, 2))
                # fluent_vel_x03=np.zeros((11, 2))
                # fluent_vel_x05=np.zeros((11, 2))
    
            for i in range(12):
                data_buffer1 [i,:] = [float(values_vt0[i][1]), float(values_vt0[i][0])]
                data_buffer2 [i,:] = [float(values_vt0[12+i][1]) , float(values_vt0[12+i][0])]
                data_buffer3 [i,:] = [float(values_vt0[24+i][1]), float(values_vt0[24+i][0])]
                # data_buffer4 [i,:] = [float(values_vt0[57+i][1])-h_n, float(values_vt0[57+i][0])]
       
            #fluent_vel= fluent_vel[fluent_vel[:, 0,:].argsort(axis=1)]
            # fluent_vel[:,:,1]= fluent_vel[fluent_vel[:, 0,1].argsort(), 1]
            # fluent_vel[:,:,2]= fluent_vel[fluent_vel[:, 0,2].argsort(),2]
            # fluent_vel[:,:,3]= fluent_vel[fluent_vel[:, 0,3].argsort(),3]
            data_buffer1= data_buffer1[data_buffer1[:,0].argsort()]
            data_buffer2= data_buffer2[data_buffer2[:,0].argsort()]
            data_buffer3= data_buffer3[data_buffer3[:,0].argsort()]
            # data_buffer4= data_buffer4[data_buffer4[:,0].argsort()]
            
            fluent_vel[:,:, 0] =data_buffer1
            fluent_vel[:,:, 1] =data_buffer2
            fluent_vel[:,:, 2] =data_buffer3
            # fluent_vel[:,:, 3] =data_buffer4
        return (fluent_vel)
    
# filename = "fluent_files/start/ax/vel_prof_start_ax_t10.xy"
# fluent_vel_t10=fluent_velprof(filename)

# filename = "fluent_files/start/ax/vel_prof_start_ax_t20.xy"
# fluent_vel_t20=fluent_velprof(filename)
# filename ="fluent_files/start/ax/vel_prof_start_ax_t30.xy"
# fluent_vel_t30=fluent_velprof(filename)
# filename ="fluent_files/start/ax/vel_prof_start_ax_t60.xy"
# fluent_vel_t60=fluent_velprof(filename)


# filename = "fluent_files/pulsed/ax/vel_prof_puls_ax_t5.xy"
# fluent_vel_t5=fluent_velprof(filename)
# filename = "fluent_files/pulsed/ax/vel_prof_puls_ax_t10.xy"
# fluent_vel_t10=fluent_velprof(filename)

# filename = "fluent_files/pulsed/ax/vel_prof_puls_ax_t20.xy"
# fluent_vel_t20=fluent_velprof(filename)
# filename ="fluent_files/pulsed/ax/vel_prof_puls_ax_t30.xy"
# fluent_vel_t30=fluent_velprof(filename)

filename = "fluent_files/sinsq/axisym/vel_prof_t5_sinsq_ax.xy"
fluent_vel_t5=auxaxisym.fluent_velprof(filename)
filename = "fluent_files/sinsq/axisym/vel_prof_t10_sinsq_ax.xy"
fluent_vel_t10=auxaxisym.fluent_velprof(filename)

filename = "fluent_files/sinsq/axisym/vel_prof_t20_sinsq_ax.xy"
fluent_vel_t20=auxaxisym.fluent_velprof(filename)
filename ="fluent_files/sinsq/axisym/vel_prof_t30_sinsq_ax.xy"
fluent_vel_t30=auxaxisym.fluent_velprof(filename)
# filename ="fluent_files/start/vel_prof_start_ax_t60.xy"
# fluent_vel_t60=fluent_velprof(filename)



""" plots """


import matplotlib.pyplot as plt

# plt.figure(1)

# # for i in range(0, nDt, 5):
# #     plt.plot(y_v, u_sol[:, i], label=str(i*tau/nDt))
# plt.plot(y_v,u_sol[:,0], label='T=0' )
# plt.plot(y_v,u_sol[:,m.floor(nDt/8)], label='T=Per/4' )
# plt.plot(y_v,u_sol[:,m.floor(nDt/4)], label='T=Per/2' )
    
# #plt.plot(y_v, u_ss_sol, '-.r', label='steady state')
# plt.legend()
        
# plt.figure(2)

# plt.plot(t_v, u_sol[0, :], label='u_max')
# plt.plot(t_v, Q_sol[ :], label='mass_flow')
# plt.legend()

# plt.figure(3)

# plt.plot(x_tray.t, x_tray.y[0], label='x-tray')

# plt.legend()


"""verification solutions """

[u_mort, Q_mort] = auxaxisym.u_osc_num(nDr, nDt,  rho_n ,K_n, Omega_n,  mu_n, r0_n, tf, dpdx_n, nterms, lambda_v)
[u_start, u1_start, Q_start, u_ss_start] = auxaxisym.u_start_num(nDr, nDt,  rho_n,u_max_n,  mu_n, r0_n, tf, dpdx_n, nterms, lambda_v)


plt.figure(1)

# plt.plot(r_v,u_sol[:,0],'g', label='T=0' )
plt.plot(r_v,u_sol[:,np.where(np.abs(t_v-5)<=0.1)[0]],'c', label='t=5' )
plt.plot(r_v,u_sol[:,np.where(np.abs(t_v-10)<=0.1)[0]],'b', label='t=10' )
plt.plot(r_v,u_sol[:,np.where(np.abs(t_v-20)<=0.1)[0]],'r', label='t=20' )
plt.plot(r_v,u_sol[:,np.where(np.abs(t_v-30)<=0.1)[0]],'y', label='t=30' )
# plt.plot(r_v,u_sol[:,np.where(np.abs(t_v-60)<=0.1)[0]],'--m', label='t=60' )


# plt.plot(r_v,u_sol[:,0],'g', label='T=0' )
# plt.plot(r_v,u_sol[:,m.floor(nDt/8)],'b', label='T=Per/4' )
# plt.plot(r_v,u_sol[:,m.floor(nDt/4)],'r', label='T=Per/2' )
# plt.plot(r_v,u_sol[:,m.floor(nDt/16)],'y', label='T=Per/8' )

# plt.plot(r_v, u_ss_start, '--c', label='steady state')

# plt.plot(r_v,u_start[:,0],'-g', label='t=0  start' )
# plt.plot(r_v,u_start[:,np.where(np.abs(t_v-5)<=0.1)[0]],'*c', label='t=5 start' )
# plt.plot(r_v,u_start[:,np.where(np.abs(t_v-10)<=0.1)[0]],'*b', label='t=10 start' )
# plt.plot(r_v,u_start[:,np.where(np.abs(t_v-20)<=0.1)[0]],'*r', label='t=20  start' )
# plt.plot(r_v,u_start[:,np.where(np.abs(t_v-30)<=0.1)[0]],'*y', label='t=30 start' )
# plt.plot(r_v,u_start[:,np.where(np.abs(t_v-60)<=0.1)[0]],'*m', label='t=60 start' )

# plt.plot(r_v,u1_start[:,np.where(np.abs(t_v-5)<=0.1)[0]],'*c', label='t=5 start unst' )
# plt.plot(r_v,u1_start[:,np.where(np.abs(t_v-10)<=0.1)[0]],'*b', label='t=10 start unst' )
# plt.plot(r_v,u1_start[:,np.where(np.abs(t_v-20)<=0.1)[0]],'*r', label='t=20  start unst' )
# plt.plot(r_v,u1_start[:,np.where(np.abs(t_v-30)<=0.1)[0]],'*y', label='t=30 start unst' )
# plt.plot(r_v,u1_start[:,np.where(np.abs(t_v-60)<=0.1)[0]],'*m', label='t=60 start unst' )


# plt.plot(r_v,u_mort[:,np.where(np.abs(t_v-5)<=0.1)[0]],'*c', label='t=5 puls' )
# plt.plot(r_v,u_mort[:,np.where(np.abs(t_v-10)<=0.1)[0]],'*b', label='t=10 puls' )
# plt.plot(r_v,u_mort[:,np.where(np.abs(t_v-20)<=0.1)[0]],'*r', label='t=20  puls' )
# plt.plot(r_v,u_mort[:,np.where(np.abs(t_v-30)<=0.1)[0]],'*y', label='t=30 puls' )
# plt.plot(r_v,u_mort[:,np.where(np.abs(t_v-60)<=0.1)[0]],'*m', label='t=60 puls' )




plt.plot(fluent_vel_t5[:,0, 0], fluent_vel_t5[:,1,0],'*c', label='t=5  x=0.1 fluent' )
# plt.plot(fluent_vel_t5[:,0, 1], fluent_vel_t5[:,1, 1],'+c', label='t5  x=0.3 fluent' )
# plt.plot(fluent_vel_t5[:,0, 2], fluent_vel_t5[:,1,2],'.c', label='t5  x=0.0 fluent' )



plt.plot(fluent_vel_t10[:,0, 0], fluent_vel_t10[:,1,0],'*b', label='t=10  x=0.1 fluent' )
# plt.plot(fluent_vel_t10[:,0, 1], fluent_vel_t10[:,1, 1],'+b', label='t10  x=0.3 fluent' )
# plt.plot(fluent_vel_t10[:,0, 2], fluent_vel_t10[:,1,2],'.b', label='t10  x=0.0 fluent' )

plt.plot(fluent_vel_t20[:,0, 0], fluent_vel_t20[:,1, 0],'*r', label='t=20 x=0.1 fluent' )
# plt.plot(fluent_vel_t20[:,0,1], fluent_vel_t20[:,1, 1],'+r', label='t20  x=0.3 fluent' )
# plt.plot(fluent_vel_t20[:,0, 2], fluent_vel_t20[:,1,2],'.r', label='t20 x=0.0 fluent' )

plt.plot(fluent_vel_t30[:,0, 0], fluent_vel_t30[:,1,0],'*y', label='t=30 x=0.1 fluent' )
# plt.plot(fluent_vel_t30[:,0, 1], fluent_vel_t30[:,1,1],'+y', label='t30 x=0.3 fluent' )
# plt.plot(fluent_vel_t30[:,0, 2], fluent_vel_t30[:,1,2],'.y', label='t30 x=0.0 fluent' )

# plt.plot(fluent_vel_t60[:,0, 0], fluent_vel_t60[:,1,0],'*m', label='t=60 x=0.1 fluent' )
# plt.plot(fluent_vel_t60[:,0, 1], fluent_vel_t60[:,1,1],'+m', label='t60 x=0.3 fluent' )
# plt.plot(fluent_vel_t60[:,0, 2], fluent_vel_t60[:,1,2],'.m', label='t60 x=0.0 fluent' )

plt.xlabel('r')
plt.ylabel('u[m/s]')
# plt.legend()
plt.legend(bbox_to_anchor=(0.02, 0.45), loc='upper left', borderaxespad=0)



plt.figure(2)

plt.plot(t_v, Q_sol_gen, label="gen sol")
# plt.plot(t_v, Q_mort,'*', label="mortensen")
n_fl=np.size(fluent_massflow)
plt.plot(fluent_massflow[0:n_fl:10,0], fluent_massflow[0:n_fl:10,1], '+r', label="fluent")
plt.legend()
plt.title("mass flow rate")
plt.xlabel('t')