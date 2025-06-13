# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 10:07:40 2024

@author: acimino001

solves the transient flow inside a 2D cavity of height 2h 

 /////////////////////////////////
              ^                   
 ^y           |                   
 |            |h
 |-->x _ _ _ _| _ _ _ _ _ _ _ _ 



//////////////////////////////////

du/dt=nu d^2u/dy^2 -1/rho dp/dx

dp/dx is  an arbitrary function of time and space, 


half of the cavity is modelled, assuming the flow is symmetric wrt to the midplane


BCS u(y=h, t)=0

    du/dy(y=0, t)=0
    u1(y,0)=-u_ss


Solution is obtained using separation of variables

u(y, t)=V_j(y)T_j(t)

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
import gen_unst_poiseuille_aux_2D as aux2D






""" symbolic solution
proposing a solution of the form 

u(t,y)=T(t)*V(y)

we propose 

V(y)= A_i* cos(lambda_i * y) since it satisfies the BCs

and arbitrary IC

u(0, y)=u_0(y)

"""



import sympy as sym


#symbolic indep variables, constants and params (all positive)
y_s,t_s,s,  h_s, lambda_s, Omega_s, nu_s= sym.symbols('y t  s  h λ_i Ω ν', positive=True)
#symbolic coefficients and functions
A_s,V_s,T_s=sym.symbols('A V T' , nonzero=True)
C_1_s,g_s, dpdx_s, u_s, K_s, u_0s, u_max_s= sym.symbols(' C_1 g dpdx u K u_0 u_max') # sybolic functions and coefficients
#integer symbolic variables
n_s, k_s = sym.symbols('n k', integer = True)


""" verification cases"""
#########  test function for a known case -dpdx=K*cos(omega*t), as a function of s, an integration
# variable

A_1i,B_1i, J_1i= sym.symbols('A1_i B1_i J_1', positive=True, real=True)

u_0s= K_s/Omega_s/J_1i*(- sym.cosh(B_1i*h_s)*sym.cos(B_1i*h_s)*sym.sinh(B_1i*y_s)*sym.sin(B_1i*y_s) + sym.sinh(B_1i*h_s)*sym.sin(B_1i*h_s)*sym.cosh(B_1i*y_s)*sym.cos(B_1i*y_s))#-Omega_s*sym.sinh(A_1i)*sym.sin(A_1i)*sym.cosh(B_1i*y_s)*sym.cos(B_1i*y_s) )/J_1i
dpdx_s=K_s*sym.cos(Omega_s*s) # pressure gradient function


##########    test function for a known case : sudden impulse in dpdx  #######
#dpdx=K, constant, complete solution (transient +steady state)

# u_0s=0
# dpdx_s=K_s

##############5     transient solution for the sudden impulse
# u_0s=-u_max_s*(1-(y_s/h_s)**2)

# dpdx_s=0
#dpdx_s=(1-y_s)*s**2# ploynomical test function



print("initial condition")
sym.pprint(sym.simplify(u_0s), num_columns=300)
""" spatial modes and modal decomposition of the pressure gradient function"""
#k=0
# lambda_s=sym.pi*(2*k_s+1)/2/h_s
#modes for the spatial solution
V_s=A_s*sym.cos(lambda_s*y_s)
# V_s=A_s*sym.cos(sym.pi*(2*k_s+1)/2/h_s*y_s)

#☻transformed pressure gradient function for modal decomposition
print("g_s")
g_s=2/A_s**2/h_s*sym.integrate(dpdx_s*V_s.subs(lambda_s, sym.pi*(2*k_s+1)/2/h_s), (y_s, 0, h_s))

sym.pprint(g_s , num_columns=200)
"""temporal solution of the variable separation ( for zero IC) """


def ex(e,x):
    
    """ function that expands an expression only wrt to a given factor"""
    m = [i for i in e.atoms(sym.Mul) if not i.has(x)] #searchs for evey factor which contains the variable x
    reps = dict(zip(m,[sym.Dummy() for i in m])) #creates a dictionary with temporary symbolic variables as items
    # zip is like a cartesian product of two iterables
    return sym.expand(e.xreplace(reps)).subs([(v,k) for k,v in reps.items()])




#T_s=sym.exp(-nu_s*lambda_s**2*t_s)*sym.integrate(g_s*sym.exp(nu_s*lambda_s**2*s), (s, 0, t_s))
T_s=sym.integrate(g_s*sym.exp(-nu_s*lambda_s**2*(t_s-s)), (s, 0, t_s))
T_s=sym.simplify(T_s)
# T_s=sym.expand_power_exp(T_s)


# procedure to eliminate terms with positive exponentials
#the produce overflows during the numerical evaluation


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
# C_1_s= 2/A_s**2/h_s*sym.integrate(u_0s*V_s.subs(k_s, 1), (y_s, 0, h_s))
C_1_s= 2/A_s**2/h_s*sym.integrate(u_0s*V_s, (y_s, 0, h_s))
print("done")

# substitution of constants 
C_1_s = C_1_s.subs ( J_1i, sym.sin(B_1i*h_s)**2*sym.sinh(B_1i*h_s)**2+sym.cos(B_1i*h_s)**2*sym.cosh(B_1i*h_s)**2)
C_1_s = C_1_s.subs ( B_1i , sym.sqrt(Omega_s/2/nu_s))
C_1_s=sym.simplify(C_1_s)

print("C_1")
sym.pprint((C_1_s), num_columns=300)

"""complete temporal  solution
initial condition and temporal variation of the pressure gradient
"""

T_s= T_s+ C_1_s*sym.exp(-nu_s*lambda_s**2*(t_s))
print("T_s")
sym.pprint(T_s, num_columns=500)
T_s=T_s.subs(lambda_s, sym.pi*(2*k_s+1)/2/h_s)

""" complete solution"""

u_s=T_s*V_s.subs(lambda_s, sym.pi*(2*k_s+1)/2/h_s)
print("u_s")


sym.pprint(u_s, num_columns=500)

""" numerical calculations"""


# nt=10 # number of terms of the solution series

""" functions for numerical evaluation"""



Deltap_n=-12.5 #pa
L_n=1 #[m]
h_n=10e-3


dpdx_n=Deltap_n/L_n


mu_n=1.00e-3
rho_n=1000
nu_n=mu_n/rho_n
u_max_n=-h_n**2/2/mu_n*dpdx_n


# parameters for the cosenoidal pressure gradient
Period=30 # Period of the pressure pulse [s]
Omega_n=2*np.pi/Period # frequency of the pressure pulse [1/s]
K_n=-Deltap_n/L_n/rho_n

tau=np.log(2)/(np.pi/2/h_n)**2/nu_n


# number of space and time discretizations
nDy=11
nDt=121
tf=2*Period
nterms=5
# #arrays of discretized space and time

t_v=np.linspace(0, tf, nDt)

y_v=np.linspace(0, h_n, nDy)

u_sol=np.zeros((nDy, nDt))
Q_sol_gen=np.zeros(( nDt))
# u_ss_sol=u_ss(y_v)


#transient solution for a starting flow with a step pressure gradient
u_num_trans=sym.lambdify([y_s,t_s,K_s,k_s, h_s, nu_s, u_max_s, Omega_s], u_s)
for i in range(nDt):
    # Q_sol[i]=rho_n*2*integrate.quad(lambda x: u_num_trans(x, i*tf/nDt, K_n,k_n, h_n, nu_n, u_max_n, Omega_n), 0,h_n)[0]
    for j in range(nDy):
        u_sol[j,i]=0#u_ss(y_v[j])
        for k in range(nterms):
            u_sol[j,i]= u_sol[j,i]+u_num_trans(y_v[j], t_v[i] ,-dpdx_n/rho_n,k, h_n, nu_n, u_max_n, Omega_n)

for i in range(nDt):
    Q_sol_gen[i] = rho_n*2*integrate.trapezoid((u_sol[:,i]), x=y_v)

# """ trajectory determination """

# def u_tray(t,x):
#     """ form of the velocity eq to be used with the ode solver"""
#     return u(y_0, t)

# x_tray=integrate.solve_ivp(u_tray,[0 ,tf], [0] )

# """ plots """


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





#fourier coefficients

""" verification solutions """
[u_panton, Q_panton] = aux2D.u_puls_panton_num(nDy, nDt,tf,  K_n, nu_n,rho_n,  Omega_n, h_n)
[u_start, u1_start, Q_start, u_ss_start] = aux2D.u_start_num(nDy, nDt, nterms,rho_n,  mu_n, h_n, tf, dpdx_n)

  

# dpdx_a=-K_n*np.cos(Omega_n*t_v)
""" data from fluent """

import os 

filename = "fluent_files/pulsed/2D/q_mass-rfile.out"

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

    
# filename = "fluent_files/pulsed/vel_prof_puls_To2"
# fluent_vel_To2=fluent_velprof(filename)

# filename = "fluent_files/pulsed/vel_prof_puls_To4"
# fluent_vel_To4=fluent_velprof(filename)
# filename = "fluent_files/pulsed/vel_prof_puls_To8"
# fluent_vel_To8=fluent_velprof(filename)


filename = "fluent_files/start/2D/vel_prof_start_t15.xy"
fluent_vel_t15=aux2D.fluent_velprof(filename)

filename = "fluent_files/start/2D/vel_prof_start_t30.xy"
fluent_vel_t30=aux2D.fluent_velprof(filename)
filename = "fluent_files/start/2D/vel_prof_start_t60.xy"
fluent_vel_t60=aux2D.fluent_velprof(filename)


# filename = "fluent_files/pulsed/2D/vel_prof_puls_2d_t5.xy"
# fluent_vel_t5=aux2D.fluent_velprof(filename) 

# filename = "fluent_files/pulsed/2D/vel_prof_puls_2d_t10.xy"
# fluent_vel_t10=aux2D.fluent_velprof(filename)

# filename = "fluent_files/pulsed/2D/vel_prof_puls_2d_t20.xy"
# fluent_vel_t20=aux2D.fluent_velprof(filename)
# filename = "fluent_files/pulsed/2D/vel_prof_puls_2d_t30.xy"
# fluent_vel_t30=aux2D.fluent_velprof(filename)


""" plots 

"""


plt.figure(1)

# plt.plot(y_v,u_sol[:,0],'-g', label='T=0 gen' )
# plt.plot(y_v,u_sol[:,np.where(np.abs(t_v-5)<=0.1)[0]],'-m', label='t=5 s gen' )
# plt.plot(y_v,u_sol[:,np.where(np.abs(t_v-10)<=0.1)[0]],'-r', label='t=10 s gen' )
plt.plot(y_v,u_sol[:,np.where(np.abs(t_v-15)<=0.1)[0]],'-b', label='t=15 s gen' )
# plt.plot(y_v,u_sol[:,np.where(np.abs(t_v-20)<=0.1)[0]],'-b', label='t=20 s  gen' )
plt.plot(y_v,u_sol[:,np.where(np.abs(t_v-30)<=0.1)[0]],'-r', label='t=30 s gen' )
plt.plot(y_v,u_sol[:,np.where(np.abs(t_v-60)<=0.1)[0]],'-g', label='t=30 s gen' )




# plt.plot(y_v,u_start[:,np.where(np.abs(t_v-10)<=0.1)[0]],'*r', label='t=10 s start' )
# plt.plot(y_v,u_start[:,np.where(np.abs(t_v-20)<=0.1)[0]],'*b', label='t=20 s  start' )
# plt.plot(y_v,u_start[:,np.where(np.abs(t_v-30)<=0.1)[0]],'*y', label='t= 30 s start' )
# plt.plot(y_v,u_start[:,np.where(np.abs(t_v-60)<=0.1)[0]],'*y', label='t= 60 s start' )



# plt.plot(y_v,u_ss_start,'--m', label='steady state start' )

# plt.plot(y_v,u1_start[:,0],'*g', label='T=0 transient' )
# plt.plot(y_v,u1_start[:,np.where(np.abs(t_v-10)<=0.1)[0]],'*r', label='t=10 s trans start' )
# plt.plot(y_v,u1_start[:,np.where(np.abs(t_v-20)<=0.1)[0]],'*b', label='t=20 s trans start' )
# plt.plot(y_v,u1_start[:,np.where(np.abs(t_v-30)<=0.1)[0]],'*y', label='t=30 s trans start' )
# plt.plot(y_v,u1_start[:,np.where(np.abs(t_v-60)<=0.1)[0]],'*m', label='t=60 s trans start' )

# plt.plot(y_v,u_panton[:,np.where(np.abs(t_v-5)<=0.1)[0]],'*m', label='t= 5 s pulsed' )
# plt.plot(y_v,u_panton[:,np.where(np.abs(t_v-10)<=0.1)[0]],'*r', label='t=10 s pulsed' )
# plt.plot(y_v,u_panton[:,np.where(np.abs(t_v-20)<=0.1)[0]],'*b', label='t=20 s  pulsed' )
# plt.plot(y_v,u_panton[:,np.where(np.abs(t_v-30)<=0.1)[0]],'*y', label='t= 30 s pulsed' )



# for i in range(0, nDt, 10):
#     plt.plot(y_v, u_sol[:, i], label=str(i*tau/nDt))
# plt.plot(y,u2[:,0], '-g', label='T=0 panton' )
# plt.plot(y,u2[:,m.floor(nDt/8)],'-r' ,label='T=Per/4 panton' )
# plt.plot(y,u2[:,m.floor(nDt/4)],'-b', label='T=Per/2 panton' )
# plt.plot(y,u2[:,m.floor(nDt/16)],'-y', label='T=Per/8 panton' )
# plt.plot(fluent_vel_t5[:,0, 0], fluent_vel_t5[:,1,0],'*m',markerfacecolor='none', markersize=10, label='t=5 x=0.1 fluent' )

# plt.plot(fluent_vel_t10[:,0, 0], fluent_vel_t10[:,1,0],'*r',markerfacecolor='none', markersize=10, label='t=10 x=0.1 fluent' )
plt.plot(fluent_vel_t15[:,0, 0], fluent_vel_t15[:,1,0],'*b',markerfacecolor='none', markersize=10, label='t=10 x=0.1 fluent' )

# plt.plot(fluent_vel_t20[:,0, 0], fluent_vel_t20[:,1,0],'*b',markerfacecolor='none', markersize=10, label='t=20 x=0.1 fluent' )


plt.plot(fluent_vel_t30[:,0, 0], fluent_vel_t30[:,1,0],'*r',markerfacecolor='none', markersize=10, label='t=30 x=0.1 fluent' )
plt.plot(fluent_vel_t60[:,0, 0], fluent_vel_t60[:,1,0],'*g',markerfacecolor='none', markersize=10, label='t=30 x=0.1 fluent' )
 

# plt.plot(fluent_vel_To2[5:11,0, 0], fluent_vel_To2[5:11,1,0],'*b', label='T/2  x=0.5 fluent' )
# plt.plot(fluent_vel_To2[5:11,0, 1], fluent_vel_To2[5:11,1, 1],'+b', label='T/2  x=0.3 fluent' )
# plt.plot(fluent_vel_To2[5:11,0, 2], fluent_vel_To2[5:11,1,2],'.b', label='T/2  x=0.0 fluent' )

# plt.plot(fluent_vel_To4[5:11,0, 0], fluent_vel_To4[5:11,1, 0],'*r', label='T/4 x=0.5 fluent' )
# plt.plot(fluent_vel_To4[5:11,0,1], fluent_vel_To4[5:11,1, 1],'+r', label='T/4  x=0.3 fluent' )
# plt.plot(fluent_vel_To4[5:11,0, 2], fluent_vel_To4[5:11,1,2],'.r', label='T/4 x=0.0 fluent' )

# plt.plot(fluent_vel_To8[5:11,0, 0], fluent_vel_To8[5:11,1,0],'*y', label='T/8 x=0.5 fluent' )
# plt.plot(fluent_vel_To8[5:11,0, 1], fluent_vel_To8[5:11,1,1],'+y', label='T/8 x=0.3 fluent' )
# plt.plot(fluent_vel_To8[5:11,0, 2], fluent_vel_To8[5:11,1,2],'.y', label='T/8 x=0.0 fluent' )
plt.legend(bbox_to_anchor=(0.02, 0.35), loc='upper left', borderaxespad=0)
plt.xlabel('y')
plt.ylabel('u[m/s]')
# plt.title('Comparison of velocity profiles for the  cstarting flow solution ')
plt.figure(2)

plt.plot(t_v, Q_sol_gen, label="gen sol")
# plt.plot(t_v, Q_start, '.r', label="starting flow")
n_fl=np.size(fluent_massflow)
plt.plot(fluent_massflow[0:n_fl:5,0], -fluent_massflow[0:n_fl:5,1],'*r' , label="fluent")
plt.legend()
plt.title("mass flow rate")
plt.xlabel('t')