# -*- coding: utf-8 -*-
"""
Created on Tue May 27 10:53:52 2025

@author: acimino001

auxiliary functions for the 2D solution of the generalized 2D poiseuille
flow 

"""
import numpy as np
import scipy.integrate as integrate



""" solutions for starting flow """


def u_ss(y, mu, h, dpdx):
    """ calculates the steady state component of the flow ( as t->infinity)
        it's the classical 2D steady state solution
    """
    
    return -1/2/mu*h**2*dpdx*(1-(y/h)**2)


def coefsA_k(y, k , mu, h, dpdx ):
    """  argument of the integral to calculate the fourier coefficients of every term in the 
    solution. 
    the integral is then solved numerically    
    """
    
    return 1/2/mu*dpdx*h**2*(1-(y/h)**2)*np.cos(k*np.pi*y/h)


def u1(y,t, A,  nu, h, dpdx ):
    """ calculates the pseudo transient solution using a fourier series with A 
     as coefficients
     u(y,t) = u_steady(y) +u_1(y, t)
     
     A is passed as a numpy array
     """
    u1=0
    for i in range(len(A)):
         k=i+1/2
         u1=u1+A[i]*np.cos(k*np.pi*y/h)*np.exp(-(k*np.pi/h)**2*nu*t)
         
    return u1

def u(y,t, A,  nu,mu, h, dpdx):
    """ calculates the total solution using a fourier series with A 
     as coefficients
     """
    u=u_ss(y, mu, h, dpdx)
    for i in range(len(A)):
         k=i+1/2
         u=u+A[i]*np.cos(k*np.pi*y/h)*np.exp(-(k*np.pi/h)**2*nu*t)
         
    return u

def u_start_num(nDy, nDt, nt,rho,  mu, h, tf, dpdx):
    """ numerical evaluation of the analytical solution for 2D starting flow
        in an array of nDy rows and nDt comumns
    """
    nu= mu/rho # kinematic viscosity
    A=np.zeros( nt)
    #temporal and spatial discretization vectors
    t_v=np.linspace(0, tf, nDt)
    y_v=np.linspace(0, h, nDy)

    for i in range(nt):
        k=i+1/2    
        A[i]=2/h*integrate.quad(lambda x: coefsA_k(x, k, mu, h, dpdx), 0,h)[0]






    u_v=np.zeros((nDy, nDt))
    u1_v=np.zeros((nDy, nDt))
    Q_v=np.zeros(( nDt))
    u_ss_v=u_ss(y_v, mu, h, dpdx)

    for i in range(nDt):
        Q_v[i]=rho*2*integrate.quad(lambda x: u(x, i*tf/nDt, A,  nu,mu, h, dpdx), 0,h)[0]
        for j in range(nDy):
            u_v[j,i]=u(y_v[j], t_v[i],  A,  nu,mu, h, dpdx)

    for i in range(nDt):
        for j in range(nDy):
            u1_v[j,i]=u1(y_v[j], t_v[i],  A,  nu, h, dpdx)
    return(u_v, u1_v, Q_v, u_ss_v)


""" panton's nondimensional solution for oscillating pressure gradient"""

""" auxiliary functions """


def C(x):
    """ función auxiliar para obtener la solución
        C(x)=cosh(x)cos(x)
    """    
    return np.cosh(x)*np.cos(x)

def S(x):
    """ función auxiliar para obtener la solución
        S(x)=sinh(x)sin(x)
    """    
    return np.sinh(x)*np.sin(x)

def u_puls_panton_num(nDy, nDt,t_f,  K, nu,rho, Omega, h):
    """ 
    numerical evaluation of the nondimensional analytical solution in Panton
    outputs an array with values
    of the u velocity for  discretised time and y coordinates
    nu= kinematic viscosity, K= temporal amplitude of the pressure gradient
    Omega= frequency of the pressure gradient, h= midheight between the plates
    """
    alpha= K/ Omega
    Lambda=h/np.sqrt(2*nu/Omega)

    Y=np.linspace(0, 1, num=nDy)# nondimensionalised space discretization vector
    t=np.linspace(0,t_f, num=nDt ) #vector de tiempo adimensionalizado



    U2=np.zeros([nDy, nDt]) #espacio de memoria para la velocidad adimensional
    M=np.zeros(nDy); N=np.zeros(nDy);


    J=C(Lambda)**2+S(Lambda)**2
    for j in range(nDy):
      M[j]=C(Lambda*Y[j])*C(Lambda)+S(Lambda*Y[j])*S(Lambda)
      N[j]=C(Lambda*Y[j])*S(Lambda)-S(Lambda*Y[j])*C(Lambda)
      for i in range(nDt): 
          U2[j,i]=(1-M[j]/J)*np.sin(Omega*t[i])+N[j]/J*np.cos(Omega*t[i]) # there's an error in pantons book
        #the cos term should be positive




#dimensionalisation of the solution
    u2=U2*alpha
    Q=np.zeros(nDt);
    for i in range(nDt):
        Q[i] = rho*2*integrate.trapezoid((u2[:,i]), x=Y*h)
    return(u2, Q)



""" import data from fluent """

def fluent_velprof(file):
    """opens a profile file from fluent with velocity profiles and converts 
    them into a numpy array"""
    # if not os.path.isfile(file):
    #     print('File does not exist.')

    # else:
    try:
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
            for i in range(52,65):
                values_vt0.append(content[i].split())
             #a  numpy array for every line where there is data storaged
         # print(values_vt0)
            fluent_vel=np.zeros((12, 2, 4))
            data_buffer1=np.zeros((12, 2))
            data_buffer2=np.zeros((12, 2))
            data_buffer3=np.zeros((12, 2))
            data_buffer4=np.zeros((12, 2))
             # data_buffer4=np.zeros((11, 2))
             # fluent_vel_x01=np.zeros((11, 2))
             # fluent_vel_x03=np.zeros((11, 2))
             # fluent_vel_x05=np.zeros((11, 2))
 
            for i in range(12):
                 data_buffer1 [i,:] = [float(values_vt0[i][1]), float(values_vt0[i][0])]
                 data_buffer2 [i,:] = [float(values_vt0[12+i][1]) , float(values_vt0[12+i][0])]
                 data_buffer3 [i,:] = [float(values_vt0[24+i][1]), float(values_vt0[24+i][0])]
                 data_buffer4 [i,:] = [float(values_vt0[36+i][1]), float(values_vt0[36+i][0])]
       
            #fluent_vel= fluent_vel[fluent_vel[:, 0,:].argsort(axis=1)]
            # fluent_vel[:,:,1]= fluent_vel[fluent_vel[:, 0,1].argsort(), 1]
            # fluent_vel[:,:,2]= fluent_vel[fluent_vel[:, 0,2].argsort(),2]
            # fluent_vel[:,:,3]= fluent_vel[fluent_vel[:, 0,3].argsort(),3]
            # data_buffer1= data_buffer1[data_buffer1[:,0].argsort()]
            # data_buffer2= data_buffer2[data_buffer2[:,0].argsort()]
            # data_buffer3= data_buffer3[data_buffer3[:,0].argsort()]
            # data_buffer4= data_buffer4[data_buffer4[:,0].argsort()]
            
            fluent_vel[:,:, 0] =data_buffer1
            fluent_vel[:,:, 1] =data_buffer2
            fluent_vel[:,:, 2] =data_buffer3
            fluent_vel[:,:, 3] =data_buffer4
            # fluent_vel[:,:, 3] =data_buffer4
        return (fluent_vel)
    except Exception as inst:
        print("exception type \n")
        print(type(inst))    # the exception type
        print("exception args \n")
        print(inst.args)
        return ([])
