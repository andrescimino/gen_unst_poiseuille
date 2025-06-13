# -*- coding: utf-8 -*-
"""
Created on Wed May 28 11:04:06 2025
auxiliary functions for the axisymetric  solution of the generalized unsteady poiseuille
flow 

@author: acimino001
"""
import numpy as np
from scipy.special import jv
import scipy.integrate as integrate


"""white's starting flow solution"""

# auxiliary functions


def u_ss(r, r0, u_max):
    """ calculates the steadt state component of the flow"""
    
    return u_max*(1-(r/r0)**2)


# def coefsA_k(y, k):
#     """ integral to calculate the fourier coefficients"""
    
#     return 1/2/mu*dpdx*h**2*(1-(y/h)**2)*np.cos(k*np.pi*y/h)





def u1_start(r,t, r0, u_max, nu,  nterms, lambda_v):
    """ calculates the pseudo transient solution using a series with every A calculated  
      as coefficients
      """
    A_v=np.zeros( nterms)
    for i in range(nterms):    
       A_v[i]=-8*u_max/lambda_v[i]**3/jv(1,lambda_v[i] )
    u1=0
    for i in range(nterms):
          u1=u1+A_v[i]*jv(0, lambda_v[i]*r/r0) *np.exp(-(lambda_v[i]/r0)**2*nu*t)
         
    return u1

def u_start(r,t,  r0, u_max, nu,  nterms, lambda_v ):
    """ calculates the total solution using a fourier series with A 
      as coefficients
      """
    u=u_ss(r,  r0, u_max)
    u_1=u1_start(r,t, r0, u_max, nu,  nterms, lambda_v)
    
    u=u+u_1
    
    return (u)
    
def u_start_num(nDr, nDt,  rho,u_max,  mu, r0, tf, dpdx, nterms, lambda_v):
    """ numerical evaluation of the analytical solution for 2D starting flow
        in an array of nDy rows and nDt comumns
    """
    nu = mu/rho
    u_num=np.zeros((nDr, nDt))
    u_1_num=np.zeros((nDr, nDt))
    Q=np.zeros( nDt)
    #temporal and spatial discretization vectors
    t_v=np.linspace(0, tf, nDt)
    r_v=np.linspace(0, r0, nDr)
    u_ss_num=u_ss(r_v, r0, u_max)

    for i in range(nDt):
        # Q_sol[i]=rho*2*integrate.quad(lambda x: u_start(x, i*tf/nDt,  r0, u_max, nu,  nterms, lambda_v), 0,r0)[0]
        for j in range(nDr):

            u_num[j,i]=u_start(r_v[j], t_v[i],  r0, u_max, nu,  nterms, lambda_v)
            u_1_num[j,i]=u1_start(r_v[j], t_v[i],  r0, u_max, nu,  nterms, lambda_v)
    
    for i in range(nDt):
        Q[i] = rho*2*integrate.trapezoid((u_num[:,i]), x=r_v)
    return(u_num, u_1_num, Q, u_ss_num)
        
""" mortensen's oscillating pressure gradient solution"""




def u_mort1(r, t, K, Omega,  nu, r0):
    
    """ calculates mortensen's solution with my parameters and with the radial coord as argument """
    return np.real(K/1j/Omega*np.exp(1j*Omega*t)*(1-jv(0, r*np.sqrt(-1j*Omega/nu))/jv(0, r0*np.sqrt(-1j*Omega/nu))))


def u_osc_num(nDr, nDt,  rho,K, Omega,  mu, r0, tf, dpdx, nterms, lambda_v):
    u_num=np.zeros((nDr, nDt))
    Q=np.zeros( nDt)
    nu=mu/rho
    #temporal and spatial discretization vectors
    t_v=np.linspace(0, tf, nDt)
    r_v=np.linspace(0, r0, nDr)
    for i in range(nDt):
        for j in range(nDr):
            u_num[j,i]=u_mort1(r_v[j], t_v[i],  K, Omega,  nu, r0)
    
    for i in range(nDt):
        Q[i] = rho*2*np.pi*integrate.trapezoid((u_num[:,i])*r_v[:], x=r_v)
    
    return(u_num,  Q)




""" importation of data from fluent solution files"""
# import os 

def fluent_massflow(filename):
    """ opens a mass flow report file from fluent and
    converts it to a numpy array """
    try:
    # Open the file and read its content.
        with open(filename) as f:
            content = f.read().splitlines()
            fluent_massflow=np.zeros((len(content)-3, 2))

    # Display the file's content line by line.
        for i in range(3,len(content)):
            values=content[i].split()
            fluent_massflow[i-3,:] = [float(values[2]), float(values[1])]
            
        return(fluent_massflow)
    except Exception as inst:
        print("exception type \n")
        print(type(inst))    # the exception type
        print("exception args \n")
        print(inst.args)
        return ([])


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
    except Exception as inst:
        print("exception type \n")
        print(type(inst))    # the exception type
        print("exception args \n")
        print(inst.args)
        return ([])

""" plotting functions"""

import matplotlib.pyplot as plt


def plot_velprof(*args, dict_legend:dict, title:str) :
    leg_vals=list(dict_legend.values())
    plt.figure(3)
    for  val in args:
        # print("%s == %s" % (k, val))
        num_vals=np.size(val, 2)
        for j in range(num_vals):
            plt.plot(val[:,0, j], val[:,1,j],marker=j ,  label=str(leg_vals[j]) )

    plt.legend()
    plt.xlabel('r')
    plt.ylabel('u[m/s]')
    plt.title(title)
        

# plt.plot(fluent_vel_t5[:,0, 0], fluent_vel_t5[:,1,0],'*c', label='t5  x=0.5 fluent' )
# plt.plot(fluent_vel_t5[:,0, 1], fluent_vel_t5[:,1, 1],'+c', label='t5  x=0.3 fluent' )
# plt.plot(fluent_vel_t5[:,0, 2], fluent_vel_t5[:,1,2],'.c', label='t5  x=0.0 fluent' )



# plt.plot(fluent_vel_t10[:,0, 0], fluent_vel_t10[:,1,0],'*b', label='t10  x=0.5 fluent' )
# plt.plot(fluent_vel_t10[:,0, 1], fluent_vel_t10[:,1, 1],'+b', label='t10  x=0.3 fluent' )
# plt.plot(fluent_vel_t10[:,0, 2], fluent_vel_t10[:,1,2],'.b', label='t10  x=0.0 fluent' )

# plt.plot(fluent_vel_t20[:,0, 0], fluent_vel_t20[:,1, 0],'*r', label='t20 x=0.5 fluent' )
# plt.plot(fluent_vel_t20[:,0,1], fluent_vel_t20[:,1, 1],'+r', label='t20  x=0.3 fluent' )
# plt.plot(fluent_vel_t20[:,0, 2], fluent_vel_t20[:,1,2],'.r', label='t20 x=0.0 fluent' )

# plt.plot(fluent_vel_t30[:,0, 0], fluent_vel_t30[:,1,0],'*y', label='t30 x=0.5 fluent' )
# plt.plot(fluent_vel_t30[:,0, 1], fluent_vel_t30[:,1,1],'+y', label='t30 x=0.3 fluent' )
# plt.plot(fluent_vel_t30[:,0, 2], fluent_vel_t30[:,1,2],'.y', label='t30 x=0.0 fluent' )
