# gen_unst_poiseuille
python codes to solve 2D and axisym poiseuille flows with arbitrary pressure gradient laws 

the implementation was done first symbolically using the library  sympy; numpy; scipy.integrate;  scipy.optimize; matplotlib

It first attempts to determine an analytical solution with sympy, which is afterwards evaluated numerically using numpy 
and plotted using matplotlib

it consists of two files:
gen_unst_poiseuille_2D.py
gen_unst_poiseuille_axisym.py

for the solution of the flow in 2D or axisymmetrical domains respectively and a module
gen_unst_poiseuille_aux_2D.py

with auxiliary functions ( verification solutions, data collection from fluent)
