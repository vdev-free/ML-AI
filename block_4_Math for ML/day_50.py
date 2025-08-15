import numpy as np

def g(x):
    return x**2 + 3*x +1

def f(x):
    return g(x)**2

def dg_dx(x):
    return 2*x + 3

def df_dx(x):
    return dg_dx(x) * g(x)*2

x = 1.0
print(f"f(x) = {f(x):.2f}")
print(f"Похідна df/dx = {df_dx(x):.2f}")