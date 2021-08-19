import numpy as np

R0 = 5.06
R1 = 0.724
R2 = 0.536
R = 0.208

T1 = (R1-R)/(R0-R)
T2 = (R2-R)/(R0-R)


def wavelength(T):
    b = 100*10**(-12)
    a = 7.6
    n = 2.75
    l = b*(np.log(T)/-a)**(1/n)
    return l

print(wavelength(T2) - wavelength(T1))

h = 6.626*10**(-34)
c = 3*10**8
m0 = 9.109*10**(-31)

print(h/(m0*c))

