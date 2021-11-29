#Symmetric Ogle
#Jack Symonds 
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import data_proc as og
import os

path = "./ogle/"

events = os.listdir(path)

def mag(t, A, u0, t0, tE):
    u = np.sqrt(u0**2 + ((t-t0)/tE)**2)
    m = (u**2 + 2)/(u * np.sqrt(u**2 + 4))
    mag_mag = -np.log10(m)+1
    return A * mag_mag

sym_list = []
for event in events:
    mags = np.loadtxt(path+event+"/phot.dat").transpose()[1]
    guess =  [np.mean(df[1]), np.ptp(df[1]), df[0][np.argmin(df[1])], 5]
    params, parms_covariance = optimize.curve_fit(mag, df[0], df[1], guess)
    event_list.append([event, A])

event_list = sorted(event_list, key=lambda x: x[1])
dims = event_list[0:10]
brights = event_list[-11:-1]

print(brights, dims)
