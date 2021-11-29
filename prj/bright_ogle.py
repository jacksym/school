#Bright Ogle
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

event_list = []
for event in events:
    mags = np.loadtxt(path+event+"/phot.dat").transpose()[1]
    # guess =  [np.mean(df[1]), np.ptp(df[1]), df[0][np.argmin(df[1])], 5]
    A = np.mean(mags)
    # params, parms_covariance = optimize.curve_fit(mag, df[0], df[1], guess)
    # event_list.append([event, params[0]])
    event_list.append([event, A])

event_list = sorted(event_list, key=lambda x: x[1])
dims = event_list[0:10]
brights = event_list[-11:-1]

print(brights, dims)

#RESULTS
[['blg-0896', 20.699592833876224],
 ['blg-1223', 20.714108695652172],
 ['blg-0923', 20.715253378378378],
 ['blg-0788', 20.726826481257557],
 ['blg-1166', 20.753653096330275],
 ['blg-0897', 20.75527118644068],
 ['blg-0194', 20.758302991725014],
 ['blg-0195', 20.77296709425544],
 ['blg-1400', 20.77531836734694],
 ['blg-0613', 20.788582352941177]]
[['blg-0148', 13.291820859227235],
 ['blg-0151', 13.464748233215547],
 ['blg-0010', 13.644160115398488],
 ['blg-1017', 13.650387358184766],
 ['blg-0551', 13.72217144319345],
 ['blg-0296', 13.92269411764706],
 ['blg-0252', 13.946517316017315],
 ['blg-0087', 14.03023856858847],
 ['blg-0653', 14.047839050131929],
 ['blg-0919', 14.138795774647889]]
