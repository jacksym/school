import numpy as np
import matplotlib.pyplot as plt
# import math
# import pandas as pd
# import os

t = np.linspace(0, 100, 1000)


def normald(x, mean_expectation, standard_deviation):
    a = 1/(standard_deviation*np.sqrt(2*np.pi))
    e = np.exp(-0.5*((x-mean_expectation)/standard_deviation)**2)
    return a*e


def tot_mag(u):
    return (u**2+2)/(u*np.sqrt(u**2+4))


def rel_lens_source_motion(t, u0, t0, tE):
    a = (t-t0)/tE
    return np.sqrt(u0**2 + a**2)


def mag(t, A, u0, t0, tE):
    u = np.sqrt(u0**2 + ((t-t0)/tE)**2)
    mag = (u**2 + 2)/(u * np.sqrt(u**2 + 4))
    mag_mag = -np.log10(mag)+1
    return A * mag_mag


guess = [17, 0.1, 60, 8]


fig, ax = plt.subplots(figsize=(10, 7))

ax.set_title(r'Test Model')
ax.plot(t, mag(t, *guess))
ax.ticklabel_format(useOffset=False)
ax.set_xlabel("HJD - 2450000")
ax.set_ylabel(r'$I$ magnitude')

fig.show()
