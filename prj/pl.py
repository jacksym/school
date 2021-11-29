#Jack Symonds Back
import numpy as np
import matplotlib.pyplot as plt
# import math
from scipy import optimize
import pandas as pd
# import os


path = "./ogle/"

file = "blg-0011/phot.dat"

df = np.loadtxt(path+file).transpose()
df[0] = df[0] - 2450000

i_mean = np.mean(df[1])

m_max, m_min = np.argmax(df[1]), np.argmin(df[1])
i_max, i_min = df[1][m_max] + df[2][m_max], df[1][m_min] - df[2][m_max]
sep = 0.1*(i_max-i_min)
# g_max, g_min = math.ceil((i_max+sep)*4)/4, math.floor((i_min-sep)*4)/4

def mag(t, A, u0, t0, tE):
    u = np.sqrt(u0**2 + ((t-t0)/tE)**2)
    mag = (u**2 + 2)/(u * np.sqrt(u**2 + 4))
    mag_mag = -np.log10(mag)+1
    return A * mag_mag


guess = [np.mean(df[1]), 1.8, df[0][m_min], 5]

params, parms_covariance = optimize.curve_fit(mag, df[0], df[1], guess)
# print(params)

fig, ax = plt.subplots(figsize=(10, 7))

t = np.linspace(np.min(df[0]), np.max(df[0]), 1000)

# print(np.min(df[0]), np.max(df[0]))


# ax.plot(t, mag(t, *guess))
ax.plot(t, mag(t, *params))

ax.set_title(r'OGLE-2019 $\rightarrow$ '+file)
ax.set_ylim(i_max+sep, i_min-sep)
ax.set_xlabel("HJD - 2450000")
ax.set_ylabel(r'$I$ magnitude')
ax.errorbar(df[0], df[1], df[2], fmt='none', color='black', linewidth=0.5)


mag_fit_string = '\n'.join([
    r'$A=%.2f$' % params[0],
    r'$u_0=%.2f$' % params[1],
    r'$t_0=%.2f$' % params[2],
    r'$t_\mathrm{E}=%.2f$' % params[3]]
)

ax.text(0.05, 0.95, mag_fit_string, transform=ax.transAxes, fontsize=14,
        verticalalignment='top')

fit = mag(t, *params)

tb_arg, te_arg = 0, 0
tb, te = 0, 0
for i, mag in enumerate(fit[1:]):
    if fit[i-1] - mag > 0.1**4:
        tb_arg = i
        tb = t[tb_arg]
        break
# print(tb_arg, t[tb_arg])

for i, mag in enumerate(fit[-2::-1], start=2):
    if fit[len(fit)-i+1] - mag > 0.1**4:
        te_arg = len(t) - i
        te = t[te_arg]
        break
print(te_arg, t[te_arg])

for i in [tb, te]: ax.axvline(i, linestyle='--', color='gray')

# df_new = np.loadtxt(path+file)
hjdb_arg = np.argmin(abs(df[0] - t[tb_arg]))
hjde_arg = np.argmin(abs(df[0] - t[te_arg]))
# print(hjdb_arg, hjde_arg)
t_const = np.append(df[0][0:hjdb_arg], df[0][hjde_arg:])
mag_const = np.append(df[1][0:hjdb_arg], df[1][hjde_arg:])
err_const = np.append(df[2][0:hjdb_arg], df[2][hjde_arg:])


def bin(arr, no_bins):
    bins = np.linspace(np.min(arr), np.max(arr), no_bins)
    bins_len = len(bins)
    bars = np.zeros(bins_len)
    if len(arr) < bins_len: print("too many bins")
    for i in arr:
        for j in range(bins_len - 1):
            if bins[j] <= i < bins[j+1]: bars[j] += 1
        if i == bins[bins_len-1]: bars[bins_len-1] += 1
    return(bins, bars)


# bars, bins = np.histogram(mag_const, 100, density=True)
bars, bins = np.histogram(mag_const, 50)
bins = np.delete(bins, -1)


def gaussian(x, amp, variance, expectation):
    a = 1/(variance * np.sqrt(2*np.pi))
    e = np.exp(-0.5 * ((x - expectation)/variance)**2)
    return amp*a*e


guess_gauss = [2, np.ptp(bins)/10, np.average(bins, weights=bars)]
params_g, parms_covariance_g = optimize.curve_fit(gaussian, bins, bars, guess_gauss)

gauss_fit_string = '\n'.join([
    r'$A=%.2f$' % params_g[0],
    r'$\sigma=%.2f$' % params_g[1],
    r'$\mu=%.2f$' % params_g[2],
])

fig2, ax2 = plt.subplots()


# ax2.hist(mag_const, 50, color='gray')
# ax2.plot(bins, gaussian(bins, *guess_gauss))
ax2.plot(bins, gaussian(bins, *params_g))
ax2.axvline(params_g[2], linestyle='--', linewidth=0.5)
ax2.scatter(bins, bars, marker='_')
ax2.set_title("Constant Magnitude Distribution")
ax2.set_xlabel("Magnitude")
ax2.set_ylabel("Frequency")

ax2.text(0.05, 0.95, gauss_fit_string, transform=ax2.transAxes, fontsize=14,
        verticalalignment='top')

gauss_formula_string = r'$\mathrm{G(x)}=\frac{A}{\sigma\sqrt{2\pi}}e^{-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2}$'
ax2.text(0.50, 0.95, gauss_formula_string, transform=ax2.transAxes, fontsize=12, verticalalignment='top')

plt.show()
