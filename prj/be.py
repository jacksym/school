#Back
#Jack Symonds 
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import pandas as pd


def mag(t, A, u0, t0, tE):
    u = np.sqrt(u0**2 + ((t-t0)/tE)**2)
    mag = (u**2 + 2)/(u * np.sqrt(u**2 + 4))
    mag_mag = -np.log10(mag)+1
    return A * mag_mag


class OGLE_data:
    def __init__(self, file, ):
        self.df = np.loadtxt(file).transpose()
        self.df[0] = self.df[0] - 2450000

        i_mean = np.mean(self.df[1])
        params, parms_covariance = optimize.curve_fit(mag, self.df[0], self.df[1], guess)

        t = np.linspace(np.min(self.df[0]), np.max(self.df[0]), 1000)
        fit = mag(t, *params)

        self.tb_arg, self.te_arg = 0, 0
        self.tb, self.te = 0, 0
        for i, mag in enumerate(fit):
            if fit[i-1] - mag > 0.1**4:
                self.tb_arg = i
                self.tb = t[self.tb_arg]
                break

        for i, mag in enumerate(fit[::-1]):
            if fit[i-1] - mag > 0.1**4:
                self.te_arg = len(t) - i
                self.te = t[self.te_arg]
                break

        for i in [tb, te]: ax.axvline(i, linestyle='--', color='gray')

        hjdb_arg = np.argmin(abs(self.df[0] - t[self.tb_arg]))
        hjde_arg = np.argmin(abs(self.df[0] - t[self.te_arg]))
        t_const = np.append(self.df[0][0:hjdb_arg], self.df[0][hjde_arg:])
        self.mag_const = np.append(self.df[1][0:hjdb_arg], self.df[1][hjde_arg:])
        err_const = np.append(self.df[2][0:hjdb_arg], self.df[2][hjde_arg:])


    def plot(self, ax, guess):
        m_max, m_min = np.argmax(self.df[1]), np.argmin(self.df[1])
        i_max, i_min = self.df[1][m_max] + self.df[2][m_max], self.df[1][m_min] - self.df[2][m_max]
        sep = 0.1*(i_max-i_min)

        t = np.linspace(np.min(self.df[0]), np.max(self.df[0]), 100)
        # ax.plot(t, mag(t, *guess))
        ax.plot(t, fit)

        ax.set_title(r'OGLE-2019 $\rightarrow$ '+file)
        ax.set_ylim(i_max+sep, i_min-sep)
        ax.set_xlabel("HJD - 2450000")
        ax.set_ylabel(r'$I$ magnitude')
        ax.errorbar(self.df[0], self.df[1], self.df[2], fmt='none', color='black', linewidth=0.5)

        mag_fit_string = '\n'.join([
            r'$A=%.2f$' % params[0],
            r'$u_0=%.2f$' % params[1],
            r'$t_0=%.2f$' % params[2],
            r'$t_\mathrm{E}=%.2f$' % params[3]
        ])

        ax.text(0.05, 0.95, mag_fit_string, transform=ax.transAxes, fontsize=14,
        verticalalignment='top')


    def const_dist(self, ax):

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

        bars, bins = np.histogram(self.mag_const, 100)
        bins = np.delete(bins, -1)

        ax.hist(self.mag_const, bins, color='lightgray')
        ax.scatter(bins, bars, marker='_')
        ax.set_title("Constant Magnitude Distribution")
        ax.set_xlabel("Magnitude")
        ax.set_ylabel("Frequency")


    def gauss_model(self, ax, guess):

        def gaussian(x, amp, variance, expectation):
            a = 1/(variance * np.sqrt(2*np.pi))
            e = np.exp(-0.5 * ((x - expectation)/variance)**2)
            return amp*a*e

        params, parms_covariance = optimize.curve_fit(gaussian, bins, bars, guess_gauss)

        gauss_fit_string = '\n'.join([
            r'$A=%.2f$' % params_g[0],
            r'$\sigma=%.2f$' % params_g[1],
            r'$\mu=%.2f$' % params_g[2],
        ])

        # ax.plot(bins, gaussian(bins, *guess_gauss))
        ax.plot(bins, gaussian(bins, *params_g))
        ax.axvline(params_g[2], linestyle='--', linewidth=0.5)

        ax.text(0.05, 0.95, gauss_fit_string, transform=ax2.transAxes, fontsize=14,
                verticalalignment='top')

        gauss_formula_string = r'$\mathrm{G(x)}=\frac{A}{\sigma\sqrt{2\pi}}e^{-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2}$'
        ax2.text(0.65, 0.95, gauss_formula_string, transform=ax2.transAxes, fontsize=12, verticalalignment='top')
