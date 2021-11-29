#Back End to Microlensing Data Processing
#Jack Symonds 
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize, integrate
import pandas as pd
import math


class OGLE_data:
    def __init__(self, event):
        self.df = np.loadtxt(event+"/phot.dat").transpose()
        self.df[0] = self.df[0] - 2450000

        with open(event+"/params.dat", 'r', errors='ignore') as params:
            lines = params.readlines()
            self.m_z = float(lines[-1].split()[1])
            self.m_ze = float(lines[-1].split()[2])
            
    
    def plot(self, ax, title):
        
        def mag(t, A, u0, t0, tE):
            u = np.sqrt(u0**2 + ((t-t0)/tE)**2)
            m = (u**2 + 2)/(u * np.sqrt(u**2 + 4))
            mag_mag = -2.5*np.log10(m)+1
            return A * mag_mag

        u0_g = np.min(self.df[1])**(-1)
        tE_g = 0.5*u0_g**(-1) * 0.1*np.ptp(self.df[0])
        # print(u0_g, tE_g)
        guess =  [np.mean(self.df[1]), u0_g, self.df[0][np.argmin(self.df[1])], tE_g]
        self.params, parms_covariance = optimize.curve_fit(mag, self.df[0], self.df[1], guess)
        t = np.linspace(np.min(self.df[0]), np.max(self.df[0]), 1000)
        fit = mag(t, *self.params)

        amp = self.params[0]
        self.tb_arg, self.te_arg = 0, 0
        self.tb, self.te = 0, 0
        for i, mag in enumerate(fit[1:]):
            if amp - mag > 0.1**3:
                self.tb_arg = i
                self.tb = t[self.tb_arg]
                break

        for i, mag in enumerate(fit[-2::-1], start=2):
            if amp - mag > 0.1**3:
                self.te_arg = len(t) - i
                self.te = t[self.te_arg]
                break

        for i in [self.tb, self.te]: ax.axvline(i, linestyle='--', color='gray')

        self.hjdb_arg = np.argmin(abs(self.df[0] - t[self.tb_arg]))
        self.hjde_arg = np.argmin(abs(self.df[0] - t[self.te_arg]))
        t_const = np.append(self.df[0][0:self.hjdb_arg], self.df[0][self.hjde_arg:])
        self.mag_const = np.append( self.df[1][0:self.hjdb_arg], self.df[1][self.hjde_arg:])
        self.err_const = np.append(self.df[2][0:self.hjdb_arg], self.df[2][self.hjde_arg:])

        m_max, m_min = np.argmax(self.df[1]), np.argmin(self.df[1])
        i_max, i_min = self.df[1][m_max] + self.df[2][m_max], self.df[1][m_min] - self.df[2][m_max]
        sep = 0.1*(i_max-i_min)

        # t = np.linspace(np.min(self.df[0]), np.max(self.df[0]), 100)
        # ax.plot(t, mag(t, *guess))
        ax.plot(t, fit, linewidth=0.7)
        # ax.plot(t, mag(t, *self.params))

        ax.set_title(r'OGLE-2019 $\rightarrow$ ' + title)
        ax.set_ylim(i_max+sep, i_min-sep)
        ax.set_xlabel("HJD - 2450000")
        ax.set_ylabel(r'$I$ magnitude')
        ax.errorbar(self.df[0], self.df[1], self.df[2], fmt='none', color='black', linewidth=0.5)

        mag_fit_string = '\n'.join([
            r'$A=%.2f$' % self.params[0],
            r'$u_0=%.2f$' % self.params[1],
            r'$t_0=%.2f$' % self.params[2],
            r'$t_\mathrm{E}=%.2f$' % self.params[3]
        ])

        ax.text(0.05, 0.95, mag_fit_string, transform=ax.transAxes, fontsize=14,
        verticalalignment='top')


    def plot_f(self, ax, title):
        #F = 10^((m_z - m)/2.5)
        for i, dm in enumerate(self.df[2]):
            exp = 10**((self.m_z - self.df[1, i])/2.5)
            ddm = exp * np.log(10) * (-1/2.5)
            self.df[2, i] *= ddm
            
        self.df[1] = 10**((self.m_z - self.df[1])/2.5)

        def mag(t, A, u0, t0, tE):
            u = np.sqrt(u0**2 + ((t-t0)/tE)**2)
            m = (u**2 + 2)/(u * np.sqrt(u**2 + 4))
            return A * m

        maxm, minm = np.max(self.df[1]), np.min(self.df[1])
        u0_g = (maxm / minm)**(-1)
        tE_g = 0.5*u0_g**(-1) * 0.1*np.ptp(self.df[0])
        guess =  [minm, u0_g, self.df[0][np.argmax(self.df[1])], tE_g]
        self.params, parms_covariance = optimize.curve_fit(mag, self.df[0], self.df[1], guess)
        t = np.linspace(np.min(self.df[0]), np.max(self.df[0]), 1000)
        fit = mag(t, *self.params)

        amp = self.params[0]
        self.tb_arg, self.te_arg = 0, 0
        self.tb, self.te = t[0], t[-1]
        for i, mag in enumerate(fit[1:]):
            if mag - amp > 0.1**4 * amp:
                self.tb_arg = i
                self.tb = t[self.tb_arg]
                break

        for i, mag in enumerate(fit[-1::-1]):
            if mag - amp > 0.1**4 * amp:
                self.te_arg = len(t) - i - 1
                self.te = t[self.te_arg]
                break

        # print(self.tb_arg, self.te_arg)
        for i in [self.tb, self.te]: ax.axvline(i, linestyle='--', color='red')

        self.hjdb_arg = np.argmin(abs(self.df[0] - self.tb))
        self.hjde_arg = np.argmin(abs(self.df[0] - self.te))
        t_const = np.append(self.df[0][0:self.hjdb_arg], self.df[0][self.hjde_arg:])
        self.mag_const = np.append(self.df[1][0:self.hjdb_arg], self.df[1][self.hjde_arg:])
        self.err_const = np.append(self.df[2][0:self.hjdb_arg], self.df[2][self.hjde_arg:])

        m_max, m_min = np.argmax(self.df[1]), np.argmin(self.df[1])
        i_max, i_min = self.df[1][m_max] + self.df[2][m_max], self.df[1][m_min] - self.df[2][m_max]
        sep = 0.1*(i_max-i_min)

        # t = np.linspace(np.min(self.df[0]), np.max(self.df[0]), 100)
        # ax.plot(t, mag(t, *guess))
        ax.plot(t, fit, linewidth=0.7)
        # ax.plot(t, mag(t, *self.params))

        ax.set_title(r'OGLE-2019 $\rightarrow$ ' + title)
        # ax.set_ylim(i_min-sep, i_max+sep)
        ax.set_xlabel("HJD - 2450000")
        ax.set_ylabel(r'$I$ flux')
        #ax.scatter(self.df[0], self.df[1], marker='.')
        ax.errorbar(self.df[0], self.df[1], self.df[2], fmt='none', color='black', linewidth=0.5)

        mag_fit_string = '\n'.join([
            r'$A=%.2f$' % self.params[0],
            r'$u_0=%.2f$' % self.params[1],
            r'$t_0=%.2f$' % self.params[2],
            r'$t_\mathrm{E}=%.2f$' % self.params[3]
        ])

        ax.text(0.05, 0.95, mag_fit_string, transform=ax.transAxes, fontsize=14,
        verticalalignment='top')


    def freq_dist(self, ax):
        
        mags = np.sort(self.mag_const)

        self.meas = np.array([]); self.freqs = np.array([])
        for i, mag in enumerate(mags):
            self.freqs = np.append(self.freqs, 1)
            if i < mags.size - 1 and mag == mags[i+1]:
                self.freqs = np.delete(self.freqs, -1)
                self.freqs[-1] += 1
                continue
            self.meas = np.append(self.meas, mag)

        ax.set_title("Frequency Distribution")
        ax.set_ylabel("Frequency")
        ax.set_xlabel("Self.Measurement")
        ax.scatter(self.meas, self.freqs)

        median, mean = np.median(mags), np.mean(mags)
        ax.axvline(median, color='red')
        ax.axvline(mean, linestyle='--', color='blue')

        sym_str = r'$\bar{x} - \tilde{x} =%.2E$' % (mean - median)
        ax.text(0.05, 0.5, sym_str, transform=ax.transAxes, fontsize=14,
                verticalalignment='top')


    def cumu_dist(self, ax):

        def cdf(measurements):
            probs = np.array([]); meas = np.array([])
            for i, mag in enumerate(measurements, start=1):
                if i < measurements.size and mag == measurements[i]: continue
                probs = np.append(probs, i/measurements.size)
                meas = np.append(meas, mag)
            return [meas, probs]

        mags = np.sort(self.mag_const)
        meas, meas_cdf = cdf(mags)
        
        # print(self.mag_const.size, self.model_dist.size)
        model_meas, model_cdf = cdf(self.model_dist)
        ax.scatter(model_meas, model_cdf, marker='.')

        ax.step(meas, meas_cdf, where='mid', marker='.', color='black')
        for i in [0,1]: ax.axhline(i, linestyle='--', color='gray', linewidth='0.5')
        ax.set_title("Cumulative Distribution Function of Constant Magnitude")
        ax.set_xlabel("Measurement")
        ax.set_ylabel(r'$P(X\leq x)$')

        def ks_test(meas_cdf, model_cdf, meas, model):
            if meas_cdf.size > model_cdf.size:
                meas_cdf = meas_cdf[0:model_cdf.size]
            elif model_cdf.size > meas_cdf.size:
                model_cdf = model_cdf[0:meas_cdf.size]
            
            supremum = np.max(abs(model_cdf - meas_cdf))
            sup_arg = np.argmax(abs(model_cdf - meas_cdf))
            return [sup_arg, supremum]

        sup_arg, supremum = ks_test(meas_cdf, model_cdf, meas, model_meas)
        meas_sup = [meas[sup_arg], model_meas[sup_arg]]
        cdf_sup = [meas_cdf[sup_arg], model_cdf[sup_arg]]
        ax.plot(meas_sup, cdf_sup, linewidth=1.2, color='red')
        

        sup_str = r'$\Delta S = %.3f$' % supremum 
        ax.text(0.05, 0.95, sup_str, transform=ax.transAxes, fontsize=14,
                verticalalignment='top')
        

    def qqplot(self, ax):
        mags = np.sort(self.mag_const)
        model_mags = np.sort(self.model_dist)

        if mags.size > model_mags.size:
            mags = mags[0:model_mags.size]
        elif model_mags.size > mags.size:
            model_mags = model_mags[0:mags.size]

        ax.set_aspect('equal')
        ax.scatter(self.model_dist, mags, color='black', marker='.')
        ax.plot(self.model_dist, self.model_dist, linestyle='--', color='green', linewidth=0.8)
        ax.set_title('Q-Q Plot')
        ax.set_xlabel("Expected Measurement")
        ax.set_ylabel("Measurement")

    
        
    def gauss_model(self, bins, bars, ax):

        def gaussian(x, amp, variance, expectation):
            a = 1/(variance * np.sqrt(2*np.pi))
            e = np.exp(-0.5 * ((x - expectation)/variance)**2)
            return amp*a*e

        guess = [2, np.ptp(bins)/10, np.average(bins, weights=bars)]
        params, parms_covariance = optimize.curve_fit(gaussian, bins, bars, guess)

        gauss_fit_string = '\n'.join([
            r'$A=%.2f$' % params[0],
            r'$\sigma=%.2f$' % params[1],
            r'$\mu=%.2f$' % params[2],
        ])

        model_freqs = gaussian(bins, *params)
        # ax.plot(self.bins, gaussian(self.bins, *guess))
        ax.plot(bins, model_freqs)
        ax.axvline(params[2], linestyle='--', linewidth=0.5)

        ax.text(0.05, 0.95, gauss_fit_string, transform=ax.transAxes, fontsize=14,
                verticalalignment='top')

        gauss_formula_string = r'$\mathrm{G(x)}=\frac{A}{\sigma\sqrt{2\pi}}e^{-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2}$'
        ax.text(0.65, 0.95, gauss_formula_string, transform=ax.transAxes, fontsize=12, verticalalignment='top')

        self.model_dist = np.array([])
        for freq, meas in zip(model_freqs, bins):
            j = 1
            while j <= freq:
                self.model_dist = np.append(self.model_dist, meas)
                j+=1


    def student_model(self, bins, bars, ax):
        
        def formula_gamma(t, A, nu):
            def gamma(x): return math.factorial(x-1)
            num = gamma((nu + 1)/2)
            den = np.sqrt(nu*np.pi) * gamma(nu / 2)
            power = (1 + t**2/nu)**(-(nu+1)/2)
            return A * (num/den) * power

        def formula(t, A, nu, mu):
            def beta(x,y):
                integrand = lambda u: u**(x-1) * (1-u)**(y-1)
                integral, err = integrate.quad(integrand, 0, 1)
                return integral
            den = np.sqrt(nu) * beta(0.5, nu/2)
            power = (1 + (t-mu)**2/nu)**(-(nu+1)/2)
            return A * (1/den) * power

        amp = np.e * np.max(bars); texp = bins[np.argmax(bars)]
        texp = np.average(bins, weights=bars)
        guess = np.array([amp, 5, texp])
        stretch = 0.8
        bounds = ([stretch * amp, 0.03, stretch * texp], [np.inf, np.inf, texp / stretch])
                
        params, parms_covariance = optimize.curve_fit(formula, bins, bars, guess, bounds=bounds, method='trf')

        fit_string = '\n'.join([
            r'$A=%.2f$' % params[0],
            r'$\nu=%.2f$' % params[1],
            r'$\mu=%.2f$' % params[2],
        ])
        
        #print(np.max(formula(bins, *params)))
        t = np.linspace(5, 20, 1000)
        ax.plot(t, formula(t, *guess))
        ax.plot(t, formula(t, *params))

        ax.text(0.05, 0.95, fit_string, transform=ax.transAxes, fontsize=14,
                verticalalignment='top')

        formula_string = r'$\frac{\Gamma(\frac{\nu +1}{2})}{\sqrt{\nu \pi}\ \Gamma (\frac{\nu}{2})} \left(1+\frac{t^2}{\nu}\right)^{-\frac{\nu +1}{2}}$'
        formula_string = r'$\frac{1}{\sqrt{\nu}\mathrm{B}(\frac{1}{2}, \frac{\nu}{2})}\left(1+\frac{t^2 - \mu}{\nu}\right)^{-\frac{\nu+1}{2}}$'
        ax.text(0.65, 0.95, formula_string, transform=ax.transAxes, fontsize=12, verticalalignment='top')


    def const_dist(self, ax1, ax2):

        def bin(arr, no_bins):
            bins = np.linspace(np.min(arr), np.max(arr), no_bins)
            bins_len = len(bins)
            bars = np.zeros(bins_len)
            if len(arr) < bins_len: print("too many bins")
            for i in arr:
                for j in range(bins_len - 1):
                    if bins[j] <= i < bins[j+1]: bars[j] += 1
                if i == bins[bins_len-1]: bars[bins_len-1] += 1
            return(bars, bins)


        self.bars, self.bins = np.histogram(self.mag_const, 80)
        #self.bars, self.bins = bin(self.mag_const, 100)
        self.bins = np.delete(self.bins, -1)

        # ax1.scatter(self.bins, self.bars, marker='_')
        ax1.set_title("Constant Magnitude Distribution")
        ax2.set_xlabel("Magnitude")
        ax2.set_ylabel("Frequency")

        #ax2.hist(self.mag_const+self.err_const, self.bins, color='lightblue')
        #ax2.hist(self.mag_const-self.err_const, self.bins, color='pink')
        ax2.hist(self.mag_const, self.bins, color='gray')
        median, mean = np.median(self.mag_const), np.mean(self.mag_const)
        ax2.axvline(np.median(self.mag_const), color='red')
        ax2.axvline(np.mean(self.mag_const), linestyle='--', color='blue')

        ax1.boxplot(self.mag_const, vert=False)

        sym_str = r'$\bar{x} - \tilde{x} =%.2E$' % (mean - median)
        ax2.text(0.05, 0.5, sym_str, transform=ax2.transAxes, fontsize=14,
                verticalalignment='top')


    def model_deviation(self, ax1, ax2):
        t = self.df[0, self.hjdb_arg:self.hjde_arg]
        mag_curve = self.df[1, self.hjdb_arg:self.hjde_arg]
        model_curve = mag(t, *self.params)
        diffs = mag_curve - model_curve
        errs = self.df[2, self.hjdb_arg:self.hjde_arg]
        self.bars_c, self.bins_c = np.histogram(diffs, 100)
        self.bins_c = np.delete(self.bins_c, -1)

        ax1.set_title("Model Deviation Distribution")
        ax2.set_xlabel("Magnitude Difference")
        ax2.set_ylabel("Frequency")

        ax2.hist(diffs+errs, self.bins_c, color='lightblue')
        ax2.hist(diffs-errs, self.bins_c, color='pink')
        ax2.hist(diffs, self.bins_c, color='gray')

        ax1.boxplot(diffs, vert=False)

