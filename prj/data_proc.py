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
            
        self.y_string = 0.95
    
    def plot(self, ax, title):
        
        def mag(t, Fa, u0, t0, tE, Fb):
            u = np.sqrt(u0**2 + ((t-t0)/tE)**2)
            m = (u**2 + 2)/(u * np.sqrt(u**2 + 4))
            f = Fa * m 
            mag_mag = -2.5*np.log10(m) + Fb
            return mag_mag

        u0_g = np.min(self.df[1])**(-1)
        tE_g = 0.5*u0_g**(-1) * 0.1*np.ptp(self.df[0])
        # print(u0_g, tE_g)
        guess =  [np.mean(self.df[1]), u0_g, self.df[0][np.argmin(self.df[1])], tE_g, 0]
        self.params, parms_covariance = optimize.curve_fit(mag, self.df[0], self.df[1], guess)
        t = np.linspace(np.min(self.df[0]), np.max(self.df[0]), 1000)
        fit = mag(t, *self.params)

        amp = self.params[4]
        self.tb_arg, self.te_arg = 0, 0
        self.tb, self.te = 0, 0
        for i, mag in enumerate(fit[1:]):
            if amp - mag > 0.1**4 * amp:
                self.tb_arg = i
                self.tb = t[self.tb_arg]
                break

        print("hello?")
        for i, mag in enumerate(fit[-2::-1], start=2):
            if amp - mag > 0.1**4 * amp:
                self.te_arg = len(t) - i
                self.te = t[self.te_arg]
                break

        for i in [self.tb, self.te]: ax.axvline(i, linestyle='--', color='gray')

        self.hjdb_arg = np.argmin(abs(self.df[0] - t[self.tb_arg]))
        self.hjde_arg = np.argmin(abs(self.df[0] - t[self.te_arg]))
        t_const = np.append(self.df[0][0:self.hjdb_arg], self.df[0][self.hjde_arg:])

        self.meas_const = np.append( self.df[1][0:self.hjdb_arg], self.df[1][self.hjde_arg:])
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
            r'$F_a=%.2f$' % self.params[0],
            r'$u_0=%.2f$' % self.params[1],
            r'$t_0=%.2f$' % self.params[2],
            r'$t_\mathrm{E}=%.2f$' % self.params[3],
            r'$F_b=%.2E$' % self.params[4],
        ])

        ax.text(0.05, 0.95, mag_fit_string, transform=ax.transAxes, fontsize=12,
        verticalalignment='top')

        self.mean_const = np.mean(self.meas_const)
        
        self.ress = np.array([])
        for i, e in zip(self.meas_const, self.err_const):
            # res = abs((i - self.mean_const) / e )
            res = (i - self.mean_const) / abs(e)
            self.ress = np.append(self.ress, res)
        self.ress = np.sort(self.ress)

        # self.diffs = np.array([])
        # for i in self.meas_const:
        #     diff = (i - self.mean_const)
        #     self.diffs = np.append(self.diffs, diff)
        self.diffs = self.meas_const - self.mean_const
        

    def plot_f(self, ax, title):
        #F = 10^((m_z - m)/2.5)
        for i, dm in enumerate(self.df[2]):
            exp = 10**((self.m_z - self.df[1, i])/2.5)
            ddm = exp * np.log(10) * (-1/2.5)
            self.df[2, i] *= ddm
            
        self.df[1] = 10**((self.m_z - self.df[1])/2.5)

        def mag(t, Fa, u0, t0, tE, Fb):
            u = np.sqrt(u0**2 + ((t-t0)/tE)**2)
            m = (u**2 + 2)/(u * np.sqrt(u**2 + 4))
            return Fa * m + Fb

        maxm, minm = np.max(self.df[1]), np.min(self.df[1])
        u0_g = (maxm / minm)**(-1)
        tE_g = 0.5*u0_g**(-1) * 0.1*np.ptp(self.df[0])
        guess =  [minm, u0_g, self.df[0][np.argmax(self.df[1])], tE_g, 0]
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

        self.meas_const = np.append(self.df[1][0:self.hjdb_arg], self.df[1][self.hjde_arg:])
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
            r'$Fa=%.2f$' % self.params[0],
            r'$u_0=%.2f$' % self.params[1],
            r'$t_0=%.2f$' % self.params[2],
            r'$t_\mathrm{E}=%.2f$' % self.params[3],
            r'$Fb=%.2f$' % self.params[4],
        ])

        ax.text(0.05, 0.95, mag_fit_string, transform=ax.transAxes, fontsize=12,
        verticalalignment='top')

        self.mean_const = np.mean(self.meas_const)
        
        self.ress = np.array([])
        for i, e in zip(self.meas_const, self.err_const):
            # res = abs((i - self.mean_const) / e)
            res = (i - self.mean_const) / abs(e)
            self.ress = np.append(self.ress, res)
        self.ress = np.sort(self.ress)

        # self.diffs = np.array([])
        # for i in self.meas_const:
        #     diff = (i - self.mean_const)
        #     self.diffs = np.append(self.diffs, diff)

        self.diffs = self.meas_const - self.mean_const


    def freq_dist_r(self, ax):
        
        meas_sor = np.sort(self.meas_const)

        self.meas = np.array([meas_sor[0]]); self.freqs = np.array([1])
        for i, mag in enumerate(meas_sor[1:], start=1):
            self.freqs = np.append(self.freqs, 1)
            if mag == meas_sor[i-1]:
                self.freqs = np.delete(self.freqs, -1)
                self.freqs[-1] += 1
                continue
            self.meas = np.append(self.meas, mag)

        ax.set_title("Frequency Distribution")
        ax.set_ylabel("Frequency")
        ax.set_xlabel("Measurement")
        # ax.step(self.meas, self.freqs, where='mid')
        ax.scatter(self.meas, self.freqs, marker='.', color='black')

        median, mean = np.median(meas_sor), np.mean(meas_sor)
        ax.axvline(median, color='red')
        ax.axvline(mean, linestyle='--', color='blue')

        sym_str = r'${\bar{x}} - {\tilde{x}} =%.2E$' % (mean - median)
        ax.text(0.05, 0.5, sym_str, transform=ax.transAxes, fontsize=12,
                verticalalignment='top')


    def freq_dist(self, ax):

        meas_sor = self.ress
        meas_sor = abs(self.ress)

        self.meas = np.array([self.ress[0]]); self.freqs = np.array([1])
        for i, mag in enumerate(self.ress[1:], start=1):
            self.freqs = np.append(self.freqs, 1)
            if mag == self.ress[i-1]:
                self.freqs = np.delete(self.freqs, -1)
                self.freqs[-1] += 1
                continue
            self.meas = np.append(self.meas, mag)

        # area = np.sum(abs(self.ress))
        # self.freqs = self.freqs / area

        ax.set_title("Frequency Distribution")
        ax.set_ylabel("Frequency")
        # ax.set_xlabel("Measurement Deviation from Mean / Error")
        ax.set_xlabel(r'$|x - \bar{x}|/ \Delta x$')
        # ax.step(self.meas, self.freqs, where='mid')
        ax.scatter(self.meas, self.freqs, marker='.', color='black')

        median, mean = np.median(meas_sor), np.mean(meas_sor)
        ax.axvline(median, color='red')
        ax.axvline(mean, linestyle='--', color='blue')

        sym_str = r'${\bar{x}} - {\tilde{x}} =%.2E$' % (mean - median)
        ax.text(0.65, 0.5, sym_str, transform=ax.transAxes, fontsize=12,
                verticalalignment='top')

        ax.set_ylim(0)
        # ax.set_xlim(0)


    def man_err(self, diffs, err, c):
        nerr = np.array([np.sqrt(i**2 + c**2) for i in err])
        new_ress = np.array([i / j for i, j in zip(diffs, nerr)])
        return new_ress


    def cumu_dist(self, meas, ax):

        def cdf(measurements):
            probs = np.array([]); meas = np.array([])
            for i, mag in enumerate(measurements, start=1):
                if i < measurements.size and mag == measurements[i]: continue
                probs = np.append(probs, i/measurements.size)
                meas = np.append(meas, mag)
            return [meas, probs]

        meas = np.sort(meas)
        self.meas_c, self.cdf = cdf(meas)
        
        ax.step(self.meas_c, self.cdf, where='mid', color='black')
        # ax.scatter(self.meas_c, self.cdf, marker='.', color='black')
        for i in [0,1]: ax.axhline(i, linestyle='--', color='gray', linewidth='0.5')
        ax.set_title("Cumulative Distribution Function")
        ax.set_xlabel(r'$(x - \bar{x})/|\Delta x|$')
        ax.set_ylabel(r'$P(X\leq x)$')
        ax.grid(True)
        # ax.set_xlim(0)
        ax.set_ylim(0)
        

    def qqplot(self, ax, model):
        # mags = np.sort(self.meas_const)
        mags = np.sort(self.ress)
        model_mags = np.sort(model)

        if mags.size > model_mags.size:
            mags = mags[0:model_mags.size]
        elif model_mags.size > mags.size:
            model_mags = model_mags[0:mags.size]

        ax.set_aspect('equal')
        ax.scatter(model, mags, color='black', marker='.')
        ax.plot(model, model, linestyle='--', color='green', linewidth=0.8)
        ax.set_title('Q-Q Plot')
        ax.set_xlabel("Expected Measurement")
        ax.set_ylabel("Measurement")

        
    def gauss_model(self, bins, bars, ax):

        def formula(x, amp, variance, expectation):
            a = 1/(variance * np.sqrt(2*np.pi))
            e = np.exp(-0.5 * ((x - expectation)/variance)**2)
            return amp*a*e

        guess = [2, np.ptp(bins)/10, np.average(bins, weights=bars)]
        params, parms_covariance = optimize.curve_fit(formula, bins, bars, guess)

        gauss_fit_string = '\n'.join([
            r'$A=%.2f$' % params[0],
            r'$\sigma=%.2f$' % params[1],
            r'$\mu=%.2f$' % params[2],
        ])

        model_freqs = formula(bins, *params)
        # ax.plot(self.bins, formula(self.bins, *guess))
        t = np.linspace(np.min(bins), np.max(bins), 1000)
        ax.plot(t, formula(t, *params))
        ax.axvline(params[2], linestyle='--', linewidth=0.5)

        ax.text(0.05, 0.95, gauss_fit_string, transform=ax.transAxes, fontsize=12,
                verticalalignment='top')

        gauss_formula_string = r'$\mathrm{G(x)}=\frac{A}{\sigma\sqrt{2\pi}}e^{-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2}$'
        ax.text(0.65, 0.95, gauss_formula_string, transform=ax.transAxes, fontsize=12, verticalalignment='top')

        self.gauss_model_dist = np.array([])
        for freq, meas in zip(model_freqs, bins):
            j = 1
            while j <= freq:
                self.gauss_model_dist = np.append(self.gauss_model_dist, meas)
                j+=1

        return lambda x: formula(x, *params)


    def gauss_model_c(self, meas, probs, ax):
    
        def formula(x, mu, sigma):
            frac = 1 / np.sqrt(2*np.pi)
            integrand = lambda t: np.exp(-t**2 / 2)
            phi = np.array([])
            x = (x-mu)/sigma
            for i in x:
                integral, err = integrate.quad(integrand, -np.inf, i)
                phi = np.append(phi, frac*integral)
            return phi

        # guess = [0, np.average(meas, weights=probs)]
        guess = [np.mean(meas), 1]
        params, parms_covariance = optimize.curve_fit(formula, meas, probs, guess)
        # params[1] += 0.2

        self.gauss_model_cdf = formula(meas, *params)
        # ax.scatter(meas, model, color='green', marker='.')
        # ax.step(meas, self.gauss_model_cdf, color='green', marker='.', where='mid', linewidth=0.5)
        ax.plot(meas, self.gauss_model_cdf, color='green', linewidth=0.8, label="Normal")

        self.gauss_fit_string = '\n'.join([
            r'$\mu=%.2f$' % params[0],
            r'$\sigma=%.2f$' % params[1],
        ])

        ax.text(0.05, self.y_string, self.gauss_fit_string, transform=ax.transAxes,
                fontsize=12, verticalalignment='top')
        self.y_string -= 0.2
        # gauss_formula_string = r'$\mathrm{G(x)}=\frac{A}{\sigma\sqrt{2\pi}}e^{-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2}$'
        # ax.text(0.65, 0.95, gauss_formula_string, transform=ax.transAxes, fontsize=12, verticalalignment='top')
        return lambda x: formula(x, *params)


    def gauss_model_ec(self, diffs, ax):

        def cdf(measurement, measurements):
            measurements = np.sort(measurements)
            i=0; elaps = 0
            while i < len(measurements) and measurements[i] <= measurement:
                elaps +=1
                i+=1
            return elaps / len(measurements)
    
        def formula(x, mu, sigma):
            frac = 1 / np.sqrt(2*np.pi)
            integrand = lambda t: np.exp(-t**2 / 2)
            x = (x-mu)/sigma
            integral, err = integrate.quad(integrand, -np.inf, x)
            phi = frac * integral
            return phi

        def sum_squares(variables):
            c, mu, sigma = variables
            nerr = np.array([np.sqrt(x**2 + c**2) for x in self.err_const])
            ress = np.array([i/j for i,j in zip(diffs, nerr)])
            # weight = lambda x, mu: (x-mu)**2 + 1
            def weight(x, a, alpha, beta):
                b = ((beta -1)/alpha**2) * (x-a)**2 + 1
                return b
            
            max = np.max(ress)
            sum=0
            for i in ress:
                square = formula(i, mu, sigma) - cdf(i, ress)
                summand = square**2 * weight(i, mu, max, 4)
                sum += summand
            return sum

        guess = (0, 0, 1)
        bounds = ((0, 1), (-2, 2), (0, 10))
        # options = {'maxiter=100'}
        res = optimize.minimize(sum_squares, guess, bounds=bounds, method='tnc', tol=10**(-4))
        cp, mup, sigmap = res['x']

        nerr = np.array([np.sqrt(x**2 + cp**2) for x in self.err_const])
        ress = np.array([i/j for i,j in zip(diffs, nerr)])
        ress = np.sort(ress)
        ecdf = np.array([cdf(i, ress) for i in ress])
        gcdf = np.array([formula(i, mup, sigmap) for i in ress])

        ax.step(ress, ecdf, where='mid', color='black')
        ax.plot(ress, gcdf, linewidth=0.8, color='green')
        ax.set_title('Cumulative Distribution Function with Corrected Error')
        ax.set_xlabel(r'$(x-\bar x)/\sqrt{\Delta x^2 + C ^2}$')
        ax.set_ylabel(r'$P(X\leq x)$')
        ax.grid(True)

        self.gauss_ec_fit_string = '\n'.join([
            r'$C = %.2E$' % cp,
            r'$\mu = %.2E$' % mup,
            r'$\sigma = %.2E$' % sigmap,
        ])

        ax.text(0.05, 0.95, self.gauss_ec_fit_string,
                transform=ax.transAxes, fontsize=12, verticalalignment='top')

        self.meas_ec = ress; self.cdf_ec = ecdf
        self.cp = cp
        return lambda x: np.array([formula(i, mup, sigmap) for i in x])


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
        guess = np.array([amp, 10, texp])
        stretch = 0.8
        bounds = ([stretch * amp, 0.03, stretch * texp], [np.inf, np.inf, texp / stretch])
                
        params, parms_covariance = optimize.curve_fit(formula, bins, bars, guess, bounds=bounds, method='trf')

        fit_string = '\n'.join([
            r'$A=%.2f$' % params[0],
            r'$\nu=%.2f$' % params[1],
            r'$\mu=%.2f$' % params[2],
        ])

        # fit_string = '\n'.join([
        #     r'$A=%.2f$' % guess[0],
        #     r'$\nu=%.2f$' % guess[1],
        #     r'$\mu=%.2f$' % guess[2],
        # ])
        
        model_freqs = formula(bins, *params)
        # print(np.max(formula(bins, *params)))
        t = np.linspace(0, 20, 1000)
        ax.plot(t, formula(t, *guess))

        ax.text(0.05, 0.95, fit_string, transform=ax.transAxes, fontsize=12,
                verticalalignment='top')

        formula_string = r'$\frac{\Gamma(\frac{\nu +1}{2})}{\sqrt{\nu \pi}\ \Gamma (\frac{\nu}{2})} \left(1+\frac{t^2}{\nu}\right)^{-\frac{\nu +1}{2}}$'
        formula_string = r'$\frac{1}{\sqrt{\nu}\mathrm{B}(\frac{1}{2}, \frac{\nu}{2})}\left(1+\frac{t^2 - \mu}{\nu}\right)^{-\frac{\nu+1}{2}}$'
        ax.text(0.65, 0.95, formula_string, transform=ax.transAxes, fontsize=12, verticalalignment='top')

        self.student_model_dist = np.array([])
        for freq, meas in zip(model_freqs, bins):
            j = 1
            while j <= freq:
                self.student_model_dist = np.append(self.student_model_dist, meas)
                j+=1

        return lambda x: formula(x, *params)


    def student_model_c(self, meas, probs, ax):

        def formula(t, nu, mu):
            def beta(x, y):
                integrand = lambda t: t**(x-1) * (1-t)**(y-1)
                # x = np.linspace(0, 1, 15)[1:-1]
                # integral = np.trapz(integrand(x), x=x)
                integral, err = integrate.quad(integrand, 0, 1)
                return integral
            pdf = lambda u: (1/(np.sqrt(nu)*beta(0.5, (nu/2)))) * (1+((u-mu)**2)/nu) ** (-(nu+1)/2)
            # newt = np.linspace(np.min(t), np.max(t), 1000)
            # newt = np.sort(t)
            cdf = np.array([])
            lastt = np.min(t); lastc = 0
            for i in np.sort(t):
                integral, err = integrate.quad(pdf, lastt, i)
                cdf_value = integral + lastc
                cdf = np.append(cdf, cdf_value)
                lastt = i; lastc = cdf_value
            return cdf

        bounds = (-1, 1)
        mup, covariance = optimize.curve_fit(lambda t, mu: formula(t, 1, mu), meas, probs, 0, bounds=bounds)

        squares = []
        for i in range(1, 10):
            diffs = probs - formula(meas, i, mup)
            diffs2 = np.array([x**2 for x in diffs])
            squares.append([i, np.sum(diffs2)])
        squares = np.array(squares)
        nup, least_square = squares[np.argmin(squares, axis=1)[1]]

        params = [nup, mup]

        self.student_model_cdf = formula(meas, *params)
        ax.plot(meas, self.student_model_cdf, color='brown', linewidth=0.8, label="t-distribution")
        # ax.plot(meas, formula(meas, *guess), color='green', marker='.', linewidth=0.5)

        self.student_fit_string = '\n'.join([
            r'$\nu=%.2f$' % params[0],
            r'$\mu=%.2f$' % params[1],
        ])

        ax.text(0.05, 0.95, self.student_fit_string, transform=ax.transAxes,
                fontsize=12, verticalalignment='top')

        return lambda x: formula(x, *params)


    def cauchy_model(self, bins, bars, ax):
        
        def formula(x, A, x0, gamma):
            frac = 1/(gamma * np.pi)
            prod = gamma**2/((x - x0)**2 + gamma**2)
            return A * frac * prod

        A_g = np.max(bars)
        # x0_g = np.average(bins, weights=bars)
        x0_g = 0
        gamma_g = np.ptp(bins) / 3
        guess = np.array([A_g, x0_g, gamma_g])
        # bounds = ([], [])
                
        # params, parms_covariance = optimize.curve_fit(formula, bins, bars, guess, bounds=bounds, method='trf')
        params, parms_covariance = optimize.curve_fit(formula, bins, bars, guess)

        fit_string = '\n'.join([
            r'$A=%.2f$' % params[0],
            r'$x_0=%.2f$' % params[1],
            r'$\gamma=%.2f$' % params[2],
        ])

        model_freqs = formula(bins, *params)
        # print(np.max(formula(bins, *params)))
        t = np.linspace(np.min(bins), np.max(bins), 1000)
        ax.plot(t, formula(t, *guess))

        ax.text(0.05, 0.95, fit_string, transform=ax.transAxes, fontsize=12,
                verticalalignment='top')

        # formula_string = r'$\frac{1}{\pi\gamma \left[ 1+(\frac{x-x_0}{\gamma})^2\right]}$'
        formula_string = r'$\frac{1}{\pi\gamma}\left[\frac{\gamma ^2}{(x-x_0)^2+\gamma ^2}\right]$'
        ax.text(0.65, 0.95, formula_string, transform=ax.transAxes, fontsize=12, verticalalignment='top')

        self.cauchy_model_dist = np.array([])
        for freq, meas in zip(model_freqs, bins):
            j = 1
            while j <= freq:
                self.cauchy_model_dist = np.append(self.cauchy_model_dist, meas)
                j+=1

        return lambda x: formula(x, *params)


    def cauchy_model_c(self, meas, probs, ax):

        def formula(x, x0, gamma):
            frac = 1 / np.pi
            arctan = np.arctan((x-x0)/gamma)
            return frac * arctan + 0.5

        guess = [np.mean(meas), 1]
        params, parms_covariance = optimize.curve_fit(formula, meas, probs, guess)

        self.cauchy_model_cdf = formula(meas, *params)
        # ax.scatter(meas, model, color='green', marker='.')
        ax.plot(meas, self.cauchy_model_cdf, color='orange', linewidth=0.8, label='Cauchy')

        self.cauchy_fit_string = '\n'.join([
            r'$x_0=%.2f$' % params[0],
            r'$\gamma=%.2f$' % params[1],
        ])

        # self.cauchy_fit_string = '\n'.join([
        #     r'$A=%.2f$' % params[0],
        #     r'$\gamma=%.2f$' % params[1],
        # ])

        ax.text(0.05, self.y_string, self.cauchy_fit_string, transform=ax.transAxes,
                fontsize=12, verticalalignment='top')

        return lambda x: formula(x, *params)

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


    def ks_test(self, meas, meas_cdf, model, ax):
        mody = model(meas)

        def find_infl(x, y):
            secret_x = np.linspace(np.min(x), np.max(x), 4*len(x))
            secret_y = model(secret_x)
            d2 = np.gradient(np.gradient(secret_y))
            infls_args = np.where(np.diff(np.sign(d2)))[0]
            x_infls = []
            for i in infls_args: x_infls.append(secret_x[i])
            true_x_infls_arg = []; true_x_infls = []
            for i in x_infls:
                arg = np.argmin(abs(x - i))
                true_x_infls_arg.append(arg)
                true_x_infls.append(x[arg])
            return [true_x_infls_arg[0], true_x_infls[0]]

        def find_sup(measy, mody, infl):
            meas_cdf = measy[0:infl]
            model_cdf = mody[0:infl]

            pdiffs = (model_cdf - meas_cdf)
            psupremum1 = np.max(pdiffs)
            psup_arg1 = np.argmax(pdiffs)

            ndiffs = (meas_cdf - model_cdf)
            nsupremum1 = np.max(ndiffs)
            nsup_arg1 = np.argmax(ndiffs)

            meas_cdf = measy[infl:]
            model_cdf = mody[infl:]

            pdiffs = (model_cdf - meas_cdf)
            psupremum2 = np.max(pdiffs)
            psup_arg2 = np.argmax(pdiffs) + infl

            ndiffs = (meas_cdf - model_cdf)
            nsupremum2 = np.max(ndiffs)
            nsup_arg2 = np.argmax(ndiffs) + infl

            sups = np.array([psupremum1, nsupremum1, psupremum2, nsupremum2])
            sup_args = np.array([psup_arg1, nsup_arg1, psup_arg2, nsup_arg2])

            return [sup_args-1, sups]

        infl_arg, infl = find_infl(meas, mody)    
        # print(infl_arg, infl)
        ax.axvline(infl, color='black', linewidth=0.6)

        sup_args, sups = find_sup(meas_cdf, mody, infl_arg)

        def plot_s(arg):
            meas_sup = [meas[arg], meas[arg]]
            cdf_sup = [meas_cdf[arg], mody[arg]]
            ax.plot(meas_sup, cdf_sup, linewidth=4, color='red')
            ax.axvline(meas[arg], linestyle='--', color='red', linewidth=0.7)

        for i in sup_args: plot_s(i)

        # sup_str = r'$\Delta S = %.4f$' % supremum 
        sup_str = '\n'.join([
            r'$+D_1 = %.4f$' % sups[0],
            r'$-D_1 = %.4f$' % sups[1],
            r'$+D_2 = %.4f$' % sups[2],
            r'$-D_2 = %.4f$' % sups[3],
            r'$n = %s$' % len(meas)
        ])

        ax.text(0.05, 0.50, sup_str, transform=ax.transAxes, fontsize=12,
                verticalalignment='top')


    def and_dar_test(self, meas, model, ax):
        # mody = model(self.meas_c)
        mody = model(meas)

        def ad_dist(meas, mody):
            n = len(meas) -2
            sum = 0
            for i in range(0, n):
                frac = (2*i - 1) / n
                log1 = np.log(mody[i] +10**(-5))
                log2 = np.log(1 - mody[n + 1 - i] +10**(-5))
                sum += frac * (log1 - log2)

            return -n - sum

        a2 = ad_dist(meas, mody)

        self.ad_string = r'$A ^2 = %.2f$' % a2
        ax.text(0.65, 0.40, self.ad_string, transform=ax.transAxes, fontsize=12,
                verticalalignment='top')

        return a2


    def chi_test(self, meas, model, ax):
        mody = model(meas)
        sum = 0
        k = len(meas)
        for i in range(0,k):
            summand = (meas[i] - mody[i])**2 / mody[i]
            sum += summand
        sum /= k
        self.chi_string = r'$\chi ^2 = %.E$' % sum
        ax.text(0.65, 0.55, self.chi_string, transform=ax.transAxes, fontsize=12,
                verticalalignment='top')

        return sum


def student_form(t, nu, mu):
    def beta(x,y):
        integrand = lambda u: u**(x-1) * (1-u)**(y-1)
        integral, err = integrate.quad(integrand, 0, 1)
        return integral
    den = np.sqrt(nu) * beta(0.5, nu/2)
    power = (1 + (t-mu)**2/nu)**(-(nu+1)/2)
    return 1 * (1/den) * power


def cauchy_form(x, x0, gamma):
    frac = 1/(gamma * np.pi)
    prod = gamma**2/((x - x0)**2 + gamma**2)
    return frac * prod
