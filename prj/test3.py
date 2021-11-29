import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize, integrate

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

meas = np.linspace(-6, 6, 100)
# meas = np.linspace(0, 1, 100)
# meas = np.linspace(-20, 20, 100)

# params = [9.9, 0]
# params = [5, 0]

cdf = formula(meas, 3, 0)
cdf2 = formula(meas, 5, 0)

plt.plot(meas, cdf)
plt.plot(meas, cdf2)
plt.axhline(1, linestyle='--')
plt.axhline(0, linestyle='--')
plt.show()

