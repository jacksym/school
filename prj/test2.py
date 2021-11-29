import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize, integrate


def gaussian(x, amp, variance, expectation):
    a = 1/(variance * np.sqrt(2*np.pi))
    e = np.exp(-0.5 * ((x - expectation)/variance)**2)
    return amp*a*e


def formula(t, A, nu, mu):
    def beta(x,y):
        integrand = lambda u: u**(x-1) * (1-u)**(y-1)
        integral, err = integrate.quad(integrand, 0, 1)
        return integral
    den = np.sqrt(nu)*beta(0.5, nu/2)
    power = (1 + (t-mu)**2/nu)**(-(nu+1)/2)
    return A * (1/den) * power


t = np.linspace(10, 20, 100)
y = gaussian(t, 100, 0.01, 16)

amp = np.e * np.max(y); texp = t[np.argmax(y)]
guess_stu = np.array([np.e*np.max(y), 500, t[np.argmax(y)]])
stretch = 0.8
bounds = ([stretch * amp, 0.1, stretch * texp], [np.inf, np.inf, texp / stretch])

# bounds = ([1, 2, 1], [100, 10, np.max(t)])
params, parms_covariance = optimize.curve_fit(formula, t, y, guess_stu, bounds=bounds, method='trf')
# params, parms_covariance = optimize.curve_fit(formula, t, y, guess_stu, bounds=bounds)
# params, parms_covariance = optimize.curve_fit(formula, t, y, guess_stu)
print(params)

plt.plot(t, formula(t, *params), label=r'Opt Student $t$-dist')


plt.plot(t, y, label='Gaussian')
# plt.plot(t, formula(t, *guess_stu))
plt.legend()
plt.show()
