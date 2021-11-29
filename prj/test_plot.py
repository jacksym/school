import numpy as np
import matplotlib.pyplot as plt

df = np.loadtxt("error_cor.dat")
mean_brights = df[:,0]
corrections = df[:,1]
# df = np.loadtxt("test_err.dat").transpose()

m, b = np.polyfit(mean_brights, corrections, 1)
fit_string = r'$C(m) \sim %.2E\ m + %.2E$' % (m, b)
plt.text(20, 0.075, fit_string)

fit = m*mean_brights + b
plt.plot(mean_brights, fit, linestyle='--', color='blue')

plt.scatter(mean_brights, corrections, color='brown', marker='.')
plt.ylim(0)
plt.xlim(np.max(mean_brights), np.min(mean_brights))
plt.show()
