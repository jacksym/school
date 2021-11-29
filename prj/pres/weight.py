import numpy as np
import matplotlib.pyplot as plt

plt.style.use('dark_background')

fig, ax = plt.subplots()
axes = 8
ax.set_xlim(-axes, axes)
ax.set_ylim(0, 10)
x = np.linspace(-axes, axes, 500)
def weight(x, a, alpha, beta):
    b = ((beta -1)/alpha**2) * (x-a)**2 + 1
    return b
a = 1; alpha = 6; beta = 4
y = weight(x, a, alpha, beta)
ax.set_xticks([-alpha, 0, a, alpha])
ax.set_xticklabels([r'$-\alpha$', '0', r'$\mu$', r'$\alpha$'])
ax.set_yticks([0, 1, beta])
ax.set_yticklabels(['0', '1', r'$\beta$'])
lw = 0.7
ax.plot(x, weight(x, 0, alpha, beta), color='gray', linewidth=lw)
ax.plot(x, y, color='white', linewidth=2)
ax.axhline(0, color='gray', linewidth=lw)
ax.axvline(0, color='gray', linewidth=lw)
for x in [a, alpha, -alpha]: ax.axvline(x, color='gray', linestyle='--', linewidth=0.7)
for y in [1, beta]: ax.axhline(y, color='gray', linestyle='--', linewidth=0.7)
ax.set_title(r'Weighting Function: $w(x; \mu, \alpha, \beta)$')
fig.savefig('./images/weight.png')
fig.show()
