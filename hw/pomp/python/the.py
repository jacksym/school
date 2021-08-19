import numpy as np
import matplotlib.pyplot as plt

def sinc(n, a):
    An = (1/(n*np.pi))*np.sin((2*np.pi*n)/a)
    return An

def dB(amp):
    dB = 20*np.log10(abs(amp))
    return dB

f0=82
harms = np.arange(1,30)
f = np.linspace(1,30,300)

fig, ax = plt.subplots()
ax.plot(f,dB(sinc(f,12)))
for harm in harms: ax.axvline(harm,ymax=dB(sinc(harm,12)))
ax.set_ylabel('Amplitude in dB')
ax.set_xlabel('Harmonic Number')
ax.set_yticks([])

fig.savefig('../images/theory.png', quality=100)
plt.show()

