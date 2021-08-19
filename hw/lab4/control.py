import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.signal as sig

class temptime:
    def __init__(self, filename):
        self.df = pd.read_csv('./pid/'+filename)

    def plot(self, ax, color,P,I, tstab,tset):
        times = self.df['Time (s)']
        temps = self.df['Temperature (K)']


        #ax.fill_between(times, temps, avgT, color='lightgrey')
        tstabi = np.argmin(times-tstab)
        ax.axvline(tstab,color=color,linestyle='--',linewidth=0.75)
        ntemps = temps[tstabi:]
        rms = np.sqrt((1/(ntemps.size-1))*np.sum((ntemps-np.mean(ntemps))**2))
        label = r'P = %.2f' % P
        ax.plot(times, temps, marker='', color=color,label=label,linewidth=0.8)
        row = [r'$P=%.2f$' % P, r'$I=%.2f$'%I, r'$t_{stab}\approx %.2f$ s'%round(tstab,2), r'$\sigma _T=%.4f$' % rms]
        cellText.append(row)


fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.16)
ax.grid(True)
ax.set_xlabel('Time (s)')
ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')
ax.set_ylabel('Temperature (K)')
cellText=[]

#pi = temptime('vp2.csv')
#pi.plot(ax, 'black',1.6, 4.4, 10)




vp2 = temptime('p3-2.csv')
vp3 = temptime('p0-8.csv')
vp4 = temptime('p2.csv')
pi = temptime('p1-6.csv')

vp4.plot(ax, 'green',2, 4.4, 40, 10)
vp3.plot(ax, 'orange',0.8, 4.4, 30, 10)
vp2.plot(ax, 'blue',3.2, 4.4, 50, 10)
pi.plot(ax, 'black',1.6, 4.4, 25, 10)

plt.legend()
ax.table(cellText=cellText, cellLoc='left')
fig.savefig('images/ptrace.png')
plt.show()

fig2, ax2 = plt.subplots()
ax2.grid(True)
ax2.set_xlabel('Time (s)')
ax2.xaxis.tick_top()
ax2.xaxis.set_label_position('top')
ax2.set_ylabel('Temperature (K)')
cellText=[]

pi2 = temptime('50Kpi.csv')
pi2.plot(ax2, 'blue', 19.2,4.8,60,50)
ax2.table(cellText=cellText, cellLoc='left')
fig2.savefig('images/50Kpi.png')
plt.show()
