import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

temps = []
rows = []
class voltemp:
    def __init__(self, filename):
        df = pd.read_csv(filename)
        self.times = np.array(df['Time (s)'])
        self.temps = np.array(df['Temperature (K)'])
        self.amps = np.array(1050*1000*df['Amplitude (V)'])

    def plot(self, ax, label):
        ax.scatter(self.temps, self.amps, label=label, marker='.')
        r1 = np.max(self.amps)
        r0 = np.min(self.amps)
        ticks = ax.get_yticks()
        nticks = [r0,0.1*r1,0.5*r1,0.9*r1,r1]
        ax.axhline(0.5*r1, color='black', linewidth=0.5)
        if np.array_equal(ticks,np.array(nticks)) != True:
            ticksn = np.append(ticks, nticks)
            ax.set_yticks(ticksn)
            for tick in ticksn:
                ax.axhline(tick, color='grey', linewidth=0.75)

        t10i = np.argmin(abs(self.amps - 0.1*r1))
        t10 = self.temps[t10i]
        ax.axvline(t10, linestyle = '--', linewidth=0.5,color = 'green')

        t50i = np.argmin(abs(self.amps - 0.5*r1))
        t50 = self.temps[t50i]
        ax.axvline(t50, linestyle = '--', linewidth=0.5,color='yellow')

        t90i = np.argmin(abs(self.amps - 0.9*r1))
        t90 = self.temps[t90i]
        ax.axvline(t90, linestyle = '--',linewidth=0.5,color='red')

        global temps
        temps.append([t10,t50,t90])
        global rows
        rows.append([label,r'$T_{10}=%.3f$'%t10,r'$T_{50}=%.3f$'%t50,r'$T_{90}=%.3f$'%t90,r'$T_{90}-T_{10}=%.3f$'%(t90-t10)])
        

fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.2)
ax.set_yticks([])
ax.set_xlim(9.8,10.2)
ax.set_title('Ramp (1 V Lock-In Output)')
ax.set_xlabel('Temperature (K)')
ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')
ax.set_ylabel(r'Resistance ($\mu \Omega$)')

rampu = voltemp('rampu.csv')
rampd = voltemp('rampd.csv')
srampu = voltemp('srampu2.csv')
srampd = voltemp('srampd.csv')

rampu.plot(ax, 'fast and up')
rampd.plot(ax, 'fast and down')
srampu.plot(ax, 'slow and up')
srampd.plot(ax, 'slow and down')


#vrampu = voltemp('5Vsrampu.csv')
#vrampd = voltemp('5Vsrampd.csv')
#
#vrampu.plot(ax, '5 V up')
#vrampd.plot(ax, '5 V down')
#temps = np.array(temps)
#avgT10 = np.mean(temps[:,0])
#avgT50 = np.mean(temps[:,1])
#avgT90 = np.mean(temps[:,2])
#ax.axvline(avgT10,color='green')
#ax.axvline(avgT50,color='yellow')
#ax.axvline(avgT90,color='red')
#rows.append([['',r'$\langle T_{10}\rangle=%.3f$'%avgT10,r'$\langle T_{50}\rangle=%.3f$'%avgT50,r'$\langle T_{90}\rangle=%.3f$'%avgT90]])
ax.table(cellText=rows,cellLoc='left')
plt.legend()
fig.savefig('./images/superc.png')
plt.show()

