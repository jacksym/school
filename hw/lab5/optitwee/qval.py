import numpy as np
import matplotlib.pyplot as plt

c = 3*10**8
ref_fluid = 1.332
visc_water = 0.001

def power(current):
    power = 0.8154*current - 41.954
    return 0.5*power

class VTrap:
    def __init__(self, filename, radius):
        self.radius = radius
        with open(filename) as meas:
            lines = meas.readlines()
            currents = []
            vels = []
            devs = []
            for line in lines[1:]:
                line = line.split()
                for i in range(len(line)): line[i] = float(line[i])
                currents.append(line[0])
                vals = np.array(line[1:])
                avg = np.mean(vals)
                dev = np.sqrt((1/(vals.size-1))*np.sum((vals-avg)**2))/np.sqrt(vals.size)
                vels.append(avg)
                devs.append(dev)
            self.currents = np.array(currents)
            self.vels = np.array(vels)
            self.power = power(self.currents)
            self.devs = np.array(devs)

    def plot(self, ax, color):
        ax.set_xlabel('Power Output of Laser (mW)')
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')
        ax.set_ylabel(r'Average Velocity of Trapped Particle ($\mu$m/s)')
        (m, b), res, _,_,_ = np.polyfit(self.power, self.vels, 1, full=True)
        print(m, b, res)
        x = np.linspace(np.min(self.power), np.max(self.power), 4)
        ax.plot(x, m*x+b, linestyle='--', color=color, linewidth=0.8)

        Q = 6*np.pi*visc_water*self.radius/(ref_fluid*m)
        radius = r'$r = %s$ $\mu$m'%self.radius
        trend = r'$P_L = %.3f\cdot v_c %+.3f$'%(m,b)
        q_value = r'$Q = %.3f$'%Q
        cellText.append([radius, trend, q_value])
        ax.scatter(self.power, self.vels, label=radius, color=color)
        ax.errorbar(self.power, self.vels, yerr=self.devs, fmt='none', color=color)

fig, ax = plt.subplots()
ax.grid(True)
cellText=[]

size1 = VTrap('1mm.tsv', 1)
size3 = VTrap('3mm.tsv', 3)

size1.plot(ax, 'grey')
size3.plot(ax, 'black')

ax.legend(loc='upper left')
ax.table(cellText=cellText, cellLoc='left')
#fig.savefig('../images/qval.png', quality=100)
#plt.show()
