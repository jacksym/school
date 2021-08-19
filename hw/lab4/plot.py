import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.signal as sig

class temptime:
    def __init__(self, filename):
        df = pd.read_csv(filename)
        self.times = df['Time (s)']
        self.temps = df['Temperature (K)']

    def plot(self, ax, distance):
        peaki = sig.find_peaks(self.temps, distance=distance)[0]
        tc_meas = np.array([])
        enumpeaki = np.array(list(enumerate(peaki)))
        for it, i in enumpeaki[1:-1]:
            ax.axvline(self.times[i], color='lightgray', linestyle='--')
            tc = self.times[enumpeaki[it,1]] - self.times[enumpeaki[it-1,1]]
            tc_meas = np.append(tc_meas, tc)

        print(tc_meas)
        tc = np.mean(tc_meas)

        ax.plot(self.times, self.temps, marker='o', label=r'period: $\tau _c={:.2f}$'.format(tc))
        plt.legend()
        


fig, ax = plt.subplots()
ax.set_xlabel('Time (s)')
ax.set_ylabel('Temperature (K)')
ax.set_title('Temperature-Time')

#tc = temptime('tc.csv')
#tc.plot(ax, 10)
pi = temptime('tc.csv')
pi.plot(ax, 10)
fig.savefig('images/tauc.png')

plt.show()
