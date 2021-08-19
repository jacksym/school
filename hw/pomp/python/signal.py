from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt

class Wav:
    def __init__(self, filename):
        path = '../wavs/'
        fs, data = wavfile.read(path+filename)

        if data.ndim == 1:
            data = np.transpose(np.array([data]))

        Ns, Nch = data.shape

        self.t = (1/fs)*np.arange(Ns)
        self.data = data

    def plot(self, ax, title):
        mint, maxt = 0,0
        for i,val in enumerate(self.data):
            if val>5000:
                mint = i
                break
        self.t -= self.t[mint]
        ax.plot(self.t[mint:],self.data[mint:])
        ax.set_xlim(-0.02,0.2)
        ax.set_ylim(-25000,25000)
        ax.grid(True)
        ax.axhline(0,linestyle='--',color='black')
        ax.set_title(title)
        ax.set_xlabel('t (seconds)')
        ax.set_ylabel('Amplitude')

fig, ax = plt.subplots(2,2)
plt.subplots_adjust(left=0.2, hspace=0.6,wspace=0.5)

E2 = Wav('E2raw.wav')
E3 = Wav('E3raw.wav')
E4 = Wav('E4raw.wav')
E5 = Wav('E5raw.wav')

E2.plot(ax[0,0], 'Open E2 String')
E3.plot(ax[0,1], 'E2 String 12th Fret')
E4.plot(ax[1,0], 'Open E4 String')
E5.plot(ax[1,1], 'E4 String 12th Fret')


fig.savefig('../images/raw_signal.png',quality=100)
plt.show()
