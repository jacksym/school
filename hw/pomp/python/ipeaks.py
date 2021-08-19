from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt

class Wav:
    def __init__(self, filename, f0):
        path = '../wavs/'
        fs, data = wavfile.read(path+filename)

        start = 0
        stop = 0
        for i,val in enumerate(data[::-1]):
            if val>5000:
                stop = i
        for i,val in enumerate(data):
            if val>3000:
                start = i
        data = data[start:stop]

        if data.ndim == 1:
            data = np.transpose(np.array([data]))

        Ns, Nch = data.shape

        w = np.transpose( [ 0.5 - 0.5*np.cos(2*np.pi*np.arange(Ns)/Ns) ] )
        print(data.shape, w.shape, Ns)

        data_fft_windowed = np.fft.fft(w*data,Ns,0)
        self.spec = data_fft_windowed

        self.f = (fs/Ns)*np.arange(Ns)

        N_harm = 20
        #Convert frequency in Hz to bin number:
        N0 = round(f0*Ns/fs) + 1
        self.N_peaks = np.zeros((N_harm,1))
        #self.f_peaks = np.zeros((N_harm,1))
        self.peaks = np.zeros((N_harm,1))
        for n in range(N_harm):
            N_start = round((n+0.5)*N0)
            N_end = round((n+1.5)*N0)
            self.N_peaks[n] = N_start + np.argmax(abs(self.spec[N_start:N_end,0]))
            self.peaks[n] = np.max(abs(self.spec[N_start:N_end,0]))

        self.f_peaks = self.N_peaks*fs/Ns
        self.f0 = f0

    def plot(self, ax, title):
        ax.set_xlim(0,np.max(self.f_peaks))
        ax.axhline(0,linewidth=0.8,color='black')
        ax.set_title(title)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('dB')
        peaks_dB = 20*np.log10(abs(self.peaks))
        for i in range(len(self.f_peaks)):
            ax.plot([self.f_peaks[i],self.f_peaks[i]], [0,peaks_dB[i]], color='grey', linewidth=0.8, linestyle='--')
        ax.scatter(self.f_peaks,peaks_dB)
        #ax.plot(self.f, 20*np.log10(abs(self.spec)), linewidth=0.3)

        harm = self.f0; n = 0; ticks=[]
        while harm < np.max(self.f_peaks):
            ticks.append(harm)
            #ax.axvline(harm,linewidth=0.3, color='gray',linestyle='--')
            n+=8
            harm = n*self.f0
        ax.set_xticks(ticks)

fig, ax = plt.subplots(2,2)
fig.set_size_inches(10,8,forward=True)
plt.subplots_adjust(left=0.2, hspace=0.6,wspace=0.5)

Ef = 82.4
E2 = Wav('E2raw.wav', 1*Ef)
E3 = Wav('E3raw.wav', 2*Ef)
E4 = Wav('E4raw.wav', 3*Ef)
E5 = Wav('E5raw.wav', 4*Ef)

E2.plot(ax[0,0], 'Open E2 String')
E3.plot(ax[0,1], 'E2 String 12th Fret')
E4.plot(ax[1,0], 'Open E4 String')
E5.plot(ax[1,1], 'E4 String 12th Fret')


fig.savefig('../images/ipeaks.png',quality=100)
plt.show()
