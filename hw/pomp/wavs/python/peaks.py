from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt

class Wav:
    def __init__(self, filename, f0):
        path='../wavs/'
        fs, data = wavfile.read(path+filename)

        start = 0
        stop = 0
        for i,val in enumerate(data[::-1]):
            if val>5000:
                stop = i
        for i,val in enumerate(data):
            if val>5000:
                start = i
        data = data[start:stop]

        if data.ndim == 1:
            data = np.transpose(np.array([data]))


        Ns, Nch = data.shape

        w = np.transpose( [ 0.5 - 0.5*np.cos(2*np.pi*np.arange(Ns)/Ns) ] )

        data_fft_windowed = np.fft.fft(w*data,Ns,0)
        self.spec = data_fft_windowed

        #Make a frequency array (0 then fs/Ns then 2*fs/Ns etc.)
        self.f = (fs/Ns)*np.arange(Ns)

        #Number of harmonics to search for:
        N_harm = 20
        #Convert frequency in Hz to bin number:
        N0 = round(f0*Ns/fs) + 1
        N_peaks = np.zeros((N_harm,1))
        f_peaks = np.zeros((N_harm,1))
        for n in range(N_harm):
            N_start = round((n+0.5)*N0)
            N_end = round((n+1.5)*N0)
            N_peaks[n] = N_start + np.argmax(abs(self.spec[N_start:N_end,0]))
            
        self.f_peaks = N_peaks*fs/Ns

    def plot(self, ax, title):
        ax.set_xlim(0,7000)
        ax.axhline(0,linestyle='--',color='black')
        ax.set_title(title)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('dB')
        for peak in self.f_peaks:
            ax.axvline(peak,linestyle='--',color='grey',linewidth=0.5)
        ax.plot(self.f,20*np.log10(abs(self.spec)))


fig, ax = plt.subplots(2,2)
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


fig.savefig('../images/fft_peaks.png',quality=100)
plt.show()
