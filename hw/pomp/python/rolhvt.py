from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt

class Wav:
    def __init__(self, filename, f0):
        path = '../wavs/'
        fs, data = wavfile.read(path+filename)

        start = 0; stop = 0
        for i,val in enumerate(data[::-1]):
            if val>5000: stop = i; break
        for i,val in enumerate(data[::1]):
            if val>1000: start = i; break
        #data = data[start:stop]

        Ns = len(data)

        slices=3
        self.time_step = 2
        t = (1/fs)*np.arange(Ns)
        pts_step = 0
        for i,val in enumerate(t):
            if val>self.time_step: pts_step = i-1; break
        
        self.f_peaks_s = []
        self.peaks_s = []
        print(len(data)-slices*pts_step)
        for slice in range(slices):
            start = slice*pts_step
            stop = (slice+1)*pts_step
            datas = data[start:stop]

            if datas.ndim == 1:
                datas = np.transpose(np.array([datas]))
            Ns, Nch = datas.shape

            w = np.transpose( [ 0.5 - 0.5*np.cos(2*np.pi*np.arange(Ns)/Ns) ] )

            self.spec = np.fft.fft(w*datas,Ns,0)
            self.f = (fs/Ns)*np.arange(Ns)

            N_harm = 50
            #Convert frequency in Hz to bin number:
            N0 = round(f0*Ns/fs) + 1
            N_peaks = np.zeros((N_harm,1))
            peaks = np.zeros((N_harm,1))
            for n in range(N_harm):
                N_start = round((n+0.5)*N0)
                N_end = round((n+1.5)*N0)
                N_peaks[n] = N_start + np.argmax(abs(self.spec[N_start:N_end,0]))
                peaks[n] = np.max(abs(self.spec[N_start:N_end,0]))
            self.f_peaks_s.append(N_peaks*fs/Ns)
            self.peaks_s.append(peaks)

        self.f_peaks_s = np.array(self.f_peaks_s)
        self.peaks_s = np.array(self.peaks_s)
        self.f0=f0
        #print(self.f_peaks_s.shape)


    def plot(self, ax, title):
        ax.set_xlim(0,np.max(self.f_peaks_s))
        #ax.set_ylim(50,200)
        #ax.axhline(0,linewidth=0.8, color='black')
        ax.set_title(title)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('dB')

        for slice in range(len(self.f_peaks_s)):
            peaks_dB = 20*np.log10(abs(self.peaks_s[slice]))
            ax.plot(self.f_peaks_s[slice],peaks_dB,label=str(slice*self.time_step)+' s')
        ax.legend()
        #ax.plot(self.f, 20*np.log10(abs(self.spec)), linewidth=0.3)
        harm = self.f0; n = 0; ticks=[]
        while harm < np.max(self.f_peaks_s):
            ticks.append(harm)
            #ax.axvline(harm,linewidth=0.3, color='gray',linestyle='--')
            n+=8
            harm = n*self.f0
        ax.set_xticks(ticks)


fig, ax = plt.subplots(2,2)
fig.set_size_inches(10,8,forward=True)
plt.subplots_adjust(left=0.2, hspace=0.6,wspace=0.5)

Ef = 82.4
E2 = Wav('E2rol.wav', 1*Ef)
E3 = Wav('E3rol.wav', 2*Ef)
E4 = Wav('E4rol.wav', 3*Ef)
E5 = Wav('E5rol.wav', 4*Ef)

E2.plot(ax[0,0], 'Open E2 String')
E3.plot(ax[0,1], 'E2 String 12th Fret')
E4.plot(ax[1,0], 'Open E4 String')
E5.plot(ax[1,1], 'E4 String 12th Fret')


fig.savefig('../images/rolhvt.png',quality=100)
plt.show()
