import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

class SpectrumData:
    def __init__(self, filename):
        with open(filename, 'r', errors='ignore') as df:
            rawlines = df.readlines()
            self.photons = np.array([])
            for line in rawlines[23:23+255]:
                self.photons = np.append(self.photons, float(line.split()[2]))
            chan1 = float(rawlines[15].split()[4])
            chanen1 = float(rawlines[15].split()[5])
            chan2 = float(rawlines[15].split()[6])
            chanen2 = float(rawlines[15].split()[7])
            es = np.abs((chanen2-chanen1)/(chan2-chan1))
            rawmarks = rawlines[299::2]
            self.marks = []
            for term in rawmarks:
                enel = float(term.split()[6])
                countel = float(term.split()[7])
                el = term.split()[8][-3:-1]
                self.marks.append([el, enel, countel])
            self.energies = np.array([])
            for i in range(0,len(self.photons)):
                self.energies = np.append(self.energies, i*es)
        self.fig, self.ax = plt.subplots()

    def plot(self, height, threshold, extraPeaks=[]):
        self.ax.plot(self.energies, self.photons, color='black')
        self.ax.set_ylim(0,1.01*np.max(self.photons))
        self.ax.set_xlim(0, np.max(self.energies))
        self.ax.set_xlabel(r'energies')
        plt.subplots_adjust(bottom=0.2)
        scipeaki = sig.find_peaks(self.photons, height=height, threshold=threshold)[0]

        peaks = np.array([])
        for i in scipeaki:
            peaks = np.append(peaks, self.energies[i])
        peaks = np.append(peaks, extraPeaks)
        for i in peaks:
            self.ax.axvline(i, linestyle='--', linewidth=0.5, c='red')

        peakval = np.array([])
        for i in scipeaki:
            peakval = np.append(peakval, self.photons[i])
        for peak in extraPeaks:
            ind = np.searchsorted(self.energies, peak)
            peakval = np.append(peakval, self.photons[ind])

        for mark in self.marks:
            self.ax.vlines(mark[1], 0, ymax=mark[2], linewidth=2, colors='green')
            self.ax.text(mark[1], mark[2]+5, mark[0])

        for i in peakval:
            self.ax.axhline(i, linestyle='--', linewidth=0.5, c='red')

        self.ax.set_xticks(peaks)
        self.ax.set_yticks(peakval)
        plt.setp(self.ax.get_xticklabels(), rotation=45, rotation_mode='anchor', ha='right')


spectrum1 = SpectrumData('cassydata/flourescenceSpectrum1.lab')
spectrum1.ax.set_title('FeZn Flourescence Spectrum')
spectrum1.plot(150, 4, [9.45])
spectrum1.fig.savefig('./images/fFeZn')

spectrum2 = SpectrumData('cassydata/bronzeSpectrum.lab')
spectrum2.ax.set_title('(Unknown Alloy 1) Bronze Flourescence Spectrum')
spectrum2.plot(600, 4, [9.396])
spectrum2.fig.savefig('./images/fbronze')

spectrum3 = SpectrumData('cassydata/stainlessSteelSpectrum.lab')
spectrum3.ax.set_title('(Unknown Alloy 2) Stainless Steel Flourescence Spectrum')
spectrum3.plot(50,4, [7.40444, 8.02667])
spectrum3.fig.savefig('./images/fssteel')

spectrum4 = SpectrumData('cassydata/flourescenceSpectrum2.lab')
spectrum4.ax.set_title('(Unknown Alloy 3) "Aluminium" Flourescence Spectrum')
spectrum4.plot(40,4, [8.77333, 12.44444, 14.75])
spectrum4.fig.savefig('./images/fCuPb')

coin = SpectrumData('cassydata/20pCoinSpectrum.lab')
coin.ax.set_title('20p Coin Flourescence Spectrum')
coin.plot(100, 5, [7.344])
coin.fig.savefig('./images/f20p')
plt.show()
