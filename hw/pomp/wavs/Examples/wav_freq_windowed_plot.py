#Python code for reading in a wav file and displaying the
#Discrete Fourier Transform Dr Jonathan A Kemp, 2020

#Import wavfile module for working with wav files
from scipy.io import wavfile
#Import the numpy module for working with arrays etc.
import numpy as np
#Import pyplot from the matplotlib module for plotting
import matplotlib.pyplot as plt

#Import wav file into array called data and samplerate into variable called fs
#If the wav has more than one channel then each channel will be a column
#in the resulting array
fs, data = wavfile.read('guitar_fifthalong_chop2.wav')


#Check if the file is mono and if so make it a column vector
if data.ndim == 1:
    data = np.transpose(np.array([data])) #Changes vector to be 2D column vector

#Get the number of channels, Nch, and number of data samples per channel, Ns
Ns, Nch = data.shape

#Make a column vector "Hann window" for fading in and out waveforms:
w = np.transpose( [ 0.5 - 0.5*np.cos(2*np.pi*np.arange(Ns)/Ns) ] )

#Do Discrete Fourier Transform of the data array using numpy's fft commands
#Ns is the number of samples per channel and 0 says along first dimension
#multiple channels in one line of code:
#discrete Fourier transfrom without windowing:
data_fft = np.fft.fft(data,Ns,0)
#data is multiplied by the window, w, before discrete Fourier transfrom 
data_fft_windowed = np.fft.fft(w*data,Ns,0)

#Make a frequency array (0 then fs/Ns then 2*fs/Ns etc.)
f = (fs/Ns)*np.arange(Ns)

#Frequency domain plot:
plt.plot(f,20*np.log10(abs(data_fft)), label='No Window')
plt.plot(f,20*np.log10(abs(data_fft_windowed)), label='Windowed')
plt.xlabel('Frequency (Hz)')
plt.ylabel('dB')
plt.legend()

#Make the plot appear
plt.show()

#After running this code close the plot window if you want to
#be able to use the shell window in IDLE

#If the command below is uncommented it sets zoom to (xmin,xmax,ymin,ymax)
#plt.axis([0, 7000, 0, 90])
