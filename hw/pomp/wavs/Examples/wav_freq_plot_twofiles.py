#Python code for reading in a wav file and displaying the
#Discrete Fourier Transform Dr Jonathan A Kemp, 2020

#Import wavfile module for working with wav files
from scipy.io import wavfile
#Import the numpy module for working with arrays etc.
import numpy as np
#Import pyplot from the matplotlib module for plotting
import matplotlib.pyplot as plt

#Read in first file and prepare plot of it:

#Import wav file into array called data and samplerate into variable called fs
#If the wav has more than one channel then each channel will be a column
#in the resulting array
fs, data = wavfile.read('guitar_fifthalong.wav')

#Check if the file is mono and if so make it a column vector
if data.ndim == 1:
    data = np.transpose(np.array([data])) #Changes vector to be 2D column vector

#Get the number of channels, Nch, and number of data samples per channel, Ns
Ns, Nch = data.shape

#Do Discrete Fourier Transform of the data array using numpy's fft commands
#Ns is the number of samples per channel and 0 says along first dimension
#multiple channels in one line of code:
data_fft = np.fft.fft(data,Ns,0)

#Make a frequency array (0 then fs/Ns then 2*fs/Ns etc.)
f = (fs/Ns)*np.arange(Ns)

#Frequency domain plot:
plt.plot(f,20*np.log10(abs(data_fft/Ns)), label='Entire note (self-windowing)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('dB')

#Repeat for the Second file:

fs, data = wavfile.read('guitar_fifthalong_chop.wav')
if data.ndim == 1:
    data = np.transpose(np.array([data]))
Ns, Nch = data.shape
data_fft = np.fft.fft(data,Ns,0)
f = (fs/Ns)*np.arange(Ns)
plt.plot(f,20*np.log10(abs(data_fft/Ns)), label='Truncated (no windowing)')

#Make legend from labels of plots
plt.legend(loc='upper right')
#If the command below is uncommented it sets zoom to (xmin,xmax,ymin,ymax)
plt.axis([0, 5000, -40, 70])
#Make the plot appear
plt.show()

#After running this code close the plot window if you want to
#be able to use the shell window
