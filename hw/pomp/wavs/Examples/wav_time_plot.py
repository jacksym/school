#Python code for reading in a wav file and displaying the
#waveform - Dr Jonathan A Kemp, 2020

#Import wavfile module for working with wav files
from scipy.io import wavfile
#Import the numpy module for working with arrays etc.
import numpy as np
#Import pyplot from the matplotlib module for plotting
import matplotlib.pyplot as plt


#Import wav file into array called data and samplerate into variable called fs
#If the wav has more than one channel then each channel will be a column
#in the resulting array
fs, data = wavfile.read('bass_16bit.wav')
#fs, data = wavfile.read('flute_16bit.wav')

#Check if the file is mono and if so make it a column vector
if data.ndim == 1:
    data = np.transpose(np.array([data])) #Changes vector to be 2D column vector

#Get the number of channels, Nch, and number of data samples per channel, Ns
Ns, Nch = data.shape

#Make a time array (0 then 1/fs then 2/fs etc.) because sample period is 1/fs
t = (1/fs)*np.arange(Ns)

#Construct a plot of a sound file
#Time domain plot:
plt.plot(t,data)
#plt.plot(t, 20*np.log10(abs(data)))
plt.xlabel('t (seconds)')
plt.ylabel('Amplitude')
#plt.ylabel('dB')

#Make the plot appear
plt.show()

#After running this code close the plot window if you want to
#be able to use the shell window in IDLE

