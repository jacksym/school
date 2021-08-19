#Python code for reading in a wav file and displaying the
#Discrete Fourier Transform Dr Jonathan A Kemp, 2020

#Import wavfile module for working with wav files
from scipy.io import wavfile
#Import the numpy module for working with arrays etc.
import numpy as np
#Import pyplot from the matplotlib module for plotting
import matplotlib.pyplot as plt

def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[result.size//2:]

#Import wav file into array called data and samplerate into variable called fs
#If the wav has more than one channel then each channel will be a column
#in the resulting array
fs, data = wavfile.read('guitar_fifthalong.wav')

#Number of samples in analysis window
Nw = 2**12

#Number of samples between successive analysis windows
No = round(Nw/2)

#Make a column vector "Hann window" for analysis window:
w = [ 0.5 - 0.5*np.cos(2*np.pi*np.arange(Nw)/Nw) ]

#Check if the file is mono and if so make it a column vector
if data.ndim == 1:
    data = np.transpose(np.array([data])) #Changes vector to be 2D column vector

#Get the number of channels, Nch, and number of data samples per channel, Ns
Ns, Nch = data.shape

#Number of analysis windows in the signal using "floor division"
Na = ((Ns-Nw)//No) + 1

#Intialise analysis array
data_analysis = np.zeros((Nw,Na,Nch))
#Initialise correlation information
cor_peak = np.zeros((Na,Nch))

#Fill analysis array
for n in range(Na):
    for m in range(Nch):
        #Put data into nth analysis window of mth channel and
        #pointwise multiply by window
        data_analysis[:,n,m] = data[n*No:n*No+Nw,m]*w
        #Do autocorrelation of analysis windows
        data_cor = autocorr(data_analysis[:,n,m])
        #Find index of first negative value in auto-correlation:
        ind = np.argmax(data_cor < 0)
        #Find maximum in autocorrelation after first negative value:
        cor_peak[n,m] = ind + np.argmax(data_cor[ind:Nw])

#Convert from number of samples for a period to frequency
#Gives divide by zero warning (which may be safely ignored) if there are
#any sections with no pitch detected:
f_track = (fs/cor_peak)

#time array for start of windows:
t = (No/fs)*np.arange(Na)

#Frequency domain plot:
plt.plot(t,f_track)
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (s)')

#Alternative if you want cents relative to a given frequency:
#plt.plot(t,1200*np.log2(f_track/110))

#Make the plot appear
plt.show()

#After running this code close the plot window if you want to
#be able to use the shell window
