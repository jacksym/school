#Import the numpy module for working with arrays etc.
import numpy as np
#Import pyplot from the matplotlib module for plotting
import matplotlib.pyplot as plt

#Highest harmonic to include:
Nt = 100

#Fundamental frequency (Hz):
f = 110;

#Number of seconds for signal:
Secs = 2/110;

#Sample rate (Hz):
fs = 44100;

#Number of samples:
Ns = round(Secs*fs)

#Array of time values:
t = (1/fs)*np.arange(Ns)
#Make t a column vector using the reshape method:
t = t.reshape(Ns,1)

#Array of harmonic numbers to include
#All harmonics:
#m = np.arange(1,Nt+1,1)
#Odd harmonics:
m = np.arange(1,Nt+1,2)

#Initialise signal to zeros:
y = np.zeros((Ns,1))


for n in m:
    y = y + (4/(n*np.pi))*np.sin(n*2*np.pi*f*t)
    
#Construct a plot of a sound file
#Time domain plot:
plt.plot(t,y)
plt.xlabel('t (seconds)')
plt.ylabel('Amplitude')

#Make the plot appear
plt.show()
