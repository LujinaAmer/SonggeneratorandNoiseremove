import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
from scipy.signal import find_peaks
from scipy.fftpack import fft


t = np.linspace(0 , 3 , 12*1024)

f1=[0,0,0,196,220,220,196,0,0,0,0,0]
f2=[261.63,261.63,261.63,0,0,0,0,329.63,329.63,293.66,293.66,261.63]
period=[0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.175,0.175,0.175,0.175,0.35]


def x(f1,f2,period,ti,i,y):
    
    if i==12:
        return y  
        
    y += np.reshape(np.sin(2*np.pi*f1[i]*t)*[(t>=ti)&(t<=(period[i]+ti)
    )],np.shape(t))+np.reshape(np.sin(2*np.pi*f2[i]*t)*[(t>=ti
    )&(t<=(ti+period[i]))],np.shape(t))         
    i += 1
    ti += 0.05+period[i-1]
    return x(f1,f2,period,ti,i,y)
   
   
sig = x(f1,f2,period,0,0,0)
sd.play(sig,3*1024)

n = np.random.randint(0,512,2)
N =3*1024
tf = np.linspace(0 , 512 , int(N/2))

plt.subplot(3,2,1)
plt.plot(t,sig)

x_f = fft(sig)
x_f = 2/N * np.abs(x_f [0:np.int(N/2)]) 

plt.subplot(3,2,2)
plt.plot(tf,x_f)

waveform1 = np.sin(2*n[0]*np.pi*t)
waveform2 = np.sin(2*n[1]*np.pi*t)
time_data = waveform1 + waveform2
xnt = sig + time_data

plt.subplot(3,2,3)
plt.plot(t,xnt)

xn_f = fft(xnt)
xn_f = 2/N * np.abs(xn_f [0:np.int(N/2)]) 

plt.subplot(3,2,4)
plt.plot(tf,xn_f)

x_peak = np.amax(x_f)


length = len(xn_f)

peaks=[]

i=0
while i<length :
    if xn_f[i]>x_peak:
        peaks.append(i)
     
   
fn3 = np.round(tf[peaks[0]])
fn4 = np.round(tf[peaks[1]])


x_filtered = xnt - (np.sin(2*fn3 *np.pi* t) + np.sin(2*fn4 *np.pi* t)) * (t >= 0) * (t <= 3)

plt.subplot(3,2,5)
plt.plot(t,x_filtered)

xf_frequency = fft(x_filtered)
xf_frequency= 2/N * np.abs(xf_frequency [0:np.int(N/2)])

plt.subplot(3,2,6)
plt.plot(tf,xf_frequency)


sd.play(x_filtered,3*1024)









