#========================================================================
#                                  IMPORTS
#------------------------------------------------------------------------
import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt
import sys
pi = np.pi
#========================================================================
#                            FUNCOES AUXILIARES
#------------------------------------------------------------------------
def psd(signal, sr=1, N=2048):
    '''
    numpy.fft.fft:
    When the input a is a time-domain signal and A = fft(a): 
    . np.abs(A) is its amplitude spectrum; 
    . np.abs(A)**2 is its power spectrum; 
    . np.angle(A) is the phase spectrum.
    '''
    f = signal
    ft = fft.fft(f, N)
    ft_shifted = fft.fftshift(ft)
    aft = np.abs(ft)**2
    aft_shifted = abs(ft_shifted)**2 # spectrum shifted
    # shifting frequencies
    freq = np.fft.fftfreq(N, d=sr) # sr = sampling rate
    freqs = np.concatenate([freq[int(len(freq)/2):],[0]])
    freqs = np.concatenate([freqs, freq[1:int(len(freq)/2)]])
    #
    return freqs, aft_shifted
#========================================================================
#                            FUNCOES EXERCICIO 1
#------------------------------------------------------------------------
def gaussian_chirp(t, a=1., b=2*pi, c=6*pi):
    return a * np.exp( - (b- 1.j * c) * (t**2) )
#------------------------------------------------------------------------
def linear_chirp(t, a=1., b=1/4.):
    return a * np.exp( 1.j * b * (t**2) )
#------------------------------------------------------------------------
def square_chirp(t, a=1., b=1/4.):
    return a * np.exp( 1.j * b * (t**3) )
#------------------------------------------------------------------------
def hyper_chirp(t, a=1., b=6*pi, c=2.3):
    return a * np.cos( b / (c - t) ) 
#========================================================================
#                            FUNCOES EXERCICIO 1
#------------------------------------------------------------------------
N = 2048 # Number of bins (samples)
#------------------------------------
ti_1 = 0#-2 # initial time
tf_1 = 6#2  # final time
t_1 = np.linspace(ti_1, tf_1, N)  # time bins
res_1 = abs(ti_1 - tf_1)/len(t_1) # resolution (sampling rate)

g_1 = gaussian_chirp(t_1) # chirp
x_1, ft_1 = psd(g_1, res_1, N) # frequncy bins + psd of signal
#------------------------------------
ti_2 = 0#-6
tf_2 = 6
t_2 = np.linspace(ti_2, tf_2, N)
res_2 = abs(ti_2 - tf_2)/len(t_2)

g_2 = linear_chirp(t_2)
x_2, ft_2 = psd(g_2, res_2, N)
#------------------------------------
ti_3 = 0#-4
tf_3 = 6#4
t_3 = np.linspace(ti_3, tf_3, N)
res_3 = abs(ti_3 - tf_3)/len(t_3)

g_3 = square_chirp(t_3)
x_3, ft_3 = psd(g_3, .01)
#------------------------------------
ti_4 = 0
tf_4 = 6#2
t_4 = np.linspace(ti_4, tf_4, N)
res_4 = abs(ti_4 - tf_4)/len(t_4)

g_4 = hyper_chirp(t_4)
x_4, ft_4 = psd(g_4, res_4, N)
#========================================================================
#                                PLOTS
#------------------------------------------------------------------------
fig, ax = plt.subplots(4, 2, figsize=(9,7)) # (4 x 2) subplots
#---------------------------------- Row 1
# Column 1
ax[0,0].set_title('      Chirp Signals', size=15, loc='left')
ax[0,0].plot(t_1, g_1.real,'r-', label='real')
ax[0,0].plot(t_1, g_1.imag,'b--', label='imag')
ax[0,0].legend(bbox_to_anchor=(.6,1.55), loc="upper left")
ax[0,0].set_ylabel('Gaussian \nchirp', rotation=0, labelpad=25., size=12)
# Column 2
ax[0,1].set_title('Power Spectrum', size=15)
ax[0,1].plot(x_1, ft_1,'k-')
#---------------------------------- Row 2
# Column 1
ax[1,0].plot(t_2, g_2.real,'r-')
ax[1,0].plot(t_2, g_2.imag,'b--')
ax[1,0].set_ylabel('Linear \nchirp', rotation=0, labelpad=25., size=12)
# Column 2
ax[1,1].plot(x_2, ft_2,'k-')
#---------------------------------- Row 3
# Column 1
ax[2,0].plot(t_3, g_3.real,'r-')
ax[2,0].plot(t_3, g_3.imag,'b--')
ax[2,0].set_ylabel('Square \nchirp', rotation=0, labelpad=25., size=12)
# Column 2
ax[2,1].plot(x_3, ft_3,'k-')
#---------------------------------- Row 4
# Column 1
ax[3,0].plot(t_4, g_4.real,'r-')
ax[3,0].plot(t_4, g_4.imag,'b--')
ax[3,0].set_ylabel('Hyperbolic \nchirp', rotation=0, labelpad=25., size=12)
ax[3,0].set_xlabel('t', size=12)
# Column 2
ax[3,1].plot(x_4, ft_4,'k-')
ax[3,1].set_xlabel('f', size=12)
#---------------------------------- End of individual-plot settings 
for i in range(4):
    # Setting same x_lim for all axes
    ax[i,1].set_xlim(-4,4)
# Setting spacing between subplots
fig.subplots_adjust(left=None, bottom=None, \
                    right=None, top=None, \
                    wspace=.2, hspace=.3)
#---------------------------------- Saving and showing plot
fig.savefig('TEST_chirps_psd.jpg', dpi=400, bbox_inches='tight')
plt.show()
#------------------------------------------------------------------------ FIM
plt.close('all')
