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
    ft_shifted = fft.fftshift(ft)/N
    aft = np.abs(ft)**2
    aft_shifted = abs(ft_shifted)**2
    #
    freq = np.fft.fftfreq(N, d=sr)
    freqs = np.concatenate([freq[int(len(freq)/2):],[0]])
    freqs = np.concatenate([freqs, freq[1:int(len(freq)/2)]])
    #
    return freqs, ft_shifted
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
    #
    return a * np.cos( b / (c - t) ) 
#========================================================================
#                            FUNCOES EXERCICIO 1
#------------------------------------------------------------------------
N = 2048
#------------------------------------------------------------------------
ti_1 = -2
tf_1 = 2
t_1 = np.linspace(ti_1, tf_1, N)
res_1 = abs(ti_1 - tf_1)/len(t_1)

g_1 = gaussian_chirp(t_1)
x_1, ft_1 = psd(g_1, res_1, N)
#------------------------------------
ti_2 = -6
tf_2 = 6
t_2 = np.linspace(ti_2, tf_2, N)
res_2 = abs(ti_2 - tf_2)/len(t_2)

g_2 = linear_chirp(t_2)
x_2, ft_2 = psd(g_2, res_2, N)
#------------------------------------
ti_3 = -4
tf_3 = 4
t_3 = np.linspace(ti_3, tf_3, N)
res_3 = abs(ti_3 - tf_3)/len(t_3)

g_3 = square_chirp(t_3)
x_3, ft_3 = psd(g_3, .01)#res_3, N)
#------------------------------------
ti_4 = 0
tf_4 = 2
t_4 = np.linspace(ti_4, tf_4, N)
res_4 = abs(ti_4 - tf_4)/len(t_4)

g_4 = hyper_chirp(t_4)
x_4, ft_4 = psd(g_4, res_4, N)
#========================================================================
#                                PLOTS
#------------------------------------------------------------------------
fig, ax = plt.subplots(4, 2, figsize=(9,7))
#------------------------------------------------------------------------ Plot 1
ax[0,0].set_title('Chirp Signals', size=15)
ax[0,0].plot(t_1, g_1.real,'r-', label='real')
ax[0,0].plot(t_1, g_1.imag,'b--', label='imag')
ax[0,0].legend(bbox_to_anchor=(.95,1.6), loc="upper left")
ax[0,0].set_ylabel('Gaussian \nchirp', rotation=0, labelpad=25., size=12)
#
ax[0,1].set_title('Fourier Transform', size=15)
ax[0,1].plot(x_1, ft_1.real,'r-')
ax[0,1].plot(x_1, ft_1.imag,'b--')
#------------------------------------------------------------------------ Plot 2
ax[1,0].plot(t_2, g_2.real,'r-')
ax[1,0].plot(t_2, g_2.imag,'b--')
ax[1,0].set_ylabel('Linear \nchirp', rotation=0, labelpad=25., size=12)
#
ax[1,1].plot(x_2, ft_2.real,'r-')
ax[1,1].plot(x_2, ft_2.imag,'b--')
#------------------------------------------------------------------------ Plot 3
ax[2,0].plot(t_3, g_3.real,'r-')
ax[2,0].plot(t_3, g_3.imag,'b--')
ax[2,0].set_ylabel('Square \nchirp', rotation=0, labelpad=25., size=12)
#
ax[2,1].plot(x_3, ft_3.real,'r-')
ax[2,1].plot(x_3, ft_3.imag,'b--')
#------------------------------------------------------------------------ Plot 4
ax[3,0].plot(t_4, g_4.real,'r-')
ax[3,0].plot(t_4, g_4.imag,'b--')
ax[3,0].set_ylabel('Hyperbolic \nchirp', rotation=0, labelpad=25., size=12)
ax[3,0].set_xlabel('t', size=12)
#
ax[3,1].plot(x_4, ft_4.real,'r-')
ax[3,1].plot(x_4, ft_4.imag,'b--')
ax[3,1].set_xlabel('f', size=12)
#------------------------------------------------------------------------
for i in range(4):
    ax[i,1].set_xlim(-4,4)
    
#ax[1,1].set_xlim(-2,2)
#------------------------------------------------------------------------
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=.2, hspace=.3)
fig.savefig('chirps_FT.jpg', dpi=400, bbox_inches='tight')
plt.show()
#------------------------------------------------------------------------ FIM
plt.close('all')
