#========================================================================
#                                  IMPORTS
#------------------------------------------------------------------------
import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt
import sys
pi = np.pi
#========================================================================
#                                  INPUTS
#------------------------------------------------------------------------
# Sinais analisados
#
periodo_1 = 20
n_pontos = 51
n_total = n_pontos
n = np.arange(n_total)
f_1 = np.cos(2*pi*n/periodo_1)
#
periodo_2 = 20
n_pontos = 51
n_total = n_pontos
n = np.arange(n_total)
f_2 = np.cos(2*pi*n/periodo_2)
#========================================================================
#                            FUNCOES AUXILIARES
#------------------------------------------------------------------------
def psd(signal, sr=1):
    '''
    numpy.fft.fft:
    When the input a is a time-domain signal and A = fft(a): 
    . np.abs(A) is its amplitude spectrum; 
    . np.abs(A)**2 is its power spectrum; 
    . np.angle(A) is the phase spectrum.
    '''
    f = signal
    N = 2048
    ft = fft.fft(f, N)/N
    ft_shifted = fft.fftshift(ft)
    aft = np.abs(ft)**2
    aft_shifted = abs(ft_shifted)**2
    #
    freq = np.fft.fftfreq(N, d=sr)
    freqs = np.concatenate([freq[int(len(freq)/2):],[0]])
    freqs = np.concatenate([freqs, freq[1:int(len(freq)/2)]])
    #
    return freqs, ft_shifted.real
#
def rect(signal, width=10):
    center = int(n_pontos/2)
    x=np.arange(len(signal))
    signal_windowed = np.where(abs(center-x)<=width, signal, 0)
    return signal_windowed
#========================================================================
#                                 PLOTS
#------------------------------------------------------------------------
#------------------------------------------------------------------------ Plot 1
fig1, ax = plt.subplots(4, 2, figsize=(9,7))
#------------------------------------ 1 linha
n_analitico=1000
nn = np.arange(n_analitico)
f= np.cos(2*pi*nn/periodo_1)
xf=np.arange(-n_analitico/2,+n_analitico/2)
s1 = f_1
xs1 = np.arange(len(s1))
#-------------- sinal
ax[0,0].plot(xf, f, 'k')
ax[0,0].set_ylabel('Infinite\n signal x(t)', rotation=0, labelpad=25.)
ax[0,0].set_title('Observed Signal - ' + r'$x_{obs}$')
#-------------- psd
ax[0,1].vlines(1/10, 0, 1.5)
ax[0,1].vlines(-1/10, 0, 1.5)
ax[0,1].hlines(0, -.5, .5)
ax[0,1].set_title('Fourier Transform - '+r'FT$[x_{obs}]$')
#------------------------------------ 2 linha
width_2 = 50.
s2 = rect(f_1, width_2/2.)
xs2 = np.arange(len(s2))
#-------------- sinal
ax[1,0].plot(xs2, s2, 'k')
ax[1,0].set_ylabel(r'$W(t)\cdot x(t)$'+'\n'+ r'T$_{obs} = 50$', rotation=0, labelpad=25.)
#-------------- psd
x2, y2 = psd(s2)
ax[1,1].plot(x2, y2, 'k')
#ax[1,1].hlines(.5, .1/periodo_1-width_2/2., 1/periodo_1+width_2/2.)
#------------------------------------ 3 linha
s3 = rect(f_1, 10)
xs3 = np.arange(len(s3))
#-------------- sinal
ax[2,0].plot(xs3, s3, 'k')
ax[2,0].set_ylabel(r'$W(t)\cdot x(t)$'+'\n'+ r'$T_{obs} = 20$', rotation=0, labelpad=25.)
#-------------- psd
x3, y3 = psd(s3[np.where(s3!=0)])
ax[2,1].plot(x3, -y3, 'k')
#------------------------------------ 4 linha
s4 = rect(f_1, 5)
xs4 = np.arange(len(s4))
#-------------- sinal
ax[3,0].plot(xs4, s4, 'k')
ax[3,0].set_ylabel(r'$W(t)\cdot x(t)$'+'\n'+ r'$T_{obs} = 10$', rotation=0, labelpad=25.)
ax[3,0].set_xlabel('t', size=14)
#-------------- psd
x4, y4 = psd(s4[np.where(s4!=0)])
ax[3,1].plot(x4, y4, 'k')
ax[3,1].set_xlabel('f', size=14)
#------------------------------------------------------------------------
for i in range(4):
    ax[i,0].set_ylim(-1.2, 1.2)
    ax[i,1].set_yticks([])
    ax[i,0].set_xlim(-3,n_pontos+3)
    ax[i,1].set_xlim(-.5,.5)
    ax[i,1].set_xticks([-.4, -.3, -.2, -.1, 0., .1, .2, .3, .4])
    ax[i,1].set_xticklabels(['-0.4', '', '-0.2', '', '0.0', '', '0.2', '', '0.4'])
#------------------------------------------------------------------------
#fig1.suptitle('Effects of Windowing')
fig1.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=.2, hspace=0.3)
fig1.savefig('window.jpg', dpi=400, bbox_inches='tight')
plt.close()
#plt.show()
#------------------------------------------------------------------------ Plot 2
fig2, ax = plt.subplots(4, 2, figsize=(9,7))
#------------------------------------ 1 linha
sr_0 = 1
#
sr = 1
s1 = f_2[0::sr]
xs1 = np.arange(0,len(f_2),sr)
#-------------- sinal
ax[0,0].plot(xs1, s1, color='darkgray',  linestyle='solid', linewidth=.85)#(0, (3, 5, 1, 5)))
ax[0,0].plot(xs1, s1, 'k.')
ax[0,0].set_ylabel(r'$t_{sampl} = 1$', rotation=0, labelpad=25.)
ax[0,0].set_title('Signal')
#-------------- psd
x1, y1 = psd(s1)
ax[0,1].plot(x1, y1, 'k')
ax[0,1].vlines(1/periodo_2, 1.1*min(y1), 1.1*max(y1), label=r'$f_{0} = $'+str(round(1/periodo_2,2)))
ax[0,1].legend(loc='upper left')
ax[0,1].set_title('Fourier Transform')
#------------------------------------ 2 linha
sr = 2
#-------------- psd
x2, y2 = psd(s1, sr)
ax[1,1].plot(x2, y2, 'k')
ax[1,1].vlines(abs(x2[np.argmax(y2)]), 1.1*min(y2), 1.1*max(y2), label=r'$f_{alias} = $' + str(round(abs(x2[np.argmax(y2)]),3)))
ax[1,1].legend(loc='upper left')
#-------------- sinal
xloc = np.arange(0,len(f_2),.001)
n2 = np.cos(x2[np.argmax(y2)]*2*pi*xloc)
#
sr_real = 40#20
s2 = s1[0::sr_real]
xs2 = xs1[0::sr_real]
#
ax[1,0].plot(xs1, s1, color='darkgray',  linestyle='solid', linewidth=.75)#(0, (3, 5, 1, 5)))
ax[1,0].plot(xloc,n2, color='k', linestyle='solid', linewidth=1)#(0, (3, 5, 1, 5)))
ax[1,0].plot(xs2, s2, 'k.')
ax[1,0].set_ylabel(r'$t_{sampl} = $'+str(sr_real*sr_0), rotation=0, labelpad=25.)
#------------------------------------ 3 linha
sr = 5 #3 
#-------------- psd
x3, y3 = psd(s1, sr)
ax[2,1].plot(x3, y3, 'k')
ax[2,1].vlines(abs(x3[np.argmax(y3)]), 1.1*min(y3), 1.1*max(y3), label=r'$f_{alias} = $' + str(round(abs(x3[np.argmax(y3)]),3)))
ax[2,1].legend(loc='upper left')
#-------------- sinal
n3 = np.cos(x3[np.argmax(y3)]*2*pi*xloc)
#
sr_real = 25 #15
s3 = s1[0::sr_real]
xs3 = xs1[0::sr_real]
#
ax[2,0].plot(xs1, s1, color='darkgray',  linestyle='solid', linewidth=.75)#(0, (3, 5, 1, 5)))
ax[2,0].plot(xloc, n3, color='k', linestyle='solid', linewidth=1)#(0, (3, 5, 1, 5)))
ax[2,0].plot(xs3, s3, 'k.')
ax[2,0].set_ylabel(r'$t_{sampl} = $ '+str(sr_real*sr_0), rotation=0, labelpad=25.)
#------------------------------------ 4 linha
sr = 4
#-------------- psd
x4, y4 = psd(s1, sr)
ax[3,1].plot(x4, y4, 'k')
ax[3,1].vlines(abs(x4[np.argmax(y4)]), 1.1*min(y4), 1.1*max(y4), label=r'$f_{alias} = $' + str(round(abs(x4[np.argmax(y4)]),3)))
ax[3,1].legend(loc='upper left')
#
ax[3,1].set_xlabel('f', size=14)
#-------------- sinal
n4 = np.cos(x4[np.argmax(y4)]*2*pi*xloc)
#
sr_real = 16 #8
s4 = s1[0::sr_real]
xs4 = xs1[0::sr_real]
#
ax[3,0].plot(xs1, s1, color='darkgray',  linestyle='solid', linewidth=.75)#(0, (3, 5, 1, 5)))
ax[3,0].plot(xloc, n4, color='k', linestyle='solid', linewidth=1)#(0, (3, 5, 1, 5)))
ax[3,0].plot(xs4, s4, 'k.')
ax[3,0].set_ylabel(r'$t_{sampl} = $ '+str(sr_real*sr_0), rotation=0, labelpad=25.)
ax[3,0].set_xlabel('t', size=14)
#------------------------------------------------------------------------
for i in range(4):
    ax[i,0].set_ylim(-1.2, 1.2)
    ax[i,1].set_yticks([])
    ax[i,0].set_xlim(-3,n_pontos+3)
    ax[i,1].set_xlim(-.1,.1)
    #ax[i,1].set_xticks([-.4, -.3, -.2, -.1, 0., .1, .2, .3, .4])
    #ax[i,1].set_xticklabels(['-0.4', '', '-0.2', '', '0.0', '', '0.2', '', '0.4'])
    #ax[i,1].set_xlim(-.4,.4)
#------------------------------------------------------------------------
#fig2.suptitle('Effects of Sampling')
fig2.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=.2, hspace=.3)
fig2.savefig('sampling_20.jpg', dpi=400, bbox_inches='tight')
plt.show()
#------------------------------------------------------------------------ FIM
plt.close('all')
