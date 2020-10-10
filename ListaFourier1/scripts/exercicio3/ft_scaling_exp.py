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
    ft = fft.hfft(f, N)
    ft_shifted = fft.fftshift(ft)
    ft_shifted_real = np.real(fft.fftshift(ft))
    ft_shifted_imag = np.imag(fft.fftshift(ft))
    aft = np.abs(ft)**2
    aft_shifted = abs(ft_shifted)**2
    #
    freq = np.fft.fftfreq(N, d=1/sr)
    freqs = np.concatenate([freq[int(len(freq)/2):],[0]])
    freqs = np.concatenate([freqs, freq[1:int(len(freq)/2)]])
    #
    return freqs, ft_shifted#, aft_shifted[int(N/2)+1:]#, [int(N/2)+1:]
#
# Square wave (step) function
def sq_wave(x, center, size, height):
    y = np.zeros(len(x))
    # y = 1, if -pi < x < pi
    #while step_it < xf:
    y[np.where(np.logical_and(x>=center-size/2, x<=+size/2+center))] = height
    #step_it = step_it + 2*pi
    # y = -1, elsewhere
    #
    return y
#
def tr(x, cte):
    return (2*cte)/(cte**2+x**2)
#t=np.arange(-10,10,.1)
#y=tr(t, 2.)
#plt.plot(y)
#plt.show()
#sys.exit()
#========================================================================
#                                  INPUTS
#------------------------------------------------------------------------
periodo_1 = 1.
res = .1
a = 2.
n_pontos = 64
#========================================================================
#                                  SINAIS
#------------------------------------------------------------------------
# Sinais analisados
#n = np.arange(-8*pi,8*pi, res)
n=np.arange(0, n_pontos, res)
f1 = np.exp(-periodo_1*n)
#f1=sq_wave(n, 0, 2*np.pi, 4)
#f1 = np.cos(2*pi*n/periodo_1)    # f1 original
f2 = np.exp(-periodo_1*n*a)
#f2=sq_wave(n, 0, 2*2*np.pi, 4)
f3 =np.exp(-periodo_1*n/a)/np.abs(a)
#f3=sq_wave(n, 0, np.pi, 2)
#f3 = np.cos(2*pi*(n/a)/periodo_1)
#========================================================================
#                               PLOTS (tempo)
#------------------------------------------------------------------------
#
plt.figure(figsize=[15,7])
#
a=int(a)
plt.subplot(3,2,1)
plt.title(r'$f_{1}(t) = \exp(-|t|)$, $f_{2}(t) = f_{1}(at)$, ' + r'$f_{3}(t) = \frac{1}{|a|} f_{1} \left(\frac{t}{a}\right)$,' + '\n'+ r' $a$ = '+str(a), size=15)
#
plt.plot(n,f1, 'r.-')
plt.ylabel(r'$f_{1}(t)$', rotation='horizontal', ha='right', size=14)
plt.xlim(-.01,2.5)
plt.vlines(1., 0, 1/np.e, label = r'$y=1/e, x=1$')
plt.hlines(1/np.e, 0, 1)
plt.legend(loc='upper right', fontsize=12)
plt.ylim(0,1)
#
plt.subplot(3,2,3)
plt.plot(n,f2, 'b.-')
plt.ylabel(r'$f_{2}(t)$', rotation='horizontal', ha='right', size=14)
plt.vlines(1./a, 0, 1/np.e, label = r'$y=1/e, x=1/a$')
plt.hlines(1/np.e, 0, 1/a)
plt.legend(loc='upper right', fontsize=12)
plt.xlim(-.01,2.5)
plt.ylim(0,1)
#
plt.subplot(3,2,5)
plt.ylabel(r'$f_{3}(t)$', rotation='horizontal', ha='right', size=14)
plt.plot(n,f3, 'g.-')
plt.xlabel('t', size=14)
plt.vlines(a*1., 0, 1/np.e/a, label = r'$y=1/(ae), x=a$')
plt.hlines(1/np.e/a, 0, a*1.)
plt.legend(loc='upper right', fontsize=12)
plt.xlim(-.01,2.5)
plt.ylim(0,1)
#========================================================================
#                             PLOTS (frequencia)
#------------------------------------------------------------------------
#
xt_shifted_1, aft_shifted_1 = psd(f1, 1/res)
xt_shifted_2, aft_shifted_2 = psd(f2, 1/res)
xt_shifted_3, aft_shifted_3 = psd(f3, 1/res)
aft_shifted_1 = aft_shifted_1/np.sqrt(len(n))
aft_shifted_2 = aft_shifted_2/np.sqrt(len(n))
aft_shifted_3 = aft_shifted_3/np.sqrt(len(n))
#
alt = 1/a
#
larg = a 
#------------------------------------------------------------------------
a=float(a)
plt.subplot(3,2,2)
plt.title('Fourier Transform - real part\n', size=15)
plt.plot((1/res)*xt_shifted_1, aft_shifted_1, 'r-', label=r'$\hat{f}_{1}(\xi)$')
plt.legend(loc='upper right', fontsize=12)
#
plt.hlines(alt/2,-larg,larg,  label ='width = '+r'$2a$' ) #height = '+r'$1/a$' + ', \n
plt.legend(loc='upper right', fontsize=12)
plt.xticks(list(np.arange(-10,10+2,2)))
plt.xlim(-10,10)
plt.ylim(0,1)
#------------------------------------------------------------------------
plt.subplot(3,2,4)
plt.plot((1/res)*xt_shifted_2, aft_shifted_2, 'b-', label=r'$\hat{f}_{2}(\xi)$')
plt.hlines(alt/4,-2*larg,2*larg)
plt.xticks(list(np.arange(-10,10+2,2)))
plt.xlim(-10,10)
plt.ylim(0,1)
plt.legend(loc='upper right', fontsize=12)
#
#------------------------------------------------------------------------
plt.subplot(3,2,6)
plt.plot((1/res)*xt_shifted_3, aft_shifted_3, 'g-', label=r'$\hat{f}_{3}(\xi)$')
plt.hlines(alt/2,-larg/2,larg/2)
plt.ylim(0,1)
plt.xlabel(r'$\xi$', size=14)
plt.legend(loc='center right', fontsize=12)
plt.xticks(list(np.arange(-10,10+2,2)))
plt.xlim(-10,10)
#------------------------------------------------------------------------
plt.savefig('exp_scaling.jpg', dpi=400, bbox_inches='tight')
plt.show()
#========================================================================
