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
    ft = fft.fft(f, N)
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
def psdH(signal, sr=1):
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
def tr_sqw(x, size, height):
    return size*height*np.sinc(x*size/2.)
#
def tr(x, cte):
    return (2*cte)/(cte**2+x**2)
'''
t=np.arange(-10,10,.1)
y=tr_sqw(t, 2.*2*pi, 4)
plt.plot(t,y)
plt.show()
'''
#sys.exit()
#========================================================================
#                                  INPUTS
#------------------------------------------------------------------------
periodo_1 = 10.
res = .5
n_pontos = 64
#========================================================================
#                                  SINAIS
#------------------------------------------------------------------------
# Sinais analisados
n = np.arange(-8*pi,8*pi, res)
#=np.arange(0, n_pontos, res)
#f1 = np.exp(-periodo_1*n)
#f1=sq_wave(n, 0, 2*np.pi, 4)
f1 = 1.j*np.cos(2*pi*n/periodo_1)   # f1 original
#f2 = np.exp((periodo_1-a)*n)
#f2=sq_wave(n, 0, 2*2*np.pi, 4)
f2 = 1.j*np.sin(2*pi*n/periodo_1)
#f3 = f1*np.exp(-2.j*pi*n*b)
#f3=sq_wave(n, 0, np.pi, 2)
#========================================================================
#                               PLOTS (tempo)
#------------------------------------------------------------------------
#
plt.figure(figsize=[12,5])
#
#------------------------------------------------------------------------
plt.subplot(2,2,1)
plt.grid('on')
plt.title(r'$f_{1}(t) =\.{\i} \cos(\frac{2 \pi t}{'+str(int(periodo_1))+ '})$'+ r', $f_{2}(t) =\.{\i} \sin(\frac{2 \pi t}{'+str(int(periodo_1))+'})$' + '\n', size=15)
#plt.title(r'$f_{1}(t) = \.{\i} t^{2}$'+ r', $f_{2}(t) = \.{\i} t^{3}$' + '\n', size=15)
#\.{\i}
plt.plot(n,f1.real+f1.imag, 'k.-')
plt.ylabel(r'$f_{1}(n)$', rotation='horizontal', ha='right', size=14)
#plt.xlim(-.1,4)
#------------------------------------------------------------------------
plt.subplot(2,2,3)
plt.grid('on')
plt.plot(n,f2.real+f2.imag, 'k.-')
plt.ylabel(r'$f_{2}(n)$', rotation='horizontal', ha='right', size=14)
#
plt.xlabel('t', size=14)
#plt.xlim(-.1,4)
#========================================================================
#                             PLOTS (frequencia)
#------------------------------------------------------------------------
#
xt_shifted_1, aft_shifted_1 = psd(f1, 1/res)
xt_shifted_2, aft_shifted_2 = psd(f2, 1/res)
#
#------------------------------------------------------------------------
plt.subplot(2,2,2)
plt.title('Fourier Transform \n', size=15)
#r'$|\widehat{f}(j)|^{2}$, onde $\widehat{f}(j) = \frac{1}{N} \sum_{n=0}^{N-1} f(n) \exp \left( \frac{- i j 2 \pi n}{N} \right)$ e N = ' + str(N) + '\n')
#plt.plot(xt_shifted_1, aft_shifted_1.real, 'r-', label=r'$\hat{f}_{1}(\xi)$ real')
plt.plot(xt_shifted_1, aft_shifted_1.imag, 'b-', label=r'$\hat{f}_{1}(\xi)$ imag')
plt.legend(loc='lower right', fontsize=12)
plt.grid('on')
#plt.xlim(-0.01,0.01)
#
#------------------------------------------------------------------------
plt.subplot(2,2,4)
#
plt.plot(xt_shifted_2, aft_shifted_2.real, 'r-', label=r'$\hat{f}_{2}(\xi)$ real')
#plt.plot(xt_shifted_2, aft_shifted_2.imag, 'b-', label=r'$\hat{f}_{2}(\xi)$ imag')
plt.legend(loc=1, fontsize=12)
#plt.xlim(-0.01,0.01)
plt.xlabel(r'$\xi$', size=14)
plt.grid('on')
#
plt.savefig('ft_symetries_imag_trig.jpg', dpi=400, bbox_inches='tight')
plt.show()
#========================================================================
