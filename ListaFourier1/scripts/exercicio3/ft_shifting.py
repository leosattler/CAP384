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
res = .1
a = 2.
b=5
n_pontos = 64
#========================================================================
#                                  SINAIS
#------------------------------------------------------------------------
# Sinais analisados
#n = np.arange(-8*pi,8*pi, res)
n=np.arange(0, n_pontos, res)
#f1 = np.exp(-periodo_1*n)
#f1=sq_wave(n, 0, 2*np.pi, 4)
f1 = np.cos(2*pi*n/periodo_1)    # f1 original
#f2 = np.exp((periodo_1-a)*n)
#f2=sq_wave(n, 0, 2*2*np.pi, 4)
f2 = np.cos(2*pi*(n-a)/periodo_1) 
#f3 = f1*np.exp(-2.j*pi*n*b)
#f3=sq_wave(n, 0, np.pi, 2)
f3 = np.cos(2*pi*(n)/periodo_1) #* np.exp(2.j*pi*xt_shifted_1*a)
#========================================================================
#                               PLOTS (tempo)
#------------------------------------------------------------------------
#
plt.figure(figsize=[15,7])
#
a=int(a)
plt.subplot(3,2,1)
plt.title(r'$f_{1}(t) = \cos(\frac{2 \pi t}{'+str(int(periodo_1))+'})$, $f_{2}(t) = f_{1}(t-t_{0})$, ' + r'$f_{3}(t) = f_{1}(t) \exp(-2\pi i \xi_{0} t)$' + '\n' + 'with '+r'$t_{0} = $'+str(a)+', '+r'$\xi_{0} = 1/t_{0}$', size=15)
#
plt.plot(n,f1, 'r.-')
plt.ylabel(r'$f_{1}(t)$', rotation='horizontal', ha='right', size=14)
#plt.xlim(-.1,4)
#
plt.subplot(3,2,3)
plt.plot(n,f2, 'b.-')
plt.ylabel(r'$f_{2}(t)$', rotation='horizontal', ha='right', size=14)
#plt.xlim(-.1,4)
#
#plt.subplot(3,2,5)
#plt.ylabel(r'$f_{3}(n)$', rotation='horizontal', ha='right')
#plt.plot(n,f2, 'g.-')
#plt.xlim(-.1,4)
#========================================================================
#                             PLOTS (frequencia)
#------------------------------------------------------------------------
#
xt_shifted_1, aft_shifted_1 = psd(f1, 1/res)
xt_shifted_2, aft_shifted_2 = psd(f2, 1/res)
#------------------------------------------------------------------------
f3 = np.cos(2*pi*(n)/periodo_1) * np.exp(2.j*pi*n*1/a)
plt.subplot(3,2,5)
plt.ylabel(r'$f_{3}(t)$', rotation='horizontal', ha='right', size=14)
plt.plot(n,f3.real, 'g.-')
plt.xlabel(r'$t$', size=14)
xt_shifted_3, aft_shifted_3 = psd(f3, 1/res)
#------------------------------------------------------------------------
a=float(a)
plt.subplot(3,2,2)
plt.title('Fourier Transform  - real part\n', size=15)
plt.plot(xt_shifted_1, aft_shifted_1, 'r-', label=r'$\hat{f_{1}}(\xi)$')#, label = str(t_1) + ' e ' + str(t_2) + ' periodos')
plt.legend(loc=2, fontsize=12)
plt.xlim(-.75,.75)
#plt.xlim(-0.0075,0.0075)
#------------------------------------------------------------------------
plt.subplot(3,2,4)
#
fft3 = aft_shifted_1 * np.exp(2.j*pi*xt_shifted_1*1/a)
plt.plot(xt_shifted_2, fft3.real, 'k+', label=r'$\hat{f_{1}}(\xi) e^{-2\pi i \xi t_{0}}$')#, lab
plt.plot(xt_shifted_2, aft_shifted_2, 'b-', label=r'$\hat{f_{2}}(\xi)$')
plt.legend(loc=2, fontsize=12)
plt.xlim(-.75,.75)
#plt.xlim(-0.0075,0.0075)
#------------------------------------------------------------------------
plt.subplot(3,2,6)
plt.plot(xt_shifted_1+1/a, aft_shifted_1, 'k+', label=r'$\hat{f_{1}}(\xi - \xi_{0})$')#, lab
plt.plot(xt_shifted_3, aft_shifted_3, 'g-', label=r'$\hat{f_{3}}(\xi)$')
plt.legend(loc=2, fontsize=12)
plt.xlim(-.75,.75)
plt.xlabel(r'$\xi$', size=14)
#
plt.savefig('time_shifting.jpg', dpi=400, bbox_inches='tight')
plt.show()
#========================================================================
