#========================================================================
#                                  IMPORTS
#------------------------------------------------------------------------
import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt
pi = np.pi
#========================================================================
#                                  INPUTS
#------------------------------------------------------------------------
periodo_1 = 10.
periodo_2 = 20.
n_pontos = 60
#========================================================================
#                                  SINAIS
#------------------------------------------------------------------------
# Sinais analisados
n_total = n_pontos
n = np.arange(n_total)
f10 = np.cos(2*pi*n/periodo_1)    # f1 original
f20 = np.cos(2*pi*n/periodo_2)    # f2 original
f1 = f10#np.concatenate([f10,f20])    # f1, f2
f2 = f20#np.concatenate([f20,f10])    # f2, f1
f3 = f10 + f20                    # f1 + f2
#
# Parametros dos sinais (para plots)
t_1 = int(n_pontos/periodo_1)
t_2 = int(n_pontos/periodo_2)
#========================================================================
#                               PLOTS (tempo)
#------------------------------------------------------------------------'+str(int(periodo_2))+'
plt.figure(figsize=[15,7])
#
plt.subplot(3,2,1)
plt.title(r'$f_{1}(n) = \cos(\frac{2 \pi n}{'+str(int(periodo_1))+r'})$, $f_{2}(n) = \cos(\frac{2 \pi n}{'+str(int(periodo_2))+'})$', size=16)
plt.plot(f1, 'r.-')
plt.ylabel(r'$f_{1}(n)$', rotation='horizontal', ha='right', size=14)
plt.xlim(0, n_total)
#
plt.subplot(3,2,3)
plt.ylabel(r'$f_{2}(n)$', rotation='horizontal', ha='right', size=14)
plt.plot(f2, 'b.-')
plt.xlim(0, n_total)
#
plt.subplot(3,2,5)
plt.ylabel(r'$f_{3}(n) = f_{1}(n) + f_{2}(n)$', rotation='horizontal', ha='right', size=14)
plt.plot(f3, 'g.-')
plt.xlim(0, n_total)
#
plt.xlabel('samples n', size=14)
#========================================================================
#                                 ESPECTROS
#------------------------------------------------------------------------
'''
numpy.fft.fft:
When the input a is a time-domain signal and A = fft(a): 
. np.abs(A) is its amplitude spectrum; 
. np.abs(A)**2 is its power spectrum; 
. np.angle(A) is the phase spectrum.
'''
N = 2048
ft1 = fft.fft(f1, N)
ft2 = fft.fft(f2, N)
ft3 = fft.fft(f3, N)
ft1_shifted = fft.fftshift(ft1)
ft2_shifted = fft.fftshift(ft2)
ft3_shifted = fft.fftshift(ft3)
ft1_shifted = ft1_shifted.real
ft2_shifted = ft2_shifted.real
ft3_shifted = ft3_shifted.real
aft1 = np.abs(ft1)
aft2 = np.abs(ft2)
aft3 = np.abs(ft3)
aft1_shifted = abs(ft1_shifted)**2
aft2_shifted = abs(ft2_shifted)**2
aft3_shifted = abs(ft3_shifted)**2
#========================================================================
#                             PLOTS (frequencia)
#------------------------------------------------------------------------
xt = np.arange(N)/N
xt_shifted = np.arange(-N/2,N/2)/float(N)
#
plt.subplot(3,2,2)
plt.title('Fourier Transform - real part', size=16)#r'$|\widehat{f}(j)|$, onde $\widehat{f}(j) = \frac{1}{N} \sum_{n=0}^{N-1} f(n) \exp \left( \frac{- i j 2 \pi n}{N} \right)$ e N = ' + str(N) + '\n'
plt.plot(xt_shifted, ft1_shifted, 'r-', label = r'$\hat{f}_{1}(j)$')
plt.xlim(-.2,.2)
#plt.vlines(-1/periodo_1, -np.max(aft1_shifted)*.1, 1.1*np.max(aft1_shifted), colors='tab:red', linestyles='-.', label=r'$j = -\frac{1}{'+str(int(periodo_1))+'} = -$' + str(np.round((1./periodo_1),2)))
#plt.vlines(1/periodo_1, -np.max(aft1_shifted)*.1, 1.1*np.max(aft1_shifted), colors='tab:red', linestyles='dashed', label=r'$j = \frac{1}{'+str(int(periodo_1))+'} = $' + str(np.round((1./periodo_1),2)))
#plt.ylim(-np.max(ft1_shifted)*.05, 1.05*np.max(ft1_shifted))
plt.legend(loc='upper center', fontsize=12)
#
plt.subplot(3,2,4)
plt.plot(xt_shifted, ft2_shifted, 'b-', label = r'$\hat{f}_{2}(j)$')
plt.xlim(-.2,.2)
#plt.vlines(-1/periodo_2, -np.max(aft1_shifted)*.1, 1.1*np.max(aft1_shifted), colors='tab:blue', linestyles='-.', label=r'$j = -\frac{1}{'+str(int(periodo_2))+'} = -$' + str(np.round((1./periodo_2),2)))
#plt.vlines(1/periodo_2, -np.max(aft1_shifted)*.1, 1.1*np.max(aft1_shifted), colors='tab:blue', linestyles='dashed', label=r'$j = \frac{1}{'+str(int(periodo_2))+'} = $' + str(np.round((1./periodo_2),2)))
#plt.ylim(-np.max(ft1_shifted)*.05, 1.05*np.max(ft1_shifted))
plt.legend(loc='upper center', fontsize=12)
#
plt.subplot(3,2,6)
plt.plot(xt_shifted, ft1_shifted + ft2_shifted, 'k+', label =  r'$\hat{f}_{1}(j) + \hat{f}_{2}(j)$')
plt.plot(xt_shifted, ft3_shifted, 'g-', label = r'$\hat{f}_{3}(j)$')
plt.xlabel('frequencies j', size=14)
plt.xlim(-.2,.2)
plt.legend(loc='upper center', fontsize=12)
#
plt.savefig('exemplo_FT_linear.jpg', dpi=400, bbox_inches='tight')
plt.show()
#========================================================================
