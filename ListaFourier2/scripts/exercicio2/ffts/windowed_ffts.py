#========================================================================
#                                  IMPORTS
#------------------------------------------------------------------------
import numpy as np
import numpy.fft as fft
import matplotlib
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
    ft = fft.hfft(f, N)
    ft_shifted = fft.fftshift(ft)
    aft = np.abs(ft)**2
    aft_shifted = (abs(ft_shifted)**2)
    #
    freq = np.fft.fftfreq(N, d=sr)
    freqs = np.concatenate([freq[int(len(freq)/2):],[0]])
    freqs = np.concatenate([freqs, freq[1:int(len(freq)/2)]])
    #
    return freqs, ft_shifted
#
def spec(signal, sr=1, f_final=10, N = 2048):
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
    aft_shifted = (abs(ft_shifted)**2)
    #
    freq = np.fft.fftfreq(N, d=sr)
    freqs = np.concatenate([freq[int(len(freq)/2):],[0]])
    freqs = np.concatenate([freqs, freq[1:int(len(freq)/2)]])
    #
    for i in range(len(freq)):
        if abs(i-f_final)<=.01:
            break
    #
    return freq[:i], aft[:i]#_shifted
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
#                            FUNCOES EXERCICIO 2
#------------------------------------------------------------------------
def rect(t, window_size = 1/2.):
    r = np.zeros(len(t))
    
    r[np.where(abs(t) <= window_size)] = 1
    
    return r
#------------------------------------------------------------------------
def hanning(t, window_size = 1/2.):
    r = np.zeros(len(t))

    r[np.where(abs(t) <= window_size)] = 1/2. + (1/2.) * np.cos(2 * pi * t[np.where(abs(t) <= window_size)])
    
    return r
#------------------------------------------------------------------------
def hamming(t, window_size = 1/2.):
    r = np.zeros(len(t))
    
    r[np.where(abs(t) <= window_size)] = .54 + .46 * np.cos(2 * pi * t[np.where(abs(t) <= window_size)])
    
    return r
#------------------------------------------------------------------------
def barllet(t, window_size = 1/2.):
    r = np.zeros(len(t))
    
    r[np.where(abs(t) <= window_size)] = 1 - 2 * np.abs(t[np.where(abs(t) <= window_size)])
    
    return r
#------------------------------------------------------------------------
def papoulis(t, window_size = 1/2.):
    r = np.zeros(len(t))
    
    r[np.where(abs(t) <= window_size)] = (1/pi) * np.abs(np.sin(2*pi*t[np.where(abs(t) <= window_size)])) + \
                                         (1 - 2 * np.abs(t[np.where(abs(t) <= window_size)])) * \
                                         np.cos(2*pi*t[np.where(abs(t) <= window_size)])
#------------------------------------------------------------------------    
    return r
def tukeywin(window_length, alpha=0.5):
    '''The Tukey window, also known as the tapered cosine window, can be regarded as a cosine lobe of width \alpha * N / 2
    that is convolved with a rectangle window of width (1 - \alpha / 2). At \alpha = 1 it becomes rectangular, and
    at \alpha = 0 it becomes a Hann window.
 
    We use the same reference as MATLAB to provide the same results in case users compare a MATLAB output to this function
    output
 
    Reference
    ---------
    http://www.mathworks.com/access/helpdesk/help/toolbox/signal/tukeywin.html
 
    '''
    # Special cases
    if alpha <= 0:
        return np.ones(window_length) #rectangular window
    elif alpha >= 1:
        return np.hanning(window_length)
 
    # Normal case
    x = np.linspace(0, 1, window_length)
    w = np.ones(x.shape)
 
    # first condition 0 <= x < alpha/2
    first_condition = x<alpha/2
    w[first_condition] = 0.5 * (1 + np.cos(2*np.pi/alpha * (x[first_condition] - alpha/2) ))
 
    # second condition already taken care of
 
    # third condition 1 - alpha / 2 <= x <= 1
    third_condition = x>=(1 - alpha/2)
    w[third_condition] = 0.5 * (1 + np.cos(2*np.pi/alpha * (x[third_condition] - 1 + alpha/2))) 
 
    return w
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
x_3, ft_3 = psd(g_3, res_3, N)
#------------------------------------
ti_4 = 0
tf_4 = 2
t_4 = np.linspace(ti_4, tf_4, N)
res_4 = abs(ti_4 - tf_4)/len(t_4)

g_4 = hyper_chirp(t_4)
x_4, ft_4 = psd(g_4, res_4, N)
#========================================================================
#                                ESPECTROS
#------------------------------------------------------------------------
tau = t_1
res = res_1
window_size = .25
xjump = 1
xround = 1

f_final = 10
plot_title = 'FT of windows - window size = ' + str(window_size)
fig_name = 'Windows_FFTs_ws' + str(window_size)

w1 = rect(tau, window_size)
#------------------------------------
w2 = hanning(tau, window_size)
#------------------------------------
w3 = barllet(tau, window_size)
#------------------------------------
w4 = papoulis(tau, window_size)
#------------------------------------
w5 = hamming(tau, window_size)
#------------------------------------
w6 = tukeywin(len(tau), window_size)
#
freq_1, psd_1 = psd(w1, res, N)
freq_2, psd_2 = psd(w2, res, N)
freq_3, psd_3 = psd(w3, res, N)
freq_4, psd_4 = psd(w4, res, N)
freq_5, psd_5 = psd(w5, res, N)
freq_6, psd_6 = psd(w6, res, N)
#========================================================================
#                                PLOTS
#------------------------------------------------------------------------
fig, ax = plt.subplots(3, 2, figsize=(9,8))
my_cmap = matplotlib.cm.get_cmap('jet')
fig.suptitle(plot_title, size=16)
#
ax[0,0].set_title('Rectangular window', size = 13)
ax[0,0].plot(freq_1, psd_1.real, 'r-', label='real')
ax[0,0].plot(freq_1, psd_1.imag, 'b--', label='imag')
ax[0,0].legend(bbox_to_anchor=(1.75,1.6), loc="upper left")
ax[0,0].set_xlim(-4,4)
#
ax[1,0].set_title('Hanning window', size=13)
ax[1,0].plot(freq_2, psd_2.real, 'r-')
ax[1,0].plot(freq_2, psd_2.imag, 'b--')
ax[1,0].set_xlim(-4,4)
#
ax[2,0].set_title('Barllet window', size=13)
ax[2,0].plot(freq_3, psd_3.real, 'r-')
ax[2,0].plot(freq_3, psd_3.imag, 'b--')
ax[2,0].set_xlim(-4,4)
ax[2,0].set_xlabel('frequency', size=12)
#
ax[0,1].set_title('Papoulis window', size=13)
ax[0,1].plot(freq_4, psd_4.real, 'r-')
ax[0,1].plot(freq_4, psd_4.imag, 'b--')
ax[0,1].set_xlim(-4,4)
#
ax[1,1].set_title('Hamming window', size=13)
ax[1,1].plot(freq_5, psd_5.real, 'r-')
ax[1,1].plot(freq_5, psd_5.imag, 'b--')
ax[1,1].set_xlim(-4,4)
#
ax[2,1].set_title('Tukey window', size=13)
ax[2,1].plot(freq_6, psd_6.real, 'r-')
ax[2,1].plot(freq_6, psd_6.imag, 'b--')
ax[2,1].set_xlim(-4,4)
ax[2,1].set_xlabel('frequency', size=12)
#------------------------------------------------------------------------
#cb.set_label('Power')
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=.2, hspace=.4)
fig.savefig(fig_name+'.jpg', dpi=600, bbox_inches='tight')
plt.show()
#------------------------------------------------------------------------ FIM
plt.close('all')
