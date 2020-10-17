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
    ft = fft.fft(f, N)
    ft_shifted = fft.fftshift(ft)
    aft = np.abs(ft)**2
    aft_shifted = (abs(ft_shifted)**2)
    #
    freq = np.fft.fftfreq(N, d=sr)
    freqs = np.concatenate([freq[int(len(freq)/2):],[0]])
    freqs = np.concatenate([freqs, freq[1:int(len(freq)/2)]])
    #
    return freq, aft
#
def spec(signal, sr=1, f_final=10, N=2048):
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
    return freq[:i], aft[:i]
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
    return r
#------------------------------------------------------------------------    
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
#------------------------------------
ti_2 = -6
tf_2 = 6
t_2 = np.linspace(ti_2, tf_2, N)
res_2 = abs(ti_2 - tf_2)/len(t_2)

g_2 = linear_chirp(t_2)
#------------------------------------
ti_3 = -4
tf_3 = 4
t_3 = np.linspace(ti_3, tf_3, N)
res_3 = abs(ti_3 - tf_3)/len(t_3)

g_3 = square_chirp(t_3)
#------------------------------------
ti_4 = 0
tf_4 = 2
t_4 = np.linspace(ti_4, tf_4, N)
res_4 = abs(ti_4 - tf_4)/len(t_4)

g_4 = hyper_chirp(t_4)
#========================================================================
#                                ESPECTROS
#------------------------------------------------------------------------
name = 'Hyperbolic'
chirp = g_4
t = t_4
res = res_4
window_size = .5
xjump = .5
xround = 1
yamount = 8
yround = 2

tau = np.array(t)
f_final = 10
plot_title = 'Spectrogram of '+name+' chirp - window size = ' + str(window_size)
fig_name = 'TEST_'+name+'_ws' + str(window_size)

# Defining spectrogram arrays
spec_1 = np.zeros([f_final, len(tau)])
spec_2 = np.zeros([f_final, len(tau)])
spec_3 = np.zeros([f_final, len(tau)])
spec_4 = np.zeros([f_final, len(tau)]) 
spec_5 = np.zeros([f_final, len(tau)]) 
spec_6 = np.zeros([f_final, len(tau)]) 
# Creating spectrograms
for i in range(len(tau)):
    # Shifted times
    t_arr = t - tau[i] 
    #------------------------------------ rectangular 
    # Window at shifted times
    w1 = rect(t_arr, window_size)
    # Windowed spectra centered at tau[i]
    x1, y1 = spec(chirp * w1, res, f_final, N)
    # Updating spectrogram array 
    spec_1[:,i]=y1 
    #------------------------------------ hanning 
    w2 = hanning(t_arr, window_size)
    x2, y2 = spec(chirp * w2, res, f_final, N)
    spec_2[:,i]=y2
    #------------------------------------ barllet
    w3 = barllet(t_arr, window_size)
    x3, y3 = spec(chirp * w3, res, f_final, N)
    spec_3[:,i]=y3
    #------------------------------------ papoulis
    w4 = papoulis(t_arr, window_size)
    x4, y4 = spec(chirp * w4, res, f_final, N)
    spec_4[:,i]=y4
    #------------------------------------ hamming
    w5 = hamming(t_arr, window_size)
    x5, y5 = spec(chirp * w5, res, f_final, N)
    spec_5[:,i]=y5
    #------------------------------------ tukey
    w6 = tukeywin(len(t_arr), window_size)
    x6, y6 = spec(chirp * w6, res, f_final, N)
    spec_6[:,i]=y6

# Deleting garbage
del w1, w2, w3, w4, w5, w6
del y1, y2, y3, y4, y5, y6
del x2, x3, x4, x5, x6

# Spectrograms for plot 
spec_1_p = 20*np.log10(np.abs(spec_1))
spec_2_p = 20*np.log10(np.abs(spec_2))
spec_3_p = 20*np.log10(np.abs(spec_3))
spec_4_p = 20*np.log10(np.abs(spec_4))
spec_5_p = 20*np.log10(np.abs(spec_5))
spec_6_p = 20*np.log10(np.abs(spec_6))
#========================================================================
#                                PLOTS
#------------------------------------------------------------------------
fig, ax = plt.subplots(3, 2, figsize=(10,8)) # (3 x 2) subplots
# Global settings
# colormap
my_cmap = matplotlib.cm.get_cmap('jet')
fig.suptitle(plot_title, size=16)
# x ticks and labels
t_ticks  = np.arange(0, len(t)+1, 1/res/xjump,dtype=int)-1
t_labels = list(np.round(t[t_ticks], xround))
# minimum and maximum values for color value in all plots
p_min = np.min([spec_1_p, spec_2_p, spec_3_p, spec_4_p, spec_5_p])
p_max = np.max([spec_1_p, spec_2_p, spec_3_p, spec_4_p, spec_5_p])
#---------------------------------- Row 1 Col 1
ax[0,0].set_title('Rectangular window', size = 13)
im = ax[0,0].contourf(spec_1_p, 256, origin='upper', cmap=my_cmap, vmin=p_min, vmax=p_max)
ax[0,0].set_ylabel('frequency', size=12)
#---------------------------------- Row 2 Col 1
ax[1,0].set_title('Hanning window', size=13)
ax[1,0].contourf(spec_2_p, 256, origin='upper', cmap=my_cmap, vmin=p_min, vmax=p_max)
ax[1,0].set_xticks(t_ticks)
ax[1,0].set_xticklabels(t_labels)
ax[1,0].set_ylabel('frequency', size=12)
#---------------------------------- Row 3 Col 1
ax[2,0].set_title('Barllet window', size=13)
ax[2,0].contourf(spec_3_p, 256, origin='upper', cmap=my_cmap, vmin=p_min, vmax=p_max)
ax[2,0].set_ylabel('frequency', size=12)
ax[2,0].set_xlabel('time', size=12)
#---------------------------------- Row 1 Col 2
ax[0,1].set_title('Papoulis window', size=13)
ax[0,1].contourf(spec_4_p, 256, origin='upper', cmap=my_cmap, vmin=p_min, vmax=p_max)
#---------------------------------- Row 1 Col 2
ax[1,1].set_title('Hamming window', size=13)
ax[1,1].contourf(spec_5_p, 256, origin='upper', cmap=my_cmap, vmin=p_min, vmax=p_max)
#---------------------------------- Row 1 Col 3
ax[2,1].set_title('Tukey window', size=13)
ax[2,1].contourf(spec_6_p, 256, origin='upper', cmap=my_cmap, vmin=p_min, vmax=p_max)
ax[2,1].set_xlabel('time', size=12)
#---------------------------------- End of individual-plot settings
for i in range(3):
    # Setting same x ticks and tick labels for all plots 
    ax[i,0].set_xticks(t_ticks)
    ax[i,0].set_xticklabels(t_labels)
    ax[i,1].set_xticks(t_ticks)
    ax[i,1].set_xticklabels(t_labels)
# Setting spacing between plots
fig.subplots_adjust(left=None, bottom=None, right=.85, top=None, wspace=.2, hspace=.4)
# Setting colorbar
cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
cb = fig.colorbar(im, cax=cbar_ax)
cb.set_label('Log of Power', fontsize=15)
#---------------------------------- Saving and showing plot
fig.savefig(fig_name+'.jpg', dpi=600, bbox_inches='tight')
plt.show()
#------------------------------------------------------------------------ FIM
plt.close('all')
