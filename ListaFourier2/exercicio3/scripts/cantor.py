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
    return freqs, aft_shifted
#------------------------------------------------------------------------
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
    #for i in range(len(freq)):
    #    if abs(i-f_final)<=.01:
    #        break
    #
    return freq[1:int(N/2)], aft[1:int(N/2)]
#------------------------------------------------------------------------
def cantor ( n ) :
    return [ 0. ] + cant ( 0. , 1. , n ) + [ 1. ]
def cant ( x , y , n ) :
    if n == 0:
        return [ ]
    new_pts = [ 2. * x/3. + y/3. , x/3. + 2. * y / 3. ]
    return cant( x , new_pts[ 0 ] , n-1) + new_pts + cant( new_pts[ 1 ] , y , n-1)
seed=5
x = np.array ( cantor( seed ) )
y = np.cumsum( np.ones( len ( x ) ) / ( len (x)-2) ) - 1. / ( len (x)-2)
y[-1] = 1
#np.savetxt('cantor.dat', np.vstack( [ x , y ] ).T)
#sys.exit()
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
#                                ESPECTROS
#------------------------------------------------------------------------
seed=5
x = np.array ( cantor( seed ) )
y = np.cumsum( np.ones( len ( x ) ) / ( len (x)-2) ) - 1. / ( len (x)-2)
name = 'Cantor stair'
#chirp = g_1
t = np.array(x)
#res = res_1
res = abs(t[0]-t[-1])/len(t)

window_size = .5
xjump = 6
xround = 1
yamount = 8
yround = 1


tau = np.array(t)
f_final = 10
plot_title = 'Spectrogram of '+name+'\n'+'window size = '+str(window_size)+', seed = ' + str(seed)
fig_name = name+'_seed'+str(seed)+'_ws' + str(window_size)

spec_1 = []
spec_2 = []
spec_3 = []
spec_4 = []
spec_5 = []
spec_6 = []
for i in range(len(tau)):
    t_arr = t - tau[i]
    #------------------------------------
    w1 = rect(t_arr, window_size)
    x1, y1 = spec(y * w1, res, f_final)
    spec_1.append(y1)
    #------------------------------------
    w2 = hanning(t_arr, window_size)
    x2, y2 = spec(y * w2, res, f_final)
    spec_2.append(y2)
    #------------------------------------
    w3 = barllet(t_arr, window_size)
    x3, y3 = spec(y * w3, res, f_final)
    spec_3.append(y3)
    #------------------------------------
    w4 = papoulis(t_arr, window_size)
    x4, y4 = spec(y * w4, res, f_final)
    spec_4.append(y4)
    #------------------------------------
    w5 = hamming(t_arr, window_size)
    x5, y5 = spec(y * w5, res, f_final)
    spec_5.append(y5)
    #------------------------------------
    w6 = tukeywin(len(t_arr), window_size)
    x6, y6 = spec(y * w6, res, f_final)
    spec_6.append(y6)
    #------------------------------------
spec_1 = np.array(spec_1).T
spec_2 = np.array(spec_2).T
spec_3 = np.array(spec_3).T
spec_4 = np.array(spec_4).T
spec_5 = np.array(spec_5).T
spec_6 = np.array(spec_6).T
#
spec_1_p = 20*np.log10(np.abs(spec_1))
spec_2_p = 20*np.log10(np.abs(spec_2))
spec_3_p = 20*np.log10(np.abs(spec_3))
spec_4_p = 20*np.log10(np.abs(spec_4))
spec_5_p = 20*np.log10(np.abs(spec_5))
spec_6_p = 20*np.log10(np.abs(spec_6))
#========================================================================
#                                PLOTS
#------------------------------------------------------------------------
# ESPECTRO CANTOR STAIR
cant_freq, cant_psd = psd(y, res)
#
fig, ax = plt.subplots(1, 2, figsize=(8,4))
#
ax[0].set_title('Cantor stair - seed = ' + str(seed), size=16)
ax[0].plot(x, y)
ax[0].set_xlabel('t', size=14)
#
ax[1].set_title('Cantor Spectra', size=16)
ax[1].semilogx(cant_freq, cant_psd)
ax[1].set_xlabel('f', size=14)
#
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=.2, hspace=.4)
fig.savefig('Cantor_spectra_seed'+str(seed)+'.jpg', dpi=600, bbox_inches='tight')
plt.show(block=False)
plt.pause(.5)
plt.close()
#sys.exit()
#------------------------------------------------------------------------
fig, ax = plt.subplots(3, 2, figsize=(10,8))
my_cmap = matplotlib.cm.get_cmap('jet')
fig.suptitle(plot_title, size=16)
#
p_min = np.min([spec_1_p, spec_2_p, spec_3_p, spec_4_p, spec_5_p])
p_max = np.max([spec_1_p, spec_2_p, spec_3_p, spec_4_p, spec_5_p])
#
t_ticks = list(np.arange(0, len(t)+1, 1/res/xjump))
t_labels = list(np.round(t[::int(1/res/xjump)], xround))
#
f_ticks = np.arange(0,len(x1), int(len(x1)/(yamount-1)), dtype=int)
f_labels = list(np.round(x1[f_ticks], yround))
#
#
ax[0,0].set_title('Rectangular window', size = 13)
ax[0,0].contourf(spec_1_p, 256, origin='upper', cmap=my_cmap, vmin=p_min, vmax=p_max)#
ax[0,0].set_xticks(t_ticks)
ax[0,0].set_xticklabels(t_labels)
ax[0,0].set_yticks(f_ticks)
ax[0,0].set_yticklabels(f_labels)
ax[0,0].set_ylabel('frequency', size=12)
#
ax[1,0].set_title('Hanning window', size=13)
ax[1,0].contourf(spec_2_p, 256, origin='upper', cmap=my_cmap, vmin=p_min, vmax=p_max)#
ax[1,0].set_xticks(t_ticks)
ax[1,0].set_xticklabels(t_labels)
ax[1,0].set_yticks(f_ticks)
ax[1,0].set_yticklabels(f_labels)
ax[1,0].set_ylabel('frequency', size=12)
#
ax[2,0].set_title('Barllet window', size=13)
ax[2,0].contourf(spec_3_p, 256, origin='upper', cmap=my_cmap, vmin=p_min, vmax=p_max)#
ax[2,0].set_xticks(t_ticks)
ax[2,0].set_xticklabels(t_labels)
ax[2,0].set_yticks(f_ticks)
ax[2,0].set_yticklabels(f_labels)
ax[2,0].set_ylabel('frequency', size=12)
ax[2,0].set_xlabel('time', size=12)
#
ax[0,1].set_title('Papoulis window', size=13)
im=ax[0,1].contourf(spec_4_p, 256, origin='upper', cmap=my_cmap, vmin=p_min, vmax=p_max)#
ax[0,1].set_xticks(t_ticks)
ax[0,1].set_xticklabels(t_labels)
ax[0,1].set_yticks(f_ticks)
ax[0,1].set_yticklabels(f_labels)
#
ax[1,1].set_title('Hamming window', size=13)
ax[1,1].contourf(spec_5_p, 256, origin='upper', cmap=my_cmap, vmin=p_min, vmax=p_max)#
ax[1,1].set_xticks(t_ticks)
ax[1,1].set_xticklabels(t_labels)
ax[1,1].set_yticks(f_ticks)
ax[1,1].set_yticklabels(f_labels)
#
ax[2,1].set_title('Tukey window', size=13)
ax[2,1].contourf(spec_6_p, 256, origin='upper', cmap=my_cmap, vmin=p_min, vmax=p_max)#)
ax[2,1].set_xticks(t_ticks)
ax[2,1].set_xticklabels(t_labels)
ax[2,1].set_yticks(f_ticks)
ax[2,1].set_yticklabels(f_labels)
ax[2,1].set_xlabel('time', size=12)
#------------------------------------------------------------------------
fig.subplots_adjust(left=None, bottom=None, right=.85, top=None, wspace=.2, hspace=.4)
# Setting colorbar
cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
cb = fig.colorbar(im, cax=cbar_ax)
cb.set_label('Log of Power', fontsize=15)
#------------------------------------------------------------------------
fig.savefig(fig_name+'.jpg', dpi=600, bbox_inches='tight')
plt.show()
#------------------------------------------------------------------------ FIM
plt.close('all')
