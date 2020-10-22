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
    for i in range(len(freq)):
        if abs(i-f_final)<=.01:
            break
    #
    return freq[:i], aft[:i]
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
def rect(window_size, loc, array_size, sr):
    #-----------------------------------
    # Inputs:
    # (float) window_size: size of window in physical units
    # (int) loc: index to locate center of window in output array
    # (int) array_size: size of output array
    # (float) sr: sampling rate; used to scale window size from physical
    #             units to index values
    # Outputs:
    # (float array) r: array of zeros with a rectangular window
    #                  centered at loc
    #-----------------------------------
    # Scaling window basde on sampling rate
    window_size_SCALED = int((window_size)/sr)
    # Return array
    r = np.zeros(array_size)
    # Time array (input of window function)
    t = np.linspace(-1/2., 1/2., window_size_SCALED)
    # Window function array
    w = np.zeros(window_size_SCALED)
    # Calculating window function value
    w[np.where(abs(t) <= window_size_SCALED)] = 1
    # Factor to match size of window location inside r and window_size
    even_odd_w = window_size_SCALED%2
    # If center of window too close to initial border
    if loc < window_size_SCALED/2.:
        # Truncating window
        w = w[int(window_size_SCALED/2 -loc):]
        # Changing r values
        r[0:(len(w))] = w
    # If center of window too close to final border
    elif loc > array_size-window_size_SCALED/2.:
        # Truncating window
        w = w[0:int(array_size-(loc-window_size_SCALED/2.))]
        # Creating aux. variables to calculate index location
        r_init = int(loc-window_size_SCALED/2.)
        test = abs(array_size-r_init)
        if test != len(w):
            # Fixing index
            r_init = array_size - len(w)
        # Changing r values
        r[r_init:] = w
    # If center of window is completely inside array r
    else:
        # Changing r values
        r[(loc-int(window_size_SCALED/2.)):(loc+int(window_size_SCALED/2.)+even_odd_w)] = w
    #
    return r
#------------------------------------------------------------------------
def hanning(window_size, loc, array_size, sr):
    window_size_SCALED = int(window_size/sr)
    r = np.zeros(array_size)
    t = np.linspace(-1/2., 1/2., window_size_SCALED)
    w = 1/2. + (1/2.) * np.cos(2 * pi * t)
    even_odd_w = window_size_SCALED%2
    if loc < window_size_SCALED/2.:
        w = w[int(window_size_SCALED/2 -loc):]
        r[0:(len(w))] = w
    elif loc > array_size-window_size_SCALED/2.:
        w = w[0:int(array_size-(loc-window_size_SCALED/2.))]
        r_init = int(loc-window_size_SCALED/2.)
        test = abs(array_size-r_init)
        if test != len(w):
            r_init = array_size - len(w)
        r[r_init:] = w
    else:
        r[(loc-int(window_size_SCALED/2.)):(loc+int(window_size_SCALED/2.)+even_odd_w)] = w
    return r
#------------------------------------------------------------------------
def hamming(window_size, loc, array_size, sr):
    window_size_SCALED = int(window_size/sr)
    r = np.zeros(array_size)
    t = np.linspace(-1/2., 1/2., window_size_SCALED)
    w = .54 + .46 * np.cos(2 * pi * t)
    even_odd_w = window_size_SCALED%2
    if loc < window_size_SCALED/2.:
        w = w[int(window_size_SCALED/2 -loc):]
        r[0:(len(w))] = w
    elif loc > array_size-window_size_SCALED/2.:
        w = w[0:int(array_size-(loc-window_size_SCALED/2.))]
        r_init = int(loc-window_size_SCALED/2.)
        test = abs(array_size-r_init)
        if test != len(w):
            r_init = array_size - len(w)
        r[r_init:] = w
    else:
        r[(loc-int(window_size_SCALED/2.)):(loc+int(window_size_SCALED/2.)+even_odd_w)] = w
    return r
#------------------------------------------------------------------------
def bartlett(window_size, loc, array_size, sr):
    window_size_SCALED = int(window_size/sr)
    r = np.zeros(array_size)
    t = np.linspace(-1/2., 1/2., window_size_SCALED)
    w = 1 - 2 * np.abs(t[np.where(abs(t) <= window_size_SCALED)])
    even_odd_w = window_size_SCALED%2
    if loc < window_size_SCALED/2.:
        w = w[int(window_size_SCALED/2 -loc):]
        r[0:(len(w))] = w
    elif loc > array_size-window_size_SCALED/2.:
        w = w[0:int(array_size-(loc-window_size_SCALED/2.))]
        r_init = int(loc-window_size_SCALED/2.)
        test = abs(array_size-r_init)
        if test != len(w):
            r_init = array_size - len(w)
        r[r_init:] = w
    else:
        r[(loc-int(window_size_SCALED/2.)):(loc+int(window_size_SCALED/2.)+even_odd_w)] = w
    return r
#------------------------------------------------------------------------
def papoulis(window_size, loc, array_size, sr):
    window_size_SCALED = int(window_size/sr)
    r = np.zeros(array_size)
    t = np.linspace(-1/2., 1/2., window_size_SCALED)
    w = (1/pi) * np.abs(np.sin(2*pi*t[np.where(abs(t) <= window_size_SCALED)])) + \
                                         (1 - 2 * np.abs(t[np.where(abs(t) <= window_size_SCALED)])) * \
                                         np.cos(2*pi*t[np.where(abs(t) <= window_size_SCALED)])
    even_odd_w = window_size_SCALED%2
    if loc < window_size_SCALED/2.:
        w = w[int(window_size_SCALED/2 -loc):]
        r[0:(len(w))] = w
    elif loc > array_size-window_size_SCALED/2.:
        w = w[0:int(array_size-(loc-window_size_SCALED/2.))]
        r_init = int(loc-window_size_SCALED/2.)
        test = abs(array_size-r_init)
        if test != len(w):
            r_init = array_size - len(w)
        r[r_init:] = w
    else:
        r[(loc-int(window_size_SCALED/2.)):(loc+int(window_size_SCALED/2.)+even_odd_w)] = w
    return r
#------------------------------------------------------------------------
def tukeywin(window_length, alpha, loc, array_size, sr):
    '''The Tukey window, also known as the tapered cosine window, can be regarded as a cosine lobe of width \alpha * N / 2
    that is convolved with a rectangle window of width (1 - \alpha / 2). At \alpha = 1 it becomes rectangular, and
    at \alpha = 0 it becomes a Hann window.

    We use the same reference as MATLAB to provide the same results in case users compare a MATLAB output to this function
    output

    Reference
    ---------
    http://www.mathworks.com/access/helpdesk/help/toolbox/signal/tukeywin.html
    '''
    window_length_SCALED = int(window_length/sr)
    r = np.zeros(array_size)
    # Special cases
    if alpha <= 0:
        return np.ones(window_length_SCALED) #rectangular window
    elif alpha >= 1:
        return np.hanning(window_length_SCALED)
    # Normal case
    x = np.linspace(0, 1, window_length_SCALED)
    w = np.ones(x.shape)
    # first condition 0 <= x < alpha/2
    first_condition = x<alpha/2
    w[first_condition] = 0.5 * (1 + np.cos(2*np.pi/alpha * (x[first_condition] - alpha/2) ))
    # second condition already taken care of
    # third condition 1 - alpha / 2 <= x <= 1
    third_condition = x>=(1 - alpha/2)
    w[third_condition] = 0.5 * (1 + np.cos(2*np.pi/alpha * (x[third_condition] - 1 + alpha/2)))
    even_odd_w = window_length_SCALED%2
    if loc < window_length_SCALED/2.:
        w = w[int(window_length_SCALED/2 -loc):]
        r[0:(len(w))] = w
    elif loc > array_size-window_length_SCALED/2.:
        w = w[0:int(array_size-(loc-window_length_SCALED/2.))]
        r_init = int(loc-window_length_SCALED/2.)
        test = abs(array_size-r_init)
        if test != len(w):
            r_init = array_size - len(w)
        r[r_init:] = w
    else:
        r[(loc-int(window_length_SCALED/2.)):(loc+int(window_length_SCALED/2.)+even_odd_w)] = w
    return r
#========================================================================
#                                ESPECTROS
#------------------------------------------------------------------------
seed=5
x = np.array ( cantor( seed ) )
y = np.cumsum( np.ones( len ( x ) ) / ( len (x)-2) ) - 1. / ( len (x)-2)
cantor_f = np.array(x)
name = 'Cantor stair'


t = np.array(x)
res = abs(t[0]-t[-1])/len(t)
array_size = len(x)
window_size = .99

xamount = 8
xround = 0
yamount = 6
yround = 2


tau = np.array(t)
f_final = 30
plot_title = 'Spectrogram of '+name+'\n'+'window size = '+str(window_size)+', seed = ' + str(seed)
fig_name = name+'_seed'+str(seed)+'_ws' + str(window_size)+'_fFinal'+str(f_final)

spec_1 = []
spec_2 = []
spec_3 = []
spec_4 = []
spec_5 = []
spec_6 = []
# Defining spectrogram arrays
spec_1 = np.zeros([f_final, len(tau)])
spec_2 = np.zeros([f_final, len(tau)])
spec_3 = np.zeros([f_final, len(tau)])
spec_4 = np.zeros([f_final, len(tau)])
spec_5 = np.zeros([f_final, len(tau)])
spec_6 = np.zeros([f_final, len(tau)])
for i in range(len(tau)):
    # Printing progress
    print(str(round(100*i/len(y),2))+'%')
    # Center of window
    loc = i
    #------------------------------------ rectangular
    # Creating window array
    w1 = rect(window_size, loc, array_size, res)
    # Windowed spectra centered at loc
    x1, y1 = spec(cantor_f * w1, res, f_final)
    # Updating spectrogram array
    spec_1[:,i]=y1
    #------------------------------------ hanning
    w2 = hanning(window_size, loc, array_size, res)
    x2, y2 = spec(cantor_f * w2, res, f_final)
    spec_2[:,i]=y2
    #------------------------------------ barllet
    w3 = bartlett(window_size, loc, array_size, res)
    x3, y3 = spec(cantor_f * w3, res, f_final)
    spec_3[:,i]=y3
    #------------------------------------ papoulis
    w4 = papoulis(window_size, loc, array_size, res)
    x4, y4 = spec(cantor_f * w4, res, f_final)
    spec_4[:,i]=y4
    #------------------------------------ hamming
    w5 = hamming(window_size, loc, array_size, res)
    x5, y5 = spec(cantor_f * w5, res, f_final)
    spec_5[:,i]=y5
    #------------------------------------ tukey
    w6 = tukeywin(window_size, 0.5, loc, array_size, res)
    x6, y6 = spec(cantor_f * w6, res, f_final)
    spec_6[:,i]=y6
    #------------------------------------
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
cant_freq, cant_psd = psd(cantor_f, res)
#
fig, ax = plt.subplots(1, 2, figsize=(8,3))
#
ax[0].set_title('Cantor stair - seed = ' + str(seed), size=12)
ax[0].plot(np.linspace(0,1,len(x)), cantor_f)#ax[0].plot(x, y)
ax[0].set_xlabel('t', size=10)
#
ax[1].set_title('Cantor Spectra', size=12)
ax[1].semilogx(cant_freq, cant_psd)
ax[1].set_xlabel('f', size=10)
ax[1].set_xlim(1, 70)

ax[1].set_ylim(-10,500)
#
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=.2, hspace=.4)
fig.savefig('Cantor_spectra_seed'+str(seed)+'.jpg', dpi=600, bbox_inches='tight')
#plt.show(block=False)
#plt.pause(.5)
#plt.show()
plt.close()
#sys.exit()

#------------------------------------------------------------------------
fig, ax = plt.subplots(3, 2, figsize=(10,8))
my_cmap = matplotlib.cm.get_cmap('jet')
fig.suptitle(plot_title, size=16)
#
p_min = np.min([spec_1_p, spec_2_p, spec_3_p, spec_4_p, spec_5_p])
p_max = np.max([spec_1_p, spec_2_p, spec_3_p, spec_4_p, spec_5_p])
# x ticks and labels
t=np.linspace(0,1,len(cantor_f))
t_ticks = np.linspace(0, len(cantor_f)-1, xamount, dtype=int)
t_labels = list(np.round(t[t_ticks], yround))
t_labels[t_labels.index(0)]=0
# y ticks and labels
f_ticks = np.linspace(0, len(x1)-1, yamount, dtype=int)
f_labels = list(np.round(x1[f_ticks], yround))
#
ax[0,0].set_title('Rectangular window', size = 13)
ax[0,0].contourf(spec_1_p, 256, origin='lower', cmap=my_cmap, vmin=p_min, vmax=p_max)
ax[0,0].set_ylabel('frequency', size=12)
#
ax[1,0].set_title('Hanning window', size=13)
ax[1,0].contourf(spec_2_p, 256, origin='lower', cmap=my_cmap, vmin=p_min, vmax=p_max)
ax[1,0].set_ylabel('frequency', size=12)
#
ax[2,0].set_title('Bartlett window', size=13)
ax[2,0].contourf(spec_3_p, 256, origin='lower', cmap=my_cmap, vmin=p_min, vmax=p_max)
ax[2,0].set_ylabel('frequency', size=12)
ax[2,0].set_xlabel('time', size=12)
#
ax[0,1].set_title('Papoulis window', size=13)
im=ax[0,1].contourf(spec_4_p, 256, origin='lower', cmap=my_cmap, vmin=p_min, vmax=p_max)
#
ax[1,1].set_title('Hamming window', size=13)
ax[1,1].contourf(spec_5_p, 256, origin='lower', cmap=my_cmap, vmin=p_min, vmax=p_max)
ax[1,1].set_yticklabels(f_labels)
#
ax[2,1].set_title('Tukey window', size=13)
ax[2,1].contourf(spec_6_p, 256, origin='lower', cmap=my_cmap, vmin=p_min, vmax=p_max)
ax[2,1].set_yticklabels(f_labels)
ax[2,1].set_xlabel('time', size=12)
#------------------------------------------------------------------------
fig.subplots_adjust(left=None, bottom=None, right=.85, top=None, wspace=.2, hspace=.4)
for i in range(3):
    # Setting same x ticks and tick labels for all plots
    ax[i,0].set_xticks(t_ticks)
    ax[i,0].set_xticklabels(t_labels)
    ax[i,1].set_xticks(t_ticks)
    ax[i,1].set_xticklabels(t_labels)
    #
    ax[i,0].set_yticks(f_ticks)
    ax[i,0].set_yticklabels(f_labels)
    ax[i,1].set_yticks(f_ticks)
    ax[i,1].set_yticklabels(f_labels)
# Setting colorbar
cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
cb = fig.colorbar(im, cax=cbar_ax)
cb.set_label('Log of Power', fontsize=15)
#------------------------------------------------------------------------
fig.savefig(fig_name+'.jpg', dpi=600, bbox_inches='tight')
plt.show()
#------------------------------------------------------------------------ FIM
plt.close('all')




