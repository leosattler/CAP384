#========================================================================
#                                  IMPORTS
#------------------------------------------------------------------------
import numpy as np
import numpy.fft as fft
import matplotlib
import matplotlib.pyplot as plt
import gc
#import sys
pi = np.pi
#========================================================================
#                           FUNCOES AUXILIARES
#------------------------------------------------------------------------
def tratamento_jump(data, times_media=5):
    m = np.mean(data)
    data[np.where(data>times_media*m)] = m
#========================================================================
#                                  SINAIS
#------------------------------------------------------------------------
years = np.arange(1964, 2020)
#-------------------------------------
f1_file=np.genfromtxt('../../dados/dailyaveraged_data.txt')
f1_year = f1_file[1:, 0]
f1_doy = f1_file[1:, 1]
f1_data = f1_file[1:, 3]
tratamento_jump(f1_data, 5)
f1 = []
k=0
for y in years:
    c = 0
    for i in np.arange(0, len(f1_data)):
        if int(f1_year[i]) == int(y) and c<=365:
            d = f1_data[i]
            f1.append(d)
            c=c+1
        if c>=365:
            break
#-------------------------------------
f2_file = np.genfromtxt('../../dados/27dayaveraged_data.txt')
f2_year = f2_file[1:, 0]
f2_doy = f2_file[1:, 1]
f2_data = f2_file[1:, 3]
f2 = []
k=0
for y in years:
    c=0
    for i in np.arange(0, len(f2_data)):
        if int(f2_year[i]) == int(y) and c<=12:
            d = f2_data[i]
            f2.append(d)
            c=c+1
        if c>=12:
            break
#-------------------------------------
f3_file = np.genfromtxt('../../dados/yearlyaveraged_data.txt')
f3_year = f3_file[1:, 0]
f3_doy = f3_file[1:, 1]
f3_data = f3_file[1:, 3]
f3 = np.array(f3_data)[:-1]
#========================================================================
#                            FUNCOES AUXILIARES
#------------------------------------------------------------------------
def fourierT(signal, sr=1, N=2048):
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
def wft(signal, sr=1, N=2048):
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
    
    #for i in range(len(freq)):
    #    if abs(i-f_final)<=.01:
    #        break
    #[:i]
    return freq[1:int(N/2)], ft[1:int(N/2)]
#------------------------------------------------------------------------
def spec(signal, sr=1, N=2048):
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
#                           DATA + SPECTRA PLOT
#------------------------------------------------------------------------
# Truncando dados (ate a maior potencia de 2 possivel)
N1 = 2**14 
N2 = 512 
N3 = 32 

# A ALTERAR ############
name = 'F10.7 daily averages'
space = 2**4
N = N1
y = f1[:N:space]
res = space*1/365.
# ^^^^^^^^^ ############

N = int(N/space)
t = np.linspace(0, res*N, N)#np.arange(N)
y_freqs_trans, y_trans = fourierT(y, res, N)
y_freqs, y_psd = psd(y, res, N)

#------------------------------------------------------------------------ PLOTS

fig, ax = plt.subplots(1, 3, figsize=(17,3))

ax[0].set_title(name, size=14)
ax[0].plot(t, y)
ax[0].set_xlabel('time [years]', size=12)

ax[1].set_title('Fourier Transform of ' +name, size=14)
ax[1].semilogx(y_freqs_trans, y_trans/max(y_trans))
ax[1].set_xlabel('frequency', size=12)
ax[1].set_xlim(.02,1)
ax[1].set_ylim(-.1,.07)

ax[2].set_title('FT Zoom,'+r' f $\in [0.15, 1]$', size=14)
ax[2].semilogx(y_freqs_trans, y_trans/max(y_trans))
ax[2].set_xlabel('frequency', size=12)
ax[2].set_xlim(.15,1)
ax[2].set_ylim(-.05,.03)

fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=.22, hspace=None)
fig.savefig(name+'_wft'+'.png', dpi=600, bbox_inches='tight')
fig.savefig(name+'_wft'+'.eps', dpi=600, bbox_inches='tight')
plt.show(block=False)
plt.pause(.5)
plt.close()



#------------------------------------------------------------------------ 



fig, ax = plt.subplots(1, 3, figsize=(17,3))

ax[0].set_title(name, size=14)
ax[0].plot(t, y)
ax[0].set_xlabel('time [years]', size=12)

ax[1].set_title('Spectra of ' +name, size=14)
ax[1].semilogx(y_freqs, y_psd/max(y_psd))
ax[1].set_xlabel('frequency', size=12)
ax[1].set_xlim(.02,1)
ax[1].set_ylim(-.01,.07)

ax[2].set_title('Spectra Zoom,'+r' f $\in [0.15, 1]$', size=14)
ax[2].semilogx(y_freqs, y_psd/max(y_psd))
ax[2].set_xlabel('frequency', size=12)
ax[2].set_xlim(.15,1)
ax[2].set_ylim(-.0005,.003)

fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=.22, hspace=None)
fig.savefig(name+'_spectra'+'.png', dpi=600, bbox_inches='tight')
fig.savefig(name+'_spectra'+'.eps', dpi=600, bbox_inches='tight')
plt.show(block=False)
plt.pause(.5)
plt.close()

plt.close('all')
gc.collect()






#========================================================================
#                   RESULTS (PLOT + FILE) OF WFT 
#------------------------------------------------------------------------
window_size = .1
#xjump = 6
xamount = 7
yamount = 9
xround = 0
yround = 0

tau = np.array(t)
plot_title = 'WFT of '+name+'\n'+'window size = '+str(window_size)
fig_name = 'WFT_'+name+'_ws' + str(window_size)

window_size = window_size 
trans_1 = np.zeros([int(N/2)-1, len(tau)])
trans_2 = np.zeros([int(N/2)-1, len(tau)])
trans_3 = np.zeros([int(N/2)-1, len(tau)])
trans_4 = np.zeros([int(N/2)-1, len(tau)])
trans_5 = np.zeros([int(N/2)-1, len(tau)])
trans_6 = np.zeros([int(N/2)-1, len(tau)])
for i in range(len(tau)):
    print(str(round(100*i/len(tau),2))+'%')
    t_arr = t - tau[i]
    #------------------------------------
    w1 = rect(t_arr, window_size* 1/res)
    x1, y1 = wft(y * w1, res, N)
    trans_1[:,i]=y1#.append(y1)
    #------------------------------------
    w2 = hanning(t_arr, window_size* 1/res)
    x2, y2 = wft(y * w2, res, N)
    trans_2[:,i]=y2#.append(y2)
    #------------------------------------
    w3 = barllet(t_arr, window_size* 1/res)
    x3, y3 = wft(y * w3, res, N)
    trans_3[:,i]=y3#.append(y3)
    #------------------------------------
    w4 = papoulis(t_arr, window_size* 1/res)
    x4, y4 = wft(y * w4, res, N)
    trans_4[:,i]=y4#.append(y4)
    #------------------------------------
    w5 = hamming(t_arr, window_size* 1/res)
    x5, y5 = wft(y * w5, res, N)
    trans_5[:,i]=y5#.append(y5)
    #------------------------------------
    w6 = tukeywin(len(t_arr), window_size* 1/res)
    x6, y6 = wft(y * w6, res, N)
    trans_6[:,i]=y6#.append(y6)
    #------------------------------------
del w1, w2, w3, w4, w5, w6
del y1, y2, y3, y4, y5, y6
del x2, x3, x4, x5, x6
#trans_1 = np.array(trans_1).T
#trans_2 = np.array(trans_2).T
#trans_3 = np.array(trans_3).T
#trans_4 = np.array(trans_4).T
#trans_5 = np.array(trans_5).T
#trans_6 = np.array(trans_6).T
#
#trans_1_p = 20*np.log10(np.abs(trans_1))
#trans_2_p = 20*np.log10(np.abs(trans_2))
#trans_3_p = 20*np.log10(np.abs(trans_3))
#trans_4_p = 20*np.log10(np.abs(trans_4))
#trans_5_p = 20*np.log10(np.abs(trans_5))
#trans_6_p = 20*np.log10(np.abs(trans_6))

#------------------------------------------------------------------------ PLOTS
print('plotting...')
fig, ax = plt.subplots(3, 2, figsize=(10,8))
my_cmap = matplotlib.cm.get_cmap('jet')
fig.suptitle(plot_title, size=16)
#
print('finding max min...')
p_min = np.min(np.abs([trans_1, trans_2, trans_3, trans_4, trans_5]))
p_max = 1e3#np.max([trans_1, trans_2, trans_3, trans_4, trans_5])
print('max min found!...')
#
t_ticks = np.arange(0,len(tau), int(len(tau)/(xamount-1)), dtype=int)
t_labels = list(np.round(tau[t_ticks], xround))
#
f_ticks = np.arange(0,len(x1), int(len(x1)/(yamount-1)), dtype=int)
f_labels = list(np.round(x1[f_ticks], yround))
#
print('now i am plotting the first')
ax[0,0].set_title('Rectangular window', size = 13)
ax[0,0].contourf(trans_1, 256, origin='upper', cmap=my_cmap, vmin=p_min, vmax=p_max)
ax[0,0].set_xticks(t_ticks)
ax[0,0].set_xticklabels(t_labels)
ax[0,0].set_yticks(f_ticks)
ax[0,0].set_yticklabels(f_labels)
ax[0,0].set_ylabel('frequency [cycles/year]', size=12)
#
print('plotting second')
ax[1,0].set_title('Hanning window', size=13)
ax[1,0].contourf(trans_2, 256, origin='upper', cmap=my_cmap, vmin=p_min, vmax=p_max)
ax[1,0].set_xticks(t_ticks)
ax[1,0].set_xticklabels(t_labels)
ax[1,0].set_yticks(f_ticks)
ax[1,0].set_yticklabels(f_labels)
ax[1,0].set_ylabel('frequency [cycles/year]', size=12)
#
print('plotting 3rd')
ax[2,0].set_title('Barllet window', size=13)
ax[2,0].contourf(trans_3, 256, origin='upper', cmap=my_cmap, vmin=p_min, vmax=p_max)
ax[2,0].set_xticks(t_ticks)
ax[2,0].set_xticklabels(t_labels)
ax[2,0].set_yticks(f_ticks)
ax[2,0].set_yticklabels(f_labels)
ax[2,0].set_ylabel('frequency [cycles/year]', size=12)
ax[2,0].set_xlabel('time [years]', size=12)
#
print('plotting 4th')
ax[0,1].set_title('Papoulis window', size=13)
im = ax[0,1].contourf(trans_4, 256, origin='upper', cmap=my_cmap, vmin=p_min, vmax=p_max)
ax[0,1].set_xticks(t_ticks)
ax[0,1].set_xticklabels(t_labels)
ax[0,1].set_yticks(f_ticks)
ax[0,1].set_yticklabels(f_labels)
#
print('plotting 5th')
ax[1,1].set_title('Hamming window', size=13)
ax[1,1].contourf(trans_5, 256, origin='upper', cmap=my_cmap, vmin=p_min, vmax=p_max)
ax[1,1].set_xticks(t_ticks)
ax[1,1].set_xticklabels(t_labels)
ax[1,1].set_yticks(f_ticks)
ax[1,1].set_yticklabels(f_labels)
#
print('plotting 6th')
ax[2,1].set_title('Tukey window', size=13)
ax[2,1].contourf(trans_6, 256, origin='upper', cmap=my_cmap, vmin=p_min, vmax=p_max)
ax[2,1].set_xticks(t_ticks)
ax[2,1].set_xticklabels(t_labels)
ax[2,1].set_yticks(f_ticks)
ax[2,1].set_yticklabels(f_labels)
ax[2,1].set_xlabel('time [years]', size=12)
#------------------------------------------------------------------------
#cb.set_label('Power')
fig.subplots_adjust(left=None, bottom=None, right=.85, top=None, wspace=.2, hspace=.4)
# Setting colorbar
cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
cb = fig.colorbar(im, cax=cbar_ax)
cb.set_label('Log of Power', fontsize=15)
#
fig.savefig(fig_name+'.png', dpi=600, bbox_inches='tight')
fig.savefig(fig_name+'.eps', dpi=600, bbox_inches='tight')
print('WFT plots saved!')
print(' ')
#plt.show()
#------------------------------------------------------------------------
'''
file_name = 'OUT_wft_'+name.split(' ')[1]+'_ws'+str(window_size)+'_'
print('printing first wft file...')
np.savez(file_name, **{'rectangular':trans_1, 'hanning':trans_2, \
                       'barllet':trans_3, 'pappoulis':trans_4, \
                       'hamming':trans_5, 'tukey':trans_6})
#
print('printing second wft file...')
np.savez(file_name+'coordinates', **{'time':tau, 'frequency':x1})
'''



plt.close('all')
gc.collect()


































#========================================================================
#                   RESULTS (PLOT + FILE) OF SPECTROGRAM
#------------------------------------------------------------------------

tau = np.array(t)#np.linspace(0, res*N, N)
plot_title = 'Spectrogram of '+name+'\n'+'window size = '+str(window_size)
fig_name = 'Spectra_'+name+'_ws' + str(window_size)

spec_1 = np.zeros([int(N/2)-1, len(tau)])
spec_2 = np.zeros([int(N/2)-1, len(tau)])
spec_3 = np.zeros([int(N/2)-1, len(tau)])
spec_4 = np.zeros([int(N/2)-1, len(tau)])
spec_5 = np.zeros([int(N/2)-1, len(tau)])
spec_6 = np.zeros([int(N/2)-1, len(tau)])
for i in range(len(tau)):
    print(str(round(100*i/len(tau),2))+'%')
    t_arr = t - tau[i]
    #------------------------------------
    w1 = rect(t_arr, window_size* 1/res)
    x1, y1 = spec(y * w1, res, N)
    spec_1[:,i]=y1#.append(y1)
    #------------------------------------
    w2 = hanning(t_arr, window_size* 1/res)
    x2, y2 = spec(y * w2, res, N)
    spec_2[:,i]=y2#.append(y2)
    #------------------------------------
    w3 = barllet(t_arr, window_size* 1/res)
    x3, y3 = spec(y * w3, res, N)
    spec_3[:,i]=y3#.append(y3)
    #------------------------------------
    w4 = papoulis(t_arr, window_size* 1/res)
    x4, y4 = spec(y * w4, res, N)
    spec_4[:,i]=y4#.append(y4)
    #------------------------------------
    w5 = hamming(t_arr, window_size* 1/res)
    x5, y5 = spec(y * w5, res, N)
    spec_5[:,i]=y5#.append(y5)
    #------------------------------------
    w6 = tukeywin(len(t_arr), window_size* 1/res)
    x6, y6 = spec(y * w6, res, N)
    spec_6[:,i]=y6#.append(y6)
    #------------------------------------
del w1, w2, w3, w4, w5, w6
del y1, y2, y3, y4, y5, y6
del x2, x3, x4, x5, x6
#spec_1 = np.array(spec_1).T
#spec_2 = np.array(spec_2).T
#spec_3 = np.array(spec_3).T
#spec_4 = np.array(spec_4).T
#spec_5 = np.array(spec_5).T
#spec_6 = np.array(spec_6).T
#
spec_1_p = 20*np.log10(np.abs(spec_1))
spec_2_p = 20*np.log10(np.abs(spec_2))
spec_3_p = 20*np.log10(np.abs(spec_3))
spec_4_p = 20*np.log10(np.abs(spec_4))
spec_5_p = 20*np.log10(np.abs(spec_5))
spec_6_p = 20*np.log10(np.abs(spec_6))

#------------------------------------------------------------------------ PLOTS

fig, ax = plt.subplots(3, 2, figsize=(10,8))
my_cmap = matplotlib.cm.get_cmap('jet')
fig.suptitle(plot_title, size=16)
#
p_min = np.min([spec_1_p, spec_2_p, spec_3_p, spec_4_p, spec_5_p])
p_max = np.max([spec_1_p, spec_2_p, spec_3_p, spec_4_p, spec_5_p])
#
t_ticks = np.arange(0,len(tau), int(len(tau)/(xamount-1)), dtype=int)
t_labels = list(np.round(tau[t_ticks], xround))
#
f_ticks = np.arange(0,len(x1), int(len(x1)/(yamount-1)), dtype=int)
f_labels = list(np.round(x1[f_ticks], yround))
#
print('plotting 1')
ax[0,0].set_title('Rectangular window', size = 13)
ax[0,0].contourf(spec_1_p, 256, origin='upper', cmap=my_cmap, vmin=p_min, vmax=p_max)
ax[0,0].set_xticks(t_ticks)
ax[0,0].set_xticklabels(t_labels)
ax[0,0].set_yticks(f_ticks)
ax[0,0].set_yticklabels(f_labels)
ax[0,0].set_ylabel('frequency [cycles/year]', size=12)
#
print('plotting 2')
ax[1,0].set_title('Hanning window', size=13)
ax[1,0].contourf(spec_2_p, 256, origin='upper', cmap=my_cmap, vmin=p_min, vmax=p_max)
ax[1,0].set_xticks(t_ticks)
ax[1,0].set_xticklabels(t_labels)
ax[1,0].set_yticks(f_ticks)
ax[1,0].set_yticklabels(f_labels)
ax[1,0].set_ylabel('frequency [cycles/year]', size=12)
#
print('plotting 3')
ax[2,0].set_title('Barllet window', size=13)
ax[2,0].contourf(spec_3_p, 256, origin='upper', cmap=my_cmap, vmin=p_min, vmax=p_max)
ax[2,0].set_xticks(t_ticks)
ax[2,0].set_xticklabels(t_labels)
ax[2,0].set_yticks(f_ticks)
ax[2,0].set_yticklabels(f_labels)
ax[2,0].set_ylabel('frequency [cycles/year]', size=12)
ax[2,0].set_xlabel('time [years]', size=12)
#
print('plotting 4')
ax[0,1].set_title('Papoulis window', size=13)
im = ax[0,1].contourf(spec_4_p, 256, origin='upper', cmap=my_cmap, vmin=p_min, vmax=p_max)
ax[0,1].set_xticks(t_ticks)
ax[0,1].set_xticklabels(t_labels)
ax[0,1].set_yticks(f_ticks)
ax[0,1].set_yticklabels(f_labels)
#
print('plotting 5')
ax[1,1].set_title('Hamming window', size=13)
ax[1,1].contourf(spec_5_p, 256, origin='upper', cmap=my_cmap, vmin=p_min, vmax=p_max)
ax[1,1].set_xticks(t_ticks)
ax[1,1].set_xticklabels(t_labels)
ax[1,1].set_yticks(f_ticks)
ax[1,1].set_yticklabels(f_labels)
#
print('plotting 6')
ax[2,1].set_title('Tukey window', size=13)
ax[2,1].contourf(spec_6_p, 256, origin='upper', cmap=my_cmap, vmin=p_min, vmax=p_max)
ax[2,1].set_xticks(t_ticks)
ax[2,1].set_xticklabels(t_labels)
ax[2,1].set_yticks(f_ticks)
ax[2,1].set_yticklabels(f_labels)
ax[2,1].set_xlabel('time [years]', size=12)
#------------------------------------------------------------------------
#cb.set_label('Power')
fig.subplots_adjust(left=None, bottom=None, right=.85, top=None, wspace=.2, hspace=.4)
# Setting colorbar
cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
cb = fig.colorbar(im, cax=cbar_ax)
cb.set_label('Log of Power', fontsize=15)
#
fig.savefig(fig_name+'.png', dpi=600, bbox_inches='tight')
fig.savefig(fig_name+'.eps', dpi=600, bbox_inches='tight')
print('Spectra plots saved!')
#plt.show()
#------------------------------------------------------------------------
'''
file_name = 'OUT_spectrogram_'+name.split(' ')[1]+'_ws'+str(window_size)+'_'
print('printing first spectra file...')
np.savez(file_name, **{'rectangular':trans_1, 'hanning':trans_2, \
                       'barllet':trans_3, 'pappoulis':trans_4, \
                       'hamming':trans_5, 'tukey':trans_6})
#
print('printing second spectra file...')
np.savez(file_name+'coordinates', **{'time':tau, 'frequency':x1})
'''
#------------------------------------------------------------------------ 
plt.close('all')
