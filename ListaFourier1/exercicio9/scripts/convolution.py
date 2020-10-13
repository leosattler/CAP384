#==================================================================
import matplotlib.pyplot as plt
import numpy as np
#==================================================================
def funcao1(t):
    r = np.exp(-t/2)*1
    #
    r[np.where(t<0)] = 0
    #
    return r
#-----------------------------------------------------------------
def funcao2(t):
    r = np.zeros(len(t))
    #
    r[np.where(t>=0)] = t[np.where(t>=0)]/5.
    r[np.where(t<=5)] = t[np.where(t<=5)]/5.
    r[np.where(t>=5)] = 0
    #
    return r
#-----------------------------------------------------------------
def conv_func(t):
    r = np.zeros(len(t))
    #
    indx_1 = np.where(np.logical_and(t>0, t<5))
    indx_2 = np.where(t>=5)
    #
    r[indx_1] = (2/5.) * t[indx_1] - (4/5.) * (1 - np.exp(-t[indx_1]/2.))
    r[indx_2] = (6/5.)*np.exp(-(t[indx_2]-5)/2.) + (4/5.)*np.exp(-t[indx_2]/2.)
    #
    return r
#==================================================================
t_final = 20
res = .01
t = np.arange(-10, t_final, res)
t_plots = [-1, 2, 4, 8, 14]
#
f1 = funcao1(t)
f2 = funcao2(t)
#==================================================================
#
xlims = [-.5,10]
ylims = [0,1.2]
#
plt.subplot(3,1,1)
plt.plot(t, f1, 'k-', linewidth=1.)
plt.ylabel(r'$x(t)$', size=14, rotation=0, labelpad=35.)
plt.xlim(xlims)
plt.ylim(ylims)
#
plt.subplot(3,1,2)
plt.plot(t, f2, 'k-', linewidth=1.)
plt.ylabel(r'$y(t)$', size=14, rotation=0, labelpad=35.)
plt.xlabel('t', size=14)
plt.xlim(xlims)
plt.ylim(ylims)
#
plt.subplot(3,1,3)
plt.plot(t, conv_func(t), 'k-', linewidth=1.)
plt.ylabel(r'$z(t)$', size=14, rotation=0, labelpad=35.)
plt.xlabel('t', size=14)
plt.xlim([-.5,16])
plt.ylim([0,1.5])
#
plt.savefig('x_y.jpg', dpi=400, bbox_inches='tight')
plt.show()
#==================================================================
fig, ax = plt.subplots(int(len(t_plots)), 2, figsize=(9,7))
#
xlims2 = [-10,14]
ylims2 = [0,1.5]
#
conv = np.zeros(len(f1))
t_local = np.array(t)
for i in range(len(t_plots)):
    t_conv = t_plots[i]
    f2 = funcao2(t_conv - t)
    #
    ax[i, 0].plot(t, f1, 'k--', linewidth=1.)
    ax[i, 0].plot(t, f2, 'k-', linewidth=1., label = 't = '+str(int(t_conv)))
    ax[i, 0].fill_between(t, np.min([f1,f2],axis=0), color='gray')
    ax[i, 0].set_xlim(xlims2)
    ax[i, 0].set_ylim(ylims)
    ax[i, 0].legend(loc=0, fontsize=12)
    #
    if t_conv >= 0:
        t_local = np.arange(0,t_conv,res)
        conv = conv_func(t_local)
    ax[i, 1].fill_between(t_local, conv, color='k', label = 't = '+str(int(t_conv)))
    ax[i, 1].set_xlim(xlims2)
    ax[i, 1].set_ylim(ylims2)
    ax[i, 1].legend(loc=0, fontsize=12)
    #
#
ax[0, 0].set_title(r'$y(t - \tau), x(\tau)$', size=15)
ax[0, 1].set_title(r'$z(t) = x(t) \star y(t)$', size=15)
#
ax[i, 0].set_xlabel(r'$\tau$', size=14)
ax[i, 1].set_xlabel(r'$\tau$', size=14)
#
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=.2, hspace=.4)
fig.savefig('z.jpg', dpi=400, bbox_inches='tight')
plt.show()
