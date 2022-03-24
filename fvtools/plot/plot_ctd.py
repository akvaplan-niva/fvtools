import matplotlib.pyplot as plt
import glob
import numpy as np
# import os
import sys

from read_data import read_ctd_manyfiles
from datetime import datetime
from datetime import timedelta

files = sorted(glob.glob('C:\\Users\\MAD\\work\\Ferskvann_lus\\Converted_2018????\\Conv_2018????.txt'))
last_prof = datetime(2018, 6, 12, 0, 0, 0)

# Collect all profiles and collect unique in one renumbered ser.
ser, meas, S, T, Opt, Dens, P, dtime = read_ctd_manyfiles(files, last_prof)

# run through profiles, identify according to list and plot

# delta = dtime[-1]-dtime[0]
# startt = dtime[0].date
# endt = dtime[-1].date
# for n in range(delta.days):
#	trackt = startt() + timedelta(days=n)
#	dayt = [dtime[i] for i in range(len(dtime)) if dtime[i].date() == trackt]
#	sert = [ser[i] for i in range(len(dtime)) if dtime[i].date() == trackt]

# Plot all, loop through profiles
for n in np.unique(map(int, ser)):
    print 'plotting CTD-profile ' + str(n) + ' of ' + str(len(np.unique(map(int, ser))))
    idx = np.where(np.asarray(ser) == str(n))
    tt = [dtime[x] for x in idx[0][:] if float(P[x]) > 0.05 and float(Dens[x]) > 0.0]
    ser_tt = [ser[x] for x in idx[0][:] if float(P[x]) > 0.05 and float(Dens[x]) > 0.0]
    meas_tt = [meas[x] for x in idx[0][:] if float(P[x]) > 0.05 and float(Dens[x]) > 0.0]
    S_tt = np.asarray([float(S[x]) for x in idx[0][:] if float(P[x]) > 0.05 and float(Dens[x]) > 0.0])
    T_tt = np.asarray([float(T[x]) for x in idx[0][:] if float(P[x]) > 0.05 and float(Dens[x]) > 0.0])
    Opt_tt = np.asarray([float(Opt[x]) for x in idx[0][:] if float(P[x]) > 0.05 and float(Dens[x]) > 0.0])
    Dens_tt = np.asarray([float(Dens[x]) for x in idx[0][:] if float(P[x]) > 0.05 and float(Dens[x]) > 0.0])
    P_tt = np.asarray([float(P[x]) for x in idx[0][:] if float(P[x]) > 0.05 and float(Dens[x]) > 0.0])

    # local time
    lt = [i + timedelta(hours=2) for i in tt]

    # bottom
    if len(tt) == 0:
        continue

    id_b = np.where(np.asarray(P_tt) == max(P_tt))[0][0]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(18, 10))

    ax1.plot(T_tt[id_b:], P_tt[id_b:], 'r-o', linewidth=1.5)
    plt.ylim(0, 5)
    ax1.yaxis.set_ticks(np.arange(0, 5.25, 0.25))
    ax1.set_ylabel('Dyp [m]', fontsize=15)
    ax1.set_xlabel(r'Temperatur [$^\circ \mathrm{C}$]', fontsize=15)
    plt.gca().invert_yaxis()  # inverts all since sharey

    ax2.plot(S_tt[id_b:], P_tt[id_b:], 'b-o', linewidth=1.5)
    ax2.set_xlabel('Saltholdighet [PSU]', fontsize=15)
    ax3.plot(Dens_tt[id_b:], P_tt[id_b:], 'k-o', linewidth=1.5)
    ax3.set_xlabel(r'Tetthet [$\sigma_T$]', fontsize=15)

    # fig.tight_layout()
    plt.suptitle('CTD-profil ' + str(n) + ' av ' + str(len(np.unique(ser))) + '; ' + str(lt[0]) + ' (UTC+2)',
                 fontsize=15)
    plt.savefig('CTD-profil' + str(n) + '_' + lt[0].strftime('%Y%m%d_%H-%M') + '.png', bbox_inches='tight')
    plt.close()
