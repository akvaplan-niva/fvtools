import pylab as plt
import numpy as np
from datetime import date,datetime,timedelta
import matplotlib.dates as mdates
import glob
import sys

from read_data import read_tag

def last_11chars(x):
    return(x[-11:])

#Path to files
path='C:\\Users\\MAD\\work\\Ferskvann_lus\\Seal_tags\\'
filesM3hamster = sorted(glob.glob(path+'?????_debug_M3hamster*gps.txt'),key = last_11chars)


for fn in filesM3hamster:
	depthstr = fn[-11:-8] #OBS: doesn't seem to fit P (assuming surf =10)
	tagno = fn[-33:-28]
	time,P,T,S = read_tag(fn)
	
	print "mean P = "+str(sum([float(i) for i in P])/len(P)) #mean P to compare with depthstr
	
	fig,ax1 = plt.subplots(figsize=(18,10))
	ax1.plot(time,T,'b-',linewidth=1.5)
	ax1.set_xlabel('Tid/Dato',fontsize=15)
	ax1.set_ylabel(r'Temp. [$^\circ \mathrm{C}$]', color='b',fontsize=15)
	ax1.tick_params('y',colors='b',labelsize=15)
	ax1.tick_params('x',labelsize=15)
	
	ax2=ax1.twinx()
	ax2.plot(time,S,'k-',linewidth=1.5)
	ax2.set_ylabel('Salinitet [PSU]', color='k',fontsize=15)
	ax2.tick_params('y', colors='k',labelsize=15)
	
	fig.tight_layout()
	plt.title('Tidsvariasjon v/M3-hamster (tag no. '+tagno+'), dyp='+str(int(depthstr))+' cm',fontsize=20)
	plt.savefig('Tag_'+tagno+'_M3H_'+depthstr+'cm.png', bbox_inches='tight')
	#plt.show()
	
filesM3ring = sorted(glob.glob(path+'?????_debug_M3ring*gps.txt'),key = last_11chars)
for fn in filesM3ring:
	depthstr = fn[-11:-8] #OBS: doesn't seem to fit P (assuming surf =10)
	tagno = fn[-30:-25]
	time,P,T,S = read_tag(fn)
	
	print "mean P = "+str(sum([float(i) for i in P])/len(P)) #mean P to compare with depthstr
	
	fig,ax1 = plt.subplots(figsize=(18,10))
	ax1.plot(time,T,'b-',linewidth=1.5)
	ax1.set_xlabel('Tid/Dato',fontsize=15)
	ax1.set_ylabel(r'Temp. [$^\circ \mathrm{C}$]', color='b',fontsize=15)
	ax1.tick_params('y',colors='b',labelsize=15)
	ax1.tick_params('x',labelsize=15)
	
	ax2=ax1.twinx()
	ax2.plot(time,S,'k-',linewidth=1.5)
	ax2.set_ylabel('Salinitet [PSU]', color='k',fontsize=15)
	ax2.tick_params('y', colors='k',labelsize=15)
	
	fig.tight_layout()
	plt.title('Tidsvariasjon v/M3-ring (tag no. '+tagno+'), dyp='+str(int(depthstr))+' cm',fontsize=20)
	plt.savefig('Tag_'+tagno+'_M3R_'+depthstr+'cm.png', bbox_inches='tight')
	plt.show()

filesM6 = sorted(glob.glob(path+'?????_debug_M6*gps.txt'),key = last_11chars)
for fn in filesM6:
	depthstr = fn[-11:-8] #OBS: doesn't seem to fit P (assuming surf =10)
	tagno = fn[-26:-21]
	time,P,T,S = read_tag(fn)
	
	print "mean P = "+str(sum([float(i) for i in P])/len(P)) #mean P to compare with depthstr
	
	fig,ax1 = plt.subplots(figsize=(18,10))
	ax1.plot(time,T,'b-',linewidth=1.5)
	ax1.set_xlabel('Tid/Dato',fontsize=15)
	ax1.set_ylabel(r'Temp. [$^\circ \mathrm{C}$]', color='b',fontsize=15)
	ax1.tick_params('y',colors='b',labelsize=15)
	ax1.tick_params('x',labelsize=15)
	
	ax2=ax1.twinx()
	ax2.plot(time,S,'k-',linewidth=1.5)
	ax2.set_ylabel('Salinitet [PSU]', color='k',fontsize=15)
	ax2.tick_params('y', colors='k',labelsize=15)
	
	fig.tight_layout()
	plt.title('Tidsvariasjon v/M6: (tag no. '+tagno+'), dyp='+str(int(depthstr))+' cm',fontsize=20)
	plt.savefig('Tag_'+tagno+'_M6R_'+depthstr+'cm.png', bbox_inches='tight')
	plt.show()
