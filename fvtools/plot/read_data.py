from datetime import datetime
import numpy as np

def read_tag(filename):
	#Read seal-ctd-tags used in e.g. Ferskvann mot lus
        print "reading file "+filename
	f = open(filename,'r')
	dat = f.read().splitlines()
	f.close()
	dat2 = [dat[x].split("\t") for x in range(len(dat))]
	#Time and date in datetime; Only use lines with 12 entries as error check (if test)!
	tdate = [datetime.strptime('-'.join(dat2[x][0:2]),'%Y/%m/%d-%H:%M:%S') for x in range(len(dat2)) if len(dat2[x]) == 12]
	Preal = [dat2[x][5] for x in range(len(dat2)) if len(dat2[x]) == 12]
	Treal = [dat2[x][7] for x in range(len(dat2)) if len(dat2[x]) == 12]
	Sreal = [dat2[x][9] for x in range(len(dat2)) if len(dat2[x]) == 12]
	return tdate,Preal,Treal,Sreal
	
def read_ctd(filename):
	#Read one CTD txt-file and return variables in lists 
	#Usage: ser_t,meas_t,S_t,T_t,Opt_t,Dens_t,P_t,dtime_t = read_ctd(f)
	print "reading "+filename
	lookfor = "Ser;Meas;Sal.;Temp;Opt;Density;Press;Date;Time;;"
	f = open(filename,'r')
	dat = f.read().splitlines()
	f.close()
	startfrom=dat.index(lookfor)+1
	dat2 = [dat[startfrom:][x].split(";") for x in range(len(dat[startfrom:]))]
	tdate = [datetime.strptime('-'.join(dat2[x][-2:]),'%d.%m.%Y-%H.%M.%S') for x in range(len(dat2))]
	P = [dat2[x][6] for x in range(len(dat2))]
	Dens = [dat2[x][5] for x in range(len(dat2))]
	Opt = [dat2[x][4] for x in range(len(dat2))]
	T = [dat2[x][3] for x in range(len(dat2))]
	S = [dat2[x][2] for x in range(len(dat2))]
	Meas = [dat2[x][1] for x in range(len(dat2))]
	Ser = [dat2[x][0] for x in range(len(dat2))]
	return Ser,Meas,S,T,Opt,Dens,P,tdate
	
def read_ctd_manyfiles(filenames, start_date):
	#Returns one list with all (unique in time) profiles (renumbered 1-end)
	#Start date must be before first profile you want to include
	#Usage: ser,meas,S,T,Opt,Dens,P,dtime = read_ctd_manyfiles(files,last_prof)
	print "reading unique profiles from "+str(len(filenames))+" files"
	ser = []
	meas = []
	S = []
	T = []
	Opt = []
	Dens = []
	P = []
	dtime = []
	last_prof = start_date
	ser_no = 0
	for f in filenames:
		ser_t,meas_t,S_t,T_t,Opt_t,Dens_t,P_t,dtime_t = read_ctd(f)
		#Loop over profiles in that file (same ser)
		for n in np.unique(map(int,ser_t)):
			idx = np.where(np.asarray(ser_t) == str(n))
			tt = [dtime_t[x] for x in idx[0][:]] #time of profile
			#Only extend if profile is unique and not repeted from last file
			if tt[0] > last_prof:
				ser_no += 1
				print str(tt[0])+' > '+str(last_prof)+' -> new profile no: '+str(ser_no)
				last_prof = tt[0]
				#lists of profile values
				ser_tt = [str(ser_no)]*len(idx[0][:])
				meas_tt = [meas_t[x] for x in idx[0][:]]
				S_tt = [S_t[x].replace(",",".") for x in idx[0][:]]
				T_tt = [T_t[x].replace(",",".") for x in idx[0][:]]
				Opt_tt = [Opt_t[x].replace(",",".") for x in idx[0][:]]
				Dens_tt = [Dens_t[x].replace(",",".") for x in idx[0][:]]
				P_tt = [P_t[x].replace(",",".") for x in idx[0][:]]
			
				#extend profile by profile
				ser.extend(ser_tt); meas.extend(meas_tt); S.extend(S_tt); T.extend(T_tt)
				Opt.extend(Opt_tt); Dens.extend(Dens_tt); P.extend(P_tt); dtime.extend(tt)
	return ser,meas,S,T,Opt,Dens,P,dtime
