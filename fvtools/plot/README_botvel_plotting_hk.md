# Do botvel analysis:
run get_botvel_new.py (main)
- gives out a velocity file.

import plot botvel as pb
vd=pb.botvel_object(grdfile='case_out_0001.nc',fname='velocities_mars_toppsund.npy')
#Gridfile can not be a M.npy file. The script cannot handle this yet.

#make botvel figures:
Auto-plot a bunch of figures:
pb.botvel_analysis(vd) (Rarely used)
#plot 95% percentile with fish farms marked:
pb.plot_altnernatives(vd)
- this assumes you have a georef file. Usually I get this from havstraum.no. Share ->download georeferenced jpeg

-Here there are a lot of options to make the figure more pretty:

Add center points for fish-farm:
-Use provided excel sheet (edit filetype from .xlsx to .xls)

from fabm.setup.read_excel_positions import load_positions
x=load_positions('your_excel_sheet.xls')
#first set of coordinates is x[0,0][0], second set is x[0,1][0] etc
for centerpoints:
X1=x[0,0][0].T
X1=np.transpose([X1[1,:],X1[0,:]])
np.savetxt('lok1.txt',X1,delimiter=' \t', header='longitude latitude')
#For cage: (add point 1 to end to close rectangle)
X2=x[0,1][0].T
X2_=np.zeros([5,2])
X2_[0:-1,:]=X2
X2_[-1,:]=X2[0,:]
np.savetxt('lok1_frame.txt',X2_,delimiter=' \t', header='longitude latitude')

#To center the plot around the fish-farm:
pos = pb.read_posfile('lok1.txt', plot=False, clr='k')
center_y=vd.ymid[0]-np.mean(pos[1])
center_x=vd.xmid[0]-np.mean(pos[0])
pb.plot_alternatives(vd,pos_files=['lok1.txt','lok1_frame.txt'],shiftEW=-center_x, shiftNS=-center_y,scale=300)
#if the last file in pos_files ends in _frame.txt, frames will be drawn around the fish_farm.

#to add bottom velocities under each cage in map:
velstat='go'
#Only highlight bottom currents above 9 cm/s:
velmax=9




