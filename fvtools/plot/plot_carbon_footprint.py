import sys
import utm
sys.path.append('/home/hdj002/python_script/fvcom_pytools/')
from grid.fvcom_grd import FVCOM_grid
import grid.fvcom_grd as fvg
#import matplotlib
#matplotlib.use('Qt4Agg')
from matplotlib import pyplot as plt
from matplotlib.legend import Legend
from netCDF4 import Dataset
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import spatial
from itertools import groupby
from math import pi
import pandas as pd 

from matplotlib.colors import LinearSegmentedColormap

cm_data = [[0.2081, 0.1663, 0.5292], [0.2116238095, 0.1897809524, 0.5776761905], [0.212252381, 0.2137714286, 0.6269714286], [0.2081, 0.2386, 0.6770857143], [0.1959047619, 0.2644571429, 0.7279], [0.1707285714, 0.2919380952, 0.779247619], [0.1252714286, 0.3242428571, 0.8302714286], [0.0591333333, 0.3598333333, 0.8683333333], [0.0116952381, 0.3875095238, 0.8819571429], [0.0059571429, 0.4086142857, 0.8828428571], [0.0165142857, 0.4266, 0.8786333333], [0.032852381, 0.4430428571, 0.8719571429], [0.0498142857, 0.4585714286, 0.8640571429], [0.0629333333, 0.4736904762, 0.8554380952], [0.0722666667, 0.4886666667, 0.8467], [0.0779428571, 0.5039857143, 0.8383714286], [0.079347619, 0.5200238095, 0.8311809524], [0.0749428571, 0.5375428571, 0.8262714286], [0.0640571429, 0.5569857143, 0.8239571429], [0.0487714286, 0.5772238095, 0.8228285714], [0.0343428571, 0.5965809524, 0.819852381], [0.0265, 0.6137, 0.8135], [0.0238904762, 0.6286619048, 0.8037619048], [0.0230904762, 0.6417857143, 0.7912666667], [0.0227714286, 0.6534857143, 0.7767571429], [0.0266619048, 0.6641952381, 0.7607190476], [0.0383714286, 0.6742714286, 0.743552381], [0.0589714286, 0.6837571429, 0.7253857143], [0.0843, 0.6928333333, 0.7061666667], [0.1132952381, 0.7015, 0.6858571429], [0.1452714286, 0.7097571429, 0.6646285714], [0.1801333333, 0.7176571429, 0.6424333333], [0.2178285714, 0.7250428571, 0.6192619048], [0.2586428571, 0.7317142857, 0.5954285714], [0.3021714286, 0.7376047619, 0.5711857143], [0.3481666667, 0.7424333333, 0.5472666667], [0.3952571429, 0.7459, 0.5244428571], [0.4420095238, 0.7480809524, 0.5033142857], [0.4871238095, 0.7490619048, 0.4839761905], [0.5300285714, 0.7491142857, 0.4661142857], [0.5708571429, 0.7485190476, 0.4493904762], [0.609852381, 0.7473142857, 0.4336857143], [0.6473, 0.7456, 0.4188], [0.6834190476, 0.7434761905, 0.4044333333], [0.7184095238, 0.7411333333, 0.3904761905], [0.7524857143, 0.7384, 0.3768142857], [0.7858428571, 0.7355666667, 0.3632714286], [0.8185047619, 0.7327333333, 0.3497904762], [0.8506571429, 0.7299, 0.3360285714], [0.8824333333, 0.7274333333, 0.3217], [0.9139333333, 0.7257857143, 0.3062761905], [0.9449571429, 0.7261142857, 0.2886428571], [0.9738952381, 0.7313952381, 0.266647619], [0.9937714286, 0.7454571429, 0.240347619], [0.9990428571, 0.7653142857, 0.2164142857], [0.9955333333, 0.7860571429, 0.196652381], [0.988, 0.8066, 0.1793666667], [0.9788571429, 0.8271428571, 0.1633142857], [0.9697, 0.8481380952, 0.147452381], [0.9625857143, 0.8705142857, 0.1309], [0.9588714286, 0.8949, 0.1132428571], [0.9598238095, 0.9218333333, 0.0948380952], [0.9661, 0.9514428571, 0.0755333333], [0.9763, 0.9831, 0.0538]]

parula_map = LinearSegmentedColormap.from_list('parula', cm_data)

def main(filename     = 'case.nc', 
         sim_length   = 30, 
         pos_files  = None,
         carbon_org = None, 
         figure_title     = None, 
         scale    = 450,
         cage_omkrets = 140.0,
         plot_arrows    = False,
         sed_max         = 1.5,
         make_olex_isoline_file         = False,
         save_fig     = True,
         lok_number= 0,
         make_carbon_csv = True,
         addmask = False):
    '''
    Plot Carbon footprint with 10g and 0.75g (AZE) iso - lines
    This script is mainly used to plot carbon footprint after the max production month.
    example of a carbon_org file and pos_files can be found here:
    /work/hdj002/python_script/handy_script/botvel_and_sedimentation_plotting/
 
    Mandatory:
    ---------------------------------------------------------------------------------------
    
    filename - usually last outputfile from the simulation-month you want to plot
    
    Optional
    ---------------------------------------------------------------------------------------
    sim_length 	- Duration of the run. Usually number of days in the month simulated
    pos_files 	- assumes the location you want to plot carbon footprint from as first location listed
    carbon_org 	- Actual carbon amount from the excel sheet used to make the carbon fluxfile.
                          skips row 1, should be either 4 or 7 tracers. (so far, tracer 7 is never relevant, as this feed size is never active during max production)
    scale 	- In reports I usually go for 450 for a close-up plot, and 1000 for a zoom-out plot
             Tip: set below 600 to plot carbon values in map for each cage
    plot_arrows 	-  HIGHLY situatinal. When scale is >600, if cage orientation is so the carbon numbers for each cage overlap, turn this on.
    sed_max 	- upper cutoff value for colorbar. lower cutoff is always 10 times lower than AZE
    make_olex_isoline_file 	- Sometimes some aquaculture collegues wants this to put into their olex.
    lok_number	-	If several scenarios in same file. set number 1, 2 or 3 etc.
    addmask - If the calibration parameters are more than 10% off for several tracers, consider only looking at sediment nearby, to get a more accurate analysis. make a mask and call it mymask.npy.
    
    like this:
    --------------------
    #make polygons
import numpy as np
from matplotlib.path import Path
from matplotlib import pyplot as plt
from grid.fvcom_grd import FVCOM_grid
%matplotlib
filename_sed='storholmen_0032.nc'
grd=FVCOM_grid(filename_sed)
self=grd
plt.triplot(self.x[:,0],self.y[:,0],self.tri,lw=0.2,c='g')
plt.axis('equal')
plt.title('Click and create a polygon. Hit Enter when the region is bounded')

pts = plt.ginput(n=-1,timeout=-1)
x,y = zip(*pts)
x = np.array(x)
y = np.array(y)
sti = np.column_stack((x,y))
p = Path(sti)
queries       = np.column_stack((self.x,self.y))
self.ind_bool = p.contains_points(queries)
#plt.scatter(self.x[self.ind_bool],self.y[self.ind_bool],s=10,c='r')
np.save('mymask.npy',self.ind_bool)
----------------------
    
    PS: I had to downgrade my matplotlib to version 3.1.3 because of a error:
    "cannot import name '_api' from 'matplotlib'"
    
    By Hans Kristian Djuve, hkd@akvaplan.niva.no
    '''        
    filename_sed=filename
    data=Dataset(filename_sed)
    grd=fvg.FVCOM_grid(filename_sed)
    #kb=grd.get_kb()
    Feces_factor = 1.59*2.94
    Feed_factor  = 1.0/0.57
    
    pos=read_posfile(pos_files[0])[0:2]
    if len(pos_files) >1:
        for n in range(1,len(pos_files)):
            pos=pos+(read_posfile(pos_files[n]))[0:2]
    print('Found ',len(pos_files),' locations')
    
    #Center of cage we plot carbon for:
    xx=np.mean(pos[0])
    yy=np.mean(pos[1])
    
    art1=data['art1'][:]

    carbon_benchmark=np.loadtxt(carbon_org,skiprows=1)
    mask=[i for i,v in enumerate(grd.x) if v > 0] # the entire grid
    if addmask==True:
         mask=np.load('mymask.npy') # the specified area of interest, if specified, made by the routine given in line 66-87
         
    
    if len(carbon_benchmark)==4: #when running with tracer 4,5,6 and 8, where tracer 4 is tracer 1,2,3 and 4 combined, with a sinking parameter of 864 (in the fabm.yaml file)
        model_carbon=np.array([sum(data[('tracer{_4}_c_bot'.format(_4=4+lok_number*10))][-1,mask]*art1[mask])/Feces_factor, sum(data[('tracer{_5}_c_bot'.format(_5=5+lok_number*10))][-1,mask]*art1[mask])/Feces_factor, sum(data[('tracer{_6}_c_bot'.format(_6=6+lok_number*10))][-1,mask]*art1[mask])/Feces_factor, sum(data[('tracer{_8}_c_bot'.format(_8=8+lok_number*10))][-1,mask]*art1[mask])/Feed_factor])
        model_carbon=model_carbon/1000 #converts to kg as unit is in gram in output file
        calibration_param=carbon_benchmark / model_carbon
        print('calibration_params',calibration_param)
        carbon_total = (data['tracer4_c_bot'][-1,mask]*calibration_param[0] + data['tracer5_c_bot'][-1,mask]*calibration_param[1] + data['tracer6_c_bot'][-1,mask]*calibration_param[2])/Feces_factor + data['tracer8_c_bot'][-1,mask]*calibration_param[3]/Feed_factor
        sediment_total = data['tracer4_c_bot'][-1,mask]*calibration_param[0] + data['tracer5_c_bot'][-1,mask]*calibration_param[1] + data['tracer6_c_bot'][-1,mask]*calibration_param[2] + data['tracer8_c_bot'][-1,mask]*calibration_param[3]
    else: #if not 4 tracers, we assume all 8 (7) is used.
        model_carbon=np.array([sum(data[('tracer{_1}_c_bot'.format(_1=1+lok_number*10))][-1,mask]*art1[mask])/Feces_factor,sum(data[('tracer{_2}_c_bot'.format(_2=2+lok_number*10))][-1,mask]*art1[mask])/Feces_factor,sum(data[('tracer{_3}_c_bot'.format(_3=3+lok_number*10))][-1,mask]*art1[mask])/Feces_factor,sum(data[('tracer{_4}_c_bot'.format(_4=4+lok_number*10))][-1,mask]*art1[mask])/Feces_factor, sum(data[('tracer{_5}_c_bot'.format(_5=5+lok_number*10))][-1,mask]*art1[mask])/Feces_factor, sum(data[('tracer{_6}_c_bot'.format(_6=6+lok_number*10))][-1,mask]*art1[mask])/Feces_factor, sum(data[('tracer{_8}_c_bot'.format(_8=8+lok_number*10))][-1,mask]*art1[mask])/Feed_factor])
        model_carbon=model_carbon/1000 #converts to kg as unit is in gram in output file
        calibration_param=carbon_benchmark / model_carbon
        print('calibration_params',calibration_param)
        carbon_total = (data[('tracer{_1}_c_bot'.format(_1=1+lok_number*10))][-1,:]*calibration_param[0] + data[('tracer{_2}_c_bot'.format(_2=2+lok_number*10))][-1,:]*calibration_param[1] +data[('tracer{_3}_c_bot'.format(_3=3+lok_number*10))][-1,:]*calibration_param[2] + data[('tracer{_4}_c_bot'.format(_4=4+lok_number*10))][-1,:]*calibration_param[3] + data[('tracer{_5}_c_bot'.format(_5=5+lok_number*10))][-1,:]*calibration_param[5] + data[('tracer{_6}_c_bot'.format(_6=6+lok_number*10))][-1,:]*calibration_param[5])/Feces_factor + data[('tracer{_8}_c_bot'.format(_8=8+lok_number*10))][-1,:]*calibration_param[6]/Feed_factor
        
        sediment_total = data[('tracer{_1}_c_bot'.format(_1=1+lok_number*10))][-1,:]*calibration_param[0] + data[('tracer{_2}_c_bot'.format(_2=2+lok_number*10))][-1,:]*calibration_param[1] +data[('tracer{_3}_c_bot'.format(_3=3+lok_number*10))][-1,:]*calibration_param[2] + data[('tracer{_4}_c_bot'.format(_4=4+lok_number*10))][-1,:]*calibration_param[3] + data[('tracer{_5}_c_bot'.format(_5=5+lok_number*10))][-1,:]*calibration_param[5] + data[('tracer{_6}_c_bot'.format(_6=6+lok_number*10))][-1,:]*calibration_param[5] + data[('tracer{_8}_c_bot'.format(_8=8+lok_number*10))][-1,:]*calibration_param[6]
        
        
    carbon_total = carbon_total / 1000
    sediment_total = sediment_total / 1000
    
    if make_carbon_csv == True: # carbon cvs file should be added to project area. useful for Rune and Astrid for putting into gis when planning ROV surveys.
        indices = np.where(carbon_total>1e-6)
        df_carbon=pd.DataFrame({"utm y" : grd.y[indices],
        "utm x" : grd.x[indices],
        "karbon på havbunn (g/m2/måned)" : carbon_total[indices],
        "Dyp" : grd.h[indices]})      
        df_carbon.to_csv("karbon_{lok}.csv".format(lok=figure_title), index=False)
        
        df_sediment=pd.DataFrame({"utm y" : grd.y[indices],
        "utm x" : grd.x[indices],
        "sediment på havbunn (g/m2/måned)" : sediment_total[indices],
        "Dyp" : grd.h[indices]})      
        df_sediment.to_csv("sediment_{lok}.csv".format(lok=figure_title), index=False)

    
    
    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(111)


    plt.grid(True,linestyle='-',linewidth=0.2,color='black')
    ax.set_title('Carbon footprint')
    if figure_title is not None:
        ax.set_title(figure_title)
        
    ax.set_xlim([xx-scale,xx+scale,])
    ax.set_ylim([yy-scale,yy+scale,])  
    ax.set_xticks([])
    ax.set_yticks([])
    
    c_max=carbon_total.max()
    c_max_=c_max/100*95;
    area_cmax=sum(art1[np.where(carbon_total>=c_max_)])
    area_cmax=int(round(area_cmax,-1))
    annotation_string  =r"Max carbon value is %.2f $\mathrm{kg} / \mathrm{m}^2 / \mathrm{month}$" % (c_max)
    annotation_string += "\n"
    annotation_string += r"Area with values exceeding 95%% of the max value: %.2f$\mathrm{kg} / \mathrm{m}^2 / \mathrm{month}$ is %.0f $\mathrm{m}^2$" %(c_max_,area_cmax)
    annotation_string += "\n"
    annotation_string+=r"Carbon released this month is %.2f tonnes" % (sum(carbon_total*art1)/1000)
    #plt.annotate(annotation_string,xy=(-20.155,0.905), xycoords='axes fraction', bbox=dict(boxstyle="round",fc="0.8",ec="none"))
    plt.annotate(annotation_string,xy=(0.015,0.905), xycoords='axes fraction', bbox=dict(boxstyle="round",fc="0.8",ec="none"))
    
    #dot0=plt.scatter(pos[0],pos[1],marker='.',c='m',zorder=3)
    colors_=['m','r','brown','slateblue','indigo','palevioletred']
    n=0
    for pos_file in pos_files:
        pos_plt=read_posfile(pos_file)
        plt.scatter(pos_plt[0],pos_plt[1],marker='*',c=colors_[n],label=pos_file.split('.')[-2],zorder=2)
        n=n+1
    plt.legend()
    
#find concentration in x number of cells closest to each cage
#KDtree ftw!
    A_=np.stack((grd.x,grd.y), axis=-1)
    id_all=0
    for k in range(len(pos[0])):
        neighbor_number=15
        neighbors=A_[spatial.cKDTree(A_).query((pos[0][k],pos[1][k]),neighbor_number)[1]]
        id_=[(A_[:,0]==neighbors[n,0]).nonzero() for n in range(len(neighbors))]

        id__=np.array(id_[:])
        id_shape=id__.shape
        for n in range(id_shape[0]):
           if len(id__[n][0])>1:
              ind_max_point=np.argmax(carbon_total[id_[n][0]])
              id_[n]=id_[n][0][ind_max_point]
        id_ = [int(''.join(i)) for is_digit, i in groupby(str(id_), str.isdigit) if is_digit]
#    plt.plot(grd.x[id_],grd.y[id_],'.m') #plot neighbors that was considered
        id_=id_[np.argmax(carbon_total[id_])]
#    if neighbor_number>1:
#        id_all=np.append(id_all,id_[:])
#    else:
        id_all=np.append(id_all,id_)
    id_all=id_all[1:]
    id_rsp=np.reshape(id_all,(len(pos[0]),1))
    meanconc=[np.mean(carbon_total[id_rsp[k,:]]) for k in range(len(pos[0]))]
    b=-1
    if scale<600:
        for n in range(len(meanconc)):
            b=b*-1
            annotation_string  =r"%.2f $\mathrm{kg} / \mathrm{m}^2$" % (round(meanconc[n],2))
            if plot_arrows == False:
                ax.annotate(annotation_string, xy=(pos[0][n],pos[1][n]),
                xytext=(pos[0][n],pos[1][n]+15),color='black',weight='bold',fontsize='13')
            else:
                ax.annotate(annotation_string, xy=(pos[0][n],pos[1][n]),
                xytext=(pos[0][n],pos[1][n]+(15+(11*5))*b),color='k',weight='bold',fontsize='13',arrowprops=dict(arrowstyle="fancy",fc="0.6",
                ec="lime",connectionstyle="angle3,angleA=0,angleB=-90"))
                
    for midtpunk in range(len(pos[0])):
        radius_=cage_omkrets/(2*pi)
        circle1=plt.Circle((pos[0][midtpunk],pos[1][midtpunk]),radius_,color='k',fill=False,zorder=2)
        plt.gcf().gca().add_artist(circle1)
    
    ax.set_facecolor((0.4,0.6,0.3)) #define background color
    tp=grd.plot_cvs(carbon_total,cmap=parula_map)
    tp.cmap.set_under([0.8, 1, 1])
    plt.clim(sim_length*0.75/10000,sed_max)#cutoff value, 0.02325kg/month=0.75g/day :fant slik: 0.75*31,, ved 30 dager: 0.0225
#cutoff value, 0.031kg/month=1g/day (0.030 if month got 30 days, 0.028 if february)
    
    div = make_axes_locatable(ax)
    cax = div.append_axes("right", size="5%", pad=0.2)
    cb = fig.colorbar(tp, cax=cax)
    cb.set_label(r'[$ \mathrm{kg} \, / \mathrm{m}^2 / \mathrm{month}  $]',fontsize=14)
    
    #ADD ISO LINE
    CS = ax.tricontour(grd.x, grd.y, carbon_total,[sim_length*0.75/1000],colors='tomato',linestyles=['solid'],linewidths=[2],interpolation='spline')
    handles, labels = CS.legend_elements() #0.75g iso-line

#To compare 0.75g isolinjes with 1g iso_lines:    
#    CS3 = ax.tricontour(grd.x, grd.y, carbon_total,[sim_length*1/1000],colors='magenta',linestyles=['dashed'],linewidths=[2],interpolation='spline')
#    handles, labels = CS.legend_elements() #1g iso-line
    
    CS2 = ax.tricontour(grd.x, grd.y, carbon_total,[10/sim_length],colors='tan',linestyles=['solid'],linewidths=[2],interpolation='spline')
    handles2, labels2 = CS2.legend_elements() #10g iso-line
    
    leg = Legend(ax,handles,['AZE (0.75'+r'$\;\mathrm{g} / \mathrm{m}^2 / \mathrm{day}$)'],loc='lower left')
    ax.add_artist(leg)
    leg2 = Legend(ax,handles2,['10'+r'$\;\mathrm{g} / \mathrm{m}^2 / \mathrm{day}$'],loc='lower right')
    ax.add_artist(leg2)
    #Add depth countour:
    levelss=np.arange(25,np.max(grd.h[:]),25)
    CL = ax.tricontour(grd.x, grd.y, grd.tri, grd.h,levels=levelss,colors='black',linewidths=[0.7])
    handles3, labels3 = CL.legend_elements()
    leg3 = Legend(ax,handles3,['25m depth countourlines'],loc='lower center')
    ax.add_artist(leg3)
    
    #Add option to make olex plot of isoline
    if make_olex_isoline_file==True:
        isoline_nr=len(CS.allsegs[0])
        olex_str=' 1393323080 Brunsirkel'
#int(sys.argv[1])==1:
        import re
        for isoindex in range(int(isoline_nr)):
             iso_points=CS.allsegs[0][isoindex]
             latlon=iso_points
             for index in range(len(iso_points[:,0])):
                 latlon[index,0],latlon[index,1]=utm.to_latlon(iso_points[index,0],iso_points[index,1], 33, 'W')
             with open("contourlines_print_"+str(sim_length*0.75/1000)+"g"+"_line_"+str(isoindex),"w") as output:
                   print(re.sub('[\[\]]', '', (str(60*latlon[0:len(iso_points[:,0])]))),file=output)
#                   print(re.sub('[\[\]]', '', (str(60*latlon[0:len(iso_points[:,0]):2]))),file=output)
#                 print >> output, (re.sub('[\[\]]', '', (str(60*latlon[0:len(iso_points[:,0]):2]))))
             with open("contourlines_print_"+str(sim_length*0.75/1000)+"g"+"_line_"+str(isoindex),"r") as f:
                 lines=f.readlines()
             for index, line in enumerate(lines):
                 lines[index]=line.strip() + olex_str + "\n"
             lines.insert(0,'Plottsett 8\n')
             lines.insert(0,'Rute uten navn 8\n')
             lines.insert(0,'\n')
             lines.insert(0,'Ferdig forenklet\n')
             with open("olex_0.75g_isolinje_{lok}".format(lok=figure_title)+"_nr_"+str(isoindex+1),"w") as f:
                 for line in lines:
                     f.write(line)
     #f.write(line.rstrip('\n') + olex_str + '\n')

        print('iso line textfiles done. Only picking every 2nd coordinate. Should have produced this many files: ',isoline_nr)
#Need to plot this again as the making of olex files somehow removes the iso-lines from the plot
        CS = ax.tricontour(grd.x, grd.y, carbon_total,[sim_length*0.75/1000],colors='tomato',linestyles=['solid'],linewidths=[3],interpolation='spline')
    gridlines=plt.grid(True,linestyle='-',linewidth=0.2,color='black',zorder=3)
    ax.set_axisbelow(False)
    
    title_chunk=figure_title.split('_')
    chunks=filename.split('_')
    if chunks[0][-2:] == 'nc':
        savename=chunks[0][0:-3]
    else:
        savename= chunks[0]

    if save_fig == True:
        plt.savefig('Carbon_scale_{scale}_{savename}_{lokname}.png'.format(scale=scale,savename=savename,lokname=title_chunk[0]), bbox_inches='tight', format='png')

    return fig

def read_posfile(position_file, relx=0, rely=0, plot=False, clr='k'):
    '''
    reads old position file, returns positions of fish cages
    Assumes that a textfile is given
    '''
    lon, lat = np.loadtxt(position_file, delimiter='\t', skiprows=1, unpack=True)
    pos      = utm.from_latlon(lat,lon,33,'W')
    if plot:
#        plt.scatter(pos[0],pos[1],c=clr,marker='x',s=10)
        plt.plot(pos[0]-relx,pos[1]-rely,'darkslateblue')#,path_effects=[pe.Stroke(linewidth=3, foreground='w'), pe.Normal()])
        plt.grid(b=None)
    return pos
