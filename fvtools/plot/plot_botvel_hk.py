import sys
import os
import numpy as np
import utm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors 

import cmocean
import pandas as pd
from matplotlib.widgets import Button
from netCDF4 import Dataset
from glob import glob
import sys
sys.path.append('/work/hdj002/python_script/fvtools')
from fvtools.plot.fvcom_plot import geoplot
from fvtools.grid.fvcom_grd import FVCOM_grid

'''
_hk version, since this is modified by hdj002 (Hans Kristian Djuve) over several projects.
'''

# Add fvcom_pytools and its subdirectories to the path
# ----
#cwd = os.getcwd().split(os.getlogin())[0]
#for r,d,f in os.walk(cwd+os.getlogin()+'/'):
#    if 'fvcom_pytools' in r:
#        ptpath  = r
#        sys.path.append(ptpath)
#
#        print('added '+ptpath+' to the path...\nAdding all subdirectories...')
#
#        dirs    = glob(ptpath+'/*')
#        for thisdir in dirs:
#            if '.' in thisdir:
#                continue
#            if '__' in thisdir:
#                continue
#            sys.path.append(thisdir)
#        break

# apn codes
#from fvcom_plot import geoplot
#from fvcom_grd import FVCOM_grid

def botvel_object(grdfile = 'M.mat',\
                  N = None, E = None,\
                  fname   = 'velocities_'+os.getcwd().split('/')[-1]+'.npy',\
                  velkeys = ['vel95','vel95_2','vel99','vel99_2','velmax','velmax_2','velmean','velmean_2','velmean_u','velmean_v'],\
                  height  = [100]):
    
    '''
    Creates an object containing the estimate of the bottom velocities in the entire domain.
    
    The object (vd - velocity data) also contains some basic procedures
    for plotting, estimating nearest neighbors etc.

    # Input:
    # ----
    N and E: If you want to specify the origin in the figure, then specify N and E, and all distances in the plot will
             be relative to these. Either as a float decimal degree or string in deg*min.sec
    grdfile: File to read the FVCOM grid from
    fname:   File to read the velocity time series from
    velkeys: The velocity percentiles to find (those specified when creating the velocity_data.npy file
             for example ['vel95','vel99']
    height:  Height over the seafloor in centimeters, eg. [1,10,100,1000] ...

    '''

    vels    = np.load(fname, encoding = 'bytes', allow_pickle = True) # for regnekraft python 2.7 data at least
    vels    = vels.item()
    vd      = veldata(grdfile)

    print('Convert lat/lon to UTM33\n')
    if N is None:
        pts = utm.to_latlon(np.mean(vd.x),np.mean(vd.y),33,'W')
        N   = pts[0]; E = pts[1]
    if E is None:
        pts = utm.to_latlon(np.mean(vd.x),np.mean(vd.y),33,'W')
        N   = pts[0]; E = pts[1]

    convert_latlon(N,E,vd)

    print('Estimate bottom velocities based on the LOW\n')
    u_friclay, velkeys, height_s = get_botvel(vd,vels,velkeys=velkeys,height=height)
    
    # Get the cell triangles
    print('Store the data in the object')
    ctri         = get_ctri(grdfile)
    vd.u_friclay = u_friclay
    vd.velkeys   = velkeys
    vd.ctri      = ctri
    vd.height_s  = height_s
    return vd

# ---------------------------------------------------------------------------------------
#                           After the object is created...
# ---------------------------------------------------------------------------------------
def botvel_analysis(vd, show = False, velmax = [],\
                    shiftEW = 0, shiftNS = 0, scale=2000.0,\
                    dptinterval = 50.0, georef = False):
    '''
    Full botvel analysis. Saves figures to the working directory.
    
    show        - = True if you want to see the figures after they are made
    velmax      - maximum velocity on colorbar
    shift**     - Shift east/west or north/south (ie. if the facility is not in the
                  centre of the domain you are interested in).
    scale       - "radius" of the qube you are plotting
    dptinterval - The depth contours to plot
    georef      - Plot georeferenced picture together with the FVCOM output
                  georef='./Georefs/printout'
    '''

    # Ready to plot
    print('Visualize')
    for vel in vd.velkeys:
        for height in vd.height_s:
            vd.plot_vels(vd.ctri,vd.u_friclay[vel][height][:], velmax, shiftEW, shiftNS,\
                         title = os.getcwd().split('/')[-1]+'_'+vel+'_'+height,georef=georef,\
                         scale = scale, dptinterval=dptinterval)
    if show:
        plt.show()

    if  not show:
        plt.close('all')

def get_positions(vd,velkey = 'vel95',\
                  height  = '100 cm', velmax = 15,\
                  shiftEW = 0, shiftNS = 0, scale=2000.0,\
                  dptinterval = 25.0, georef = False):
    '''
    Get the lat/lon positions (in degree fractional-minute) of locations

    Mandatory:
    vd        - bottom velocities object

    Up to yourself:
    velkeys   - velkey to plot (eg. 'vel99')
    height    - height (eg. '100 cm')
    velmax    - maximum velocity on colorbar (eg. 15) would give 15 cm/s
    shift**   - Shift east/west or north/south (ie. if the facility is not in the centre of the domain you are interested in).
    georef    - Plot georeferenced picture together with the FVCOM output
    '''

    # Ready to plot
    print('Visualize')
    ax     = vd.return_plot(vd.ctri,vd.u_friclay['vel95'][vd.height_s[0]][:], velmax, shiftEW, shiftNS,\
                            title = os.getcwd().split('/')[-1]+'_'+velkey+'_'+vd.height_s[0],\
                            georef=georef, scale = scale, dptinterval=dptinterval)

    plt.title('Hit enter once you have selected all the stations you need')

    points = plt.ginput(n=-1, timeout=-1)
    x,y    = zip(*points)
    plt.scatter(x,y,c='r',marker='v')

    # Convert from UTM33 to lat lon format
    latlon = utm2sec(vd,points)
    plt.show()
    return latlon

def create_frame(point1, point2, length=450, width=180, nypen = 5, nxpen = 2, latlon=True):
    '''
    Given the extent of a frame, identify the position of the separate cages

    input:
    mandatory:
     - point1 - Used as south-easternmost point (assumed in (lat,lon) format)
     - point2 - Used to estimate angle of the structure

    optional:
     - length - length of the frame
     - width  - width of the frame

    A 90-90 grid cage structure will be created.
    - nypen   - Number of pens in the length direction
    - ncpen   - Number of pens in the width direction

    Coordinate system switch
    - latlon = True by default, set False if your input is UTM33 formatted

    ---------------

    returns:
     - position of centre points in UTM33 format (or latlon format if desired)
    '''
    if isinstance(point1[0],str):
        lat,lon = sec2frac(point1[0],point1[1])
        cart1   = utm.from_latlon(lat,lon,33,'W')

        lat,lon = sec2frac(point2[0],point2[1])
        cart2   = utm.from_latlon(lat,lon,33,'W')

    else:
        cart1   = utm.from_latlon(point1[0],point1[1],33,'W')
        cart2   = utm.from_latlon(point2[0],point2[1],33,'W')

    angle   = np.arctan2(cart1[1]-cart2[1],cart1[0]-cart2[0])+np.pi/2 # flip relative to north/south axis rather than east/west

    # Need the pens centre points - shift 45 m north and south from south eastern corner
    x       = np.linspace(cart1[0], cart1[0]+(width/nxpen)-width,nxpen)
    y       = np.linspace(cart1[1], cart1[1]-(length/nypen)+length,nypen)
    x,y     = np.meshgrid(x,y)

    # south-western corner:
    s       = cart1[1]
    w       = cart1[0]

    # rotate the facility.
    xx  = x-w; yy = y-s
    
    x   = xx*np.cos(-angle)+yy*np.sin(-angle)
    y   = -xx*np.sin(-angle)+yy*np.cos(-angle)

    # convert to vector
    x   = np.ravel(x+w)
    y   = np.ravel(y+s)

    if latlon:
        y, x = utm.to_latlon(x,y,33,'W')
    
    return x,y

def move_frame(vd, velkey, lat, lon, grdfile = 'M.mat', dptinterval = 25,\
               velmax=15, shiftEW=0, shiftNS=0, scale = 2000, pos_files = None, georef=False):
    '''
    Given the layout of a frame, move it around until you find the best fit to the bottom velocities
    input:

    mandatory (in this order):
    ----
    vd          - bottom velocities object
    velkey      - velocity percentile to plot as the background
    lat,lon     - cage structure (ie. from create_frame)

    Optional
    ----
    grdfile     - path to grid file (by default = 'M.mat', '*.nc' files are also accepted)
    dptinterval - depth contours to plot on top of the velocities
    velmax      - max velocity to contour
    shift*      - shift the centre of the figure east/west or north/south
    scale       - set the half width of the figure (default = 2000 [m])
    pos_files   - plot the pen positions from the positions file
    georef      - plot a georeferenced image in the background

    '''
    print('Loading data\n')
    # Find the position of the pens in UTM33 coordinates
    # ----------------------------------------------------
    vd.closest(lat,lon)

    # Ready to plot
    # ----------------------------------------------------
    oldpts = utm.from_latlon(lat,lon,33,'W')

    # Plot relative to this origo:
    # ----------------------------------------------------
    relx   = vd.xmid[0]; rely = vd.ymid[0]
    ptx    = vd.xmid;    pty  = vd.ymid

    # Background
    # ----------------------------------------------------
    ax     = vd.return_plot(vd.ctri,vd.u_friclay['vel95'][vd.height_s[0]][:], velmax, shiftEW, shiftNS,\
                            title = os.getcwd().split('/')[-1]+'_'+velkey+'_'+vd.height_s[0],\
                            georef=georef, scale = scale, dptinterval=dptinterval)

    # Load positions from posfolder
    # -----------------------------------------------
    if pos_files is not None:
        if type(pos_files) == 'str':
            pos = read_posfile(pos_files, plot=False, clr='k')
            plt.scatter(pos[0]-relx, pos[1]-rely, c='k')

        elif type(pos_files) == 'list':
            for pos_file in pos_files:
                pos = read_posfile(pos_file, plot=False, clr='k')
                plt.scatter(pos[0]-relx, pos[1]-rely, c='k')
        else:
            raise ValueError('\nYour pos_file must be either a string-path or a list of string-paths')

    l,     = plt.plot(ptx-relx,pty-rely,'m.')
    ll,    = plt.plot(ptx.mean()-relx,pty.mean()-rely,'mv')
    plt.plot(oldpts[0]-relx,oldpts[1]-rely,'k.')

    # Move the pen routine:
    # ----------------------------------------------------
    plt.title('The point you double-click will become the new centre point. Press a button to kill.')  
    while True:
        pts     = plt.ginput(n=1)
        xx,yy   = zip(*pts)

        ptx     = ptx-ptx.mean()+xx[0]; pty = pty-pty.mean()+yy[0]
        l.set_data(ptx,pty)
        ll.set_data(ptx.mean(),pty.mean())
        plt.pause(0.05)

        if plt.waitforbuttonpress():
            break

    # Write the output in lat-lon format
    # ----------------------------------------------------
    lon,lat = utm.to_latlon(ptx+vd.xmid[0],pty+vd.ymid[0],33,'W')
    df      = pd.DataFrame(lon.ravel(),lat.ravel())
    return df

def move_frame_nb(lat, lon, grdfile = 'M.mat',\
                  shiftEW = 0, shiftNS = 0, scale=2000.0):
    '''
    Given the layout of a frame, move it around until you find the best fit to the bottom velocities

    input:
    Mandatory:
    ----
    lat,lon   - cage structure (ie. from create_frame)

    Optional:
    ----
    grdfile   - a netcdf file from your FVCOM experiment or a M.mat file
    shift**   - Shift east/west or north/south (ie. if the facility is not in the centre of the domain you are interested in).
    scale     - set the half width of the figure
    '''

    # Ready to plot
    M      = FVCOM_grid(grdfile)
    oldpts = utm.from_latlon(lat,lon,33,'W')
    ptx    = oldpts[0]
    pty    = oldpts[1]
    l,     = plt.plot(ptx,pty,'m.')
    ll,    = plt.plot(ptx.mean(),pty.mean(),'mv')

    plt.title('The point you double-click will become the new centre point. Press a button to kill.')  

    while True:
        pts         = plt.ginput(n=1)
        xx,yy       = zip(*pts)
        
        # move the stuff
        ptx     = ptx-ptx.mean()+xx; pty = pty-pty.mean()+yy
        l.set_data(ptx,pty)
        ll.set_data(ptx.mean(),pty.mean())
        plt.pause(0.05)

        if plt.waitforbuttonpress():
            break

    lon,lat = utm.to_latlon(ptx,pty,33,'W')
    df      = pd.DataFrame(lon.ravel(),lat.ravel())
    return df


def plot_alternatives(vd, dptinterval = 25,\
                      velmax=15, shiftEW=0, shiftNS=0, scale = 2000, pos_files = None, georef=True, markers = None,velstat=None, quiver=None):
    '''
    Plot all cages tested on top of a bottom velocity plot
    input:

    Mandatory:
    ---------------------------------------------------------------------------------------
    vd          - bottom velocities object

    Optional:
    ---------------------------------------------------------------------------------------
    dptinterval - depth contour intervals
    shiftEW     - move the figure centre left or right
    shiftNS     - move the figure centre up or down
    scale       - half-width of the figure
    posfiles    - one or more path to 'position.txt' files (same as
                  used when creating flux files)
    georef      - add a georeference as background (boolean, looks for printout.j** files)
    markers     - list of markers to be used. One marker for each site (pos_file)
    velstat	- Plot velocities under each cage. type velstat='Go' or something to activate
    quiver - to add mean current quivers. Assumes
    		  Scale should be set quite small, like 400 or so, to get ok results.
    		  Also a hack to center the plot on the fish-farm that is plotted:
		  Run pos = pb_eli.read_posfile('coords_file.txt', plot=False, clr='k')
                  farmloc_x=vd.xmid[0]-np.mean(pos[0]), farmloc_y=vd.ymid[0]-np.mean(pos[1])
                  then as input to this func: shiftEW=-farmloc_x, shiftNS=-farmloc_y
    '''
    # Plot relative to this origo:
    # -------------------------------------------------------------------------------------
    relx   = vd.xmid[0]; rely = vd.ymid[0]
    ptx    = vd.xmid;    pty  = vd.ymid

    # Background
    # ----------------------------------------------------
    ax     = vd.return_plot(vd.ctri,vd.u_friclay['vel95'][vd.height_s[0]][:], velmax, shiftEW, shiftNS,\
                            title = os.getcwd().split('/')[-1]+' 95 percentile, '+vd.height_s[0],\
                            georef=georef, scale = scale, dptinterval=dptinterval)

    # Load positions from posfolder, scatter on top of bottom velocities
    # -----------------------------------------------
    cc=0
    if pos_files is not None:
        clrs = list(mcolors.TABLEAU_COLORS.keys()) 
        i = 0
        
        # Fix this string so that it becomes an iterable list
        if type(pos_files) == 'str':
            pos_files = [pos_files]

        for pos_file in pos_files:
            # Marker color
            if clrs[i]=='tab:olive':
                clr = 'w'
                
            elif clrs[i] == 'tab:green':
                clr = 'tab:cyan'
                
            else:
                clr = clrs[i]

            # Marker shape
            if markers is not None:
                mark = markers[i]
            else:
                mark = None
            if 'frame' in pos_file:
                pos = read_posfile(pos_file, relx, rely, cc, plot=True, clr='k')
                cc=cc+1
                plt.grid(b=None)
            else:
                pos = read_posfile(pos_file, plot=False, clr='k')
                plt.scatter(pos[0]-relx, pos[1]-rely, s=15, c=clr, marker = mark, label=pos_file.split('.')[-2])
                if velstat is not None:
                    lat,lon=utm.to_latlon(pos[0],pos[1],33,zone_letter='W')
                    vel_cell=vd.closest_cell(lat,lon)
                    for n in range(len(vel_cell)):
                        annotation_string  =r"%.1f $\mathrm{cm} / \mathrm{s}$" % (vd.u_friclay['vel95'][vd.height_s[0]][vel_cell[n]]*100)
                        plt.annotate(
                           annotation_string,
                           xy=(vd.xc[vel_cell[n]]-relx,vd.yc[vel_cell[n]]-rely),
                           xytext=(vd.xc[vel_cell[n]]-relx-40,vd.yc[vel_cell[n]]-rely+20),
                           color='white',
                           weight='bold',
                           fontsize='8',
                           bbox=dict(boxstyle="round", alpha=0.3))
                           #arrowprops=dict(arrowstyle='->', color='lime'))
            i+=1
        if quiver is not None:
            keep=quiver_mask_at_wanted_res(vd)
            plt.quiver(vd.xc[keep] - vd.xmid[0][0], vd.yc[keep] - vd.ymid[0][0], vd.u_friclay['velmean_u']['100 cm'][keep]*100, vd.u_friclay['velmean_v']['100 cm'][keep]*100, scale=10, scale_units='inches')
            plt.suptitle(r'$\rightarrow$'+' shows mean velocity direction at 100 cm above bottom',y=0.935,x=0.43)
        plt.legend()
        plt.grid(b=None)

def quiver_mask_at_wanted_res(vd,res=50):
    '''
    make a mask to plot quivers at fairly evenly spaces intervals.
    res	-	set wanted resolution. this is Mikes code originally developed for PO7 vector shape-maps.
    '''
    coord1=[vd.xc,vd.yc]
    coord1=np.asarray(coord1).T
    outer_bnds = [[np.floor(np.min(coord1[:,0])/res)*res, np.ceil(np.max(coord1[:,0])/res)*res],
				[np.floor(np.min(coord1[:,1])/res)*res, np.ceil(np.max(coord1[:,1])/res)*res]]

    x_pts = np.arange(outer_bnds[0][0], outer_bnds[0][1], res)
    y_pts = np.arange(outer_bnds[1][0], outer_bnds[1][1], res)

    x_mesh, y_mesh = np.meshgrid(x_pts, y_pts)
    x_all = x_mesh.ravel()
    y_all = y_mesh.ravel()

    coord1_boxes = np.asarray(np.floor((coord1[:,0]-outer_bnds[0][0])/res) 
					+ len(x_pts)*np.floor((coord1[:,1]-outer_bnds[1][0])/res), dtype=int)

    boxes, count = np.unique(coord1_boxes, return_counts=True)

    keep = np.isin(coord1_boxes, boxes[count == 1])

    tot_boxs = np.sum(count > 1)
    perc = 10
    print('0% complete')
    for i, this_box_to_thin in enumerate(boxes[count > 1]):
	    if (i/tot_boxs)*100 > perc:
		    print(f'{int(perc)}% complete')
		    perc+=10

	    c1_ind = np.where(coord1_boxes == this_box_to_thin)[0]
	    ctr_pt = [x_all[this_box_to_thin]+res/2, y_all[this_box_to_thin] + res/2]
	    keep_ind = np.argmin(np.sqrt((ctr_pt[0] - coord1[c1_ind,0])**2 + (ctr_pt[1] - coord1[c1_ind,0])**2))
	
	    keep[c1_ind[keep_ind]] = True
    print('100% complete. Keeping',sum(keep==True),' points')
    return keep
# ---------------------------------------------------------------------------------
#          Object holding the data we create and methods to visualize it
# ---------------------------------------------------------------------------------
class veldata():
    def __init__(self, grdf):
        if grdf[-3:] == 'npy':
           print('WARNING: This routine do not want M.npy. Give fvcom output-file! will now throw error about not finding attribute x')
        if grdf[-3:] == 'mat':
            d   = FVCOM_grid(grdf)
            self.x  = d.x[:,0]; self.y  = d.y[:,0]
            self.xc = d.xc[:,0]; self.yc = d.yc[:,0]
            self.nv = d.tri; self.h  = d.h[:,0]
            self.siglayz = -(d.h*d.siglay[:]).transpose(); self.siglevz = -(d.h*d.siglev[:]).transpose()

        elif grdf[-2:] == 'nc':
            d       = Dataset(grdf)
            self.x  = d['x'][:]; self.y  = d['y'][:]
            self.xc = d['xc'][:]; self.yc = d['yc'][:]
            self.nv = d['nv'][:].transpose()-1; self.h  = d['h'][:]
            self.siglayz = -self.h*d['siglay'][:]; self.siglevz = -self.h*d['siglev'][:]
            d.close()
        self.xmid = 0.0; self.ymid = 0.0
        self.cnode = 0
        
                
    def closest(self,N,E):
        '''
        Closest node given location of the fishcage in lat,lon format.
        (Lat Lon with decimal values)
        - Efficiency can be improved by adding UTM switch
        '''
        cnode = []
        if isinstance(N,list):
            for i in range(len(N)):
                u     = utm.from_latlon(N[i],E[i],33,'W')
                dst   = np.sqrt((self.x-u[0])**2+(self.y-u[1])**2)
                cnode.append(np.where(dst==dst.min())[0])

        elif isinstance(N,np.ndarray):
            for i in range(len(N)):
                u     = utm.from_latlon(N[i],E[i],33,'W')
                dst   = np.sqrt((self.x-u[0])**2+(self.y-u[1])**2)
                cnode.append(np.where(dst==dst.min())[0])

        else:
            u     = utm.from_latlon(N,E,33,'W')
            dst   = np.sqrt((self.x-u[0])**2+(self.y-u[1])**2)
            cnode.append(np.where(dst==dst.min())[0])

        cnode      = np.array(cnode, dtype='int')
        self.xmid  = self.x[cnode]
        self.ymid  = self.y[cnode]
        self.cnode = cnode[:]
        return cnode
        
    def closest_cell(self,N,E):
        '''
        Closest cell given location of the fishcage in lat,lon format.
        (Lat Lon with decimal values)
        - Efficiency can be improved by adding UTM switch
        '''
        ccell = []
        if isinstance(N,list):
            for i in range(len(N)):
                u     = utm.from_latlon(N[i],E[i],33,'W')
                dst   = np.sqrt((self.xc-u[0])**2+(self.yc-u[1])**2)
                ccell.append(np.where(dst==dst.min())[0])

        elif isinstance(N,np.ndarray):
            for i in range(len(N)):
                u     = utm.from_latlon(N[i],E[i],33,'W')
                dst   = np.sqrt((self.xc-u[0])**2+(self.yc-u[1])**2)
                ccell.append(np.where(dst==dst.min())[0])

        else:
            u     = utm.from_latlon(N,E,33,'W')
            dst   = np.sqrt((self.xc-u[0])**2+(self.yc-u[1])**2)
            ccell.append(np.where(dst==dst.min())[0])

        ccell      = np.array(ccell, dtype='int')
        self.xcmid  = self.xc[ccell]
        self.ycmid  = self.yc[ccell]
        self.ccell = ccell[:]
        return ccell
        
    def plot_vels(self, ctri, field, velmax, shiftE, shiftN, title = '', georef = False, scale=2000.0, dptinterval = 50.0):
        '''
        Automatic plot of the bottom velocities. Requires the cell triangles (ctri)
        and the field to be plotted. 
        Cell triangles are made in FVCOM_grid by calling FVCOM_grid().cell_tri()
        '''
        # Coordinates relative to...
        # -----------------------
        if isinstance(self.xmid,np.ndarray):
            xmid = self.xmid[0]
            ymid = self.ymid[0]
        else:
            xmid = self.xmid
            ymid = self.ymid

        xc = self.xc - xmid;  yc = self.yc - ymid
        x  = self.x  - xmid;  y  = self.y  - ymid

        # Do some colobar stuff
        # -----------------------
        extend = 'neither'
        if not velmax:
            vellvls = np.arange(0,(100*field).max()+1.0,1.0)
        else:
            vellvls = np.arange(0,velmax+1.0,1.0)
            if (100*field).max() >= velmax:
                extend  = 'max'

        # Create the figure
        # -----------------------
        plt.figure(figsize=(10,10))
        if georef:
            geoplot('printout.jpg', offx = xmid[0], offy = ymid[0], plot=True)
        plt.tricontourf(xc, yc, ctri, field*100, levels = vellvls, cmap = cmocean.cm.speed, extend=extend)
        plt.axis('equal')
        cbr = plt.colorbar()
        cbr.set_label('cm/s', size = 14)
        cbr.ax.tick_params(labelsize = 14)

        plt.xlim([-scale+shiftE, scale+shiftE])
        plt.ylim([-scale+shiftN, scale+shiftN])
        plt.scatter(self.xmid-xmid, self.ymid-ymid,s=10, marker='x', c='m')
        levels = np.arange(0,self.h.max()+dptinterval,dptinterval,dtype=int)
        
        fmt = '%r m'
        cb     = plt.tricontour(x,y,self.nv,self.h,levels=levels,linewidths=0.5)
        #plt.tricontour(xc,yc,ctri,field*100,levels=[10],colors='m')
        plt.clabel(cb, inline_spacing = 20, fontsize=10, fmt = fmt)

        plt.title(title+' above the bottom')
        plt.xlabel('m from 1st pen')
        plt.ylabel('m from 1st pen')
        plt.grid(b=None)
        plt.tight_layout()
        plt.savefig(title+'.png')

    def return_plot(self, ctri, field, velmax, shiftE, shiftN, title = '', georef = False, scale=2000.0, dptinterval = 50.0):
        '''
        Automatic plot of the bottom velocities. Requires the cell triangles (ctri)
        and the field to be plotted. 
        Cell triangles are made in FVCOM_grid by calling FVCOM_grid().cell_tri()
        '''
        # Coordinates relative to...
        if isinstance(self.xmid,np.ndarray):
            xmid = self.xmid[0]
            ymid = self.ymid[0]
        else:
            xmid = self.xmid
            ymid = self.ymid

        xc = self.xc - xmid;  yc = self.yc - ymid
        x  = self.x  - xmid;  y  = self.y  - ymid

        # Do some colobar stuff
        extend = 'neither'
        if not velmax:
            vellvls = np.arange(0,(100*field).max()+1.0,1.0)
        else:
            vellvls = np.arange(0,velmax+1.0,1.0)
            if (100*field).max() >= velmax:
                extend  = 'max'

        # Create the figure
        fig = plt.figure(figsize=(10,10))
        if georef:
            geoplot('printout.jpg', offx = xmid[0], offy = ymid[0], plot=True)
        if velmax==9:
           from matplotlib import colors
           plt.tricontourf(xc, yc, ctri, field*100, levels = [0,9,18], cmap=colors.ListedColormap(['w','g']), extend=extend)
        else:
           plt.tricontourf(xc, yc, ctri, field*100, levels = vellvls, cmap=cmocean.cm.speed, extend=extend)
        plt.axis('equal')
        cbr = plt.colorbar()
        cbr.set_label('cm/s', size = 14)
        cbr.ax.tick_params(labelsize = 14)

        plt.xlim([-scale+shiftE, scale+shiftE])
        plt.ylim([-scale+shiftN, scale+shiftN])
        plt.scatter(self.xmid-xmid, self.ymid-ymid,s=10, marker='x', c='m')
        levels = np.arange(0,self.h.max()+dptinterval,dptinterval,dtype=int)
        
        fmt = '%r m'
        cb     = plt.tricontour(x,y,self.nv,self.h,levels=levels,linewidths=0.5)
        #plt.tricontour(xc,yc,ctri,field*100,levels=[10],colors='m')
        plt.clabel(cb, inline_spacing = 20, fontsize=10, fmt = fmt)

        plt.title(title+' above the bottom')
        plt.xlabel('m from 1st pen')
        plt.ylabel('m from 1st pen')
        plt.grid(b=None)
        return fig

# ------------------------------------------------------------------------------------------------------
#                                            Physics
# ------------------------------------------------------------------------------------------------------

def get_botvel(grd,vels, Z0 = 0.001, velkeys = ['vel95','vel95_2','vel99','vel99_2','velmax','velmax_2','velmean','velmean_2'], height = [100]):
    '''
    Gives an estimate of bottom velocities using the law-of-the-wall equation.
    The standard value of Z0 is 0.001 in FVCOM. You will find it in your namelist
    if you are unsure.
    '''
    ZD           = grd.h-grd.siglayz[-1,:]
    ZD           = np.sum(ZD[grd.nv],axis=1)/3.0
    Cd = {}; blayer = {}; tau = {}; u_friclay = {}; height_s = []
    for high in height:
        height_s.append(str(high)+' cm')

    Cd_mesh      = low(ZD,Z0)
    for height, key in zip(height,height_s):
        ZD_const = (height/100)*np.ones(ZD.shape)
        Cd[key]  = low(ZD_const,Z0)

    for key in velkeys:
        tau[key] = {}
        for height in height_s:
            tau[key][height] = Cd_mesh*(vels[key][:]**2) #bytes(key,encoding='ascii')
            
    # Estimate velocity a given height over the seafloor
    for key in velkeys:
        u_friclay[key] = {}
            
        for height in height_s:
            u_friclay[key][height] = vels[key][:]#np.sqrt(tau[key][height][:]/Cd[height][:])

    return u_friclay, velkeys, height_s

def low(ZD, Z0):
    '''
    Law of the wall equations. Assuming steady state turbulence (no production
    of mean eddy kinetic energy), only mecanical turbulence is considered.
    The turbulence is assumed to horizontally homogeneous, and only produced
    through the mechanical terms of the equation.
    '''
    Cdn = []
    for i in range(len(ZD)):
        Cdn.append(np.max(((0.4)**2/((np.log(ZD[i]/Z0))**2),0.0025))) # FVCOM manual eq. 2.17
    return Cdn

# --------------------------------------------------------------
#                        Format stuff
# --------------------------------------------------------------
def sec2frac(N,E):
    '''
    from "70*11.19" to "70.1886111" format
    '''
    Nnew = []; Enew = []
    if isinstance(N,list):
        for i in range(len(N)):
            deg,mins  = N[i].split('*')
            Nnew.append(float(deg)+float(mins)/60.0)
            deg,mins  = E[i].split('*')
            Enew.append(float(deg)+float(mins)/60.0)
    else:
        deg,mins  = N.split('*')
        Nnew.append(float(deg)+float(mins)/60.0)
        deg,mins  = E.split('*')
        Enew.append(float(deg)+float(mins)/60.0)

    return Nnew, Enew

def utm2sec(vd,points):
    '''
    from meters to degree in "70*11.19" format
    '''
    latlon = []

    # Centre positions:
    if isinstance(vd.xmid,np.ndarray):
        xmid = vd.xmid[0]
        ymid = vd.ymid[0]
    else:
        xmid = vd.xmid
        ymid = vd.ymid
    
    if isinstance(points,list):
        for i in range(len(points)):
            ll        = utm.to_latlon(points[i][0]+xmid,points[i][1]+ymid,33,'W')
            Nnew = str(int(ll[0]))+'*'+str(np.round(((ll[0]-np.floor(ll[0]))*60)[0],4))
            Enew = str(int(ll[1]))+'*'+str(np.round(((ll[1]-np.floor(ll[1]))*60)[0],4))
            tmp  = (Nnew,Enew)
            latlon.append(tmp)

    else:
        ll        = utm.to_latlon(points[0]+xmid,points[1]+ymid,33,'W')
        Nnew = str(int(ll[0]))+'*'+str(np.round(((ll[0]-int(ll[0]))*60.0)[0],4))
        Enew = str(int(ll[1]))+'*'+str(np.round(((ll[1]-int(ll[1]))*60.0)[0],4))
        tmp  = (Nnew,Enew)
        latlon.append(tmp)

    return latlon

def deg2sec(N,E):
    '''
    Decimal degrees to seconds
    '''
    latlon = []
    Nnew   = str(int(N))+'*'+str(np.round(((N-int(N))*60.0)[0],4))
    Enew   = str(int(E))+'*'+str(np.round(((E-int(E))*60.0)[0],4))
    tmp    = (Nnew,Enew)
    latlon.append(tmp)

    return latlon

def convert_latlon(N,E,vd):
    if isinstance(N,list):
        if isinstance(N[0],str):
            N,E = sec2frac(N,E)

    elif isinstance(N,str):
        N,E = sec2frac(N,E)

    # Find the position of the pens in UTM33 coordinates
    vd.closest(N,E)

# ------------------------------------------------------------------
#                         Grid stuff
# ------------------------------------------------------------------
def get_ctri(grdfile):
    try:
        ctri = np.load('cell_tri.npy') # from fvcom_grd
    except:
        Mobj = FVCOM_grid(grdfile)
        ctri = Mobj.cell_tri()

    return ctri

# --------------------------------------------------------------------
#              Read positions we have already used
# --------------------------------------------------------------------
def read_posfile(position_file, relx=0, rely=0, cc=0, plot=True, clr='k'):
    '''
    reads old position file, returns positions of fish cages
    Assumes that a textfile is given
    '''
    colors_=['darkslateblue','brown','b','magenta']
    lon, lat = np.loadtxt(position_file, delimiter='\t', skiprows=1, unpack=True)
    pos      = utm.from_latlon(lat,lon,33,'W')
    if plot:
#        plt.scatter(pos[0],pos[1],c=clr,marker='x',s=10)
        plt.plot(pos[0]-relx,pos[1]-rely,color=colors_[cc])
        #'darkslateblue')#,path_effects=[pe.Stroke(linewidth=3, foreground='w'), pe.Normal()])
        plt.grid(b=None)
    return pos

# --------------------------------------------------------------------
#                      Create carbon file
# --------------------------------------------------------------------
