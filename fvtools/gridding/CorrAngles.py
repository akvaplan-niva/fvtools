import sys
import numpy as np
import matplotlib.pyplot as plt
from fvtools.grid.fvcom_grd import FVCOM_grid

def angles(xn, yn, tri):
    '''Computes and returns all angles in a grid'''
    x     = np.array([xn[tri[:,0]],xn[tri[:,1]],xn[tri[:,2]]])
    y     = np.array([yn[tri[:,0]],yn[tri[:,1]],yn[tri[:,2]]])
    # COMPUTE ANGLES
    # cos(theta) = a * b / |a||b|
    AB    = np.array([x[0,:]-x[1,:], y[0,:]-y[1,:]])
    BC    = np.array([x[1,:]-x[2,:], y[1,:]-y[2,:]])
    CA    = np.array([x[2,:]-x[0,:], y[2,:]-y[0,:]])
    # Get length of each triangle side
    lAB   = np.sqrt(AB[0,:]**2+AB[1,:]**2)
    lBC   = np.sqrt(BC[0,:]**2+BC[1,:]**2)
    lCA   = np.sqrt(CA[0,:]**2+CA[1,:]**2)
    # Get dot products
    ABAC  = -(AB[0,:]*CA[0,:]+AB[1,:]*CA[1,:])
    BABC  = -(AB[0,:]*BC[0,:]+AB[1,:]*BC[1,:])
    CABC  = -(CA[0,:]*BC[0,:]+CA[1,:]*BC[1,:])
    # Get angles
    theta = np.array([np.arccos(ABAC/(lAB*lCA)), np.arccos(BABC/(lAB*lBC)), np.arccos(CABC/(lCA*lBC))])

    return theta

def move_vertices(xn, yn, tri, minangle=35., N=10.):
    xnode = xn
    ynode = yn
    x = np.array([xn[tri[:,0]],xn[tri[:,1]],xn[tri[:,2]]])
    y = np.array([yn[tri[:,0]],yn[tri[:,1]],yn[tri[:,2]]])
    theta = angles(xn, yn, tri)
    atresh = minangle/180.*np.pi
    [i,j] = np.where(theta<atresh)
    a = np.arange(1,10)/N
    print('There is ' + str(len(j)) + ' elements with angles < ' + str(minangle) + ' degrees')
    #Original position of the nodes in triangles. (xv,yv) is the node located in the cornerwith small angle
    for n in np.arange(len(j)):
        print('Element ' + str(n) + ' of ' + str(len(j)))
        xv = x[i[n],j[n]]
        yv = y[i[n],j[n]]
        x1 = x[int(i[n]+1-np.floor((i[n]+1)/3)*3),j[n]]
        y1 = y[int(i[n]+1-np.floor((i[n]+1)/3)*3),j[n]]
        x2 = x[int(i[n]+2-np.floor((i[n]+2)/3)*3),j[n]]
        y2 = y[int(i[n]+2-np.floor((i[n]+2)/3)*3),j[n]]

        #Move node at position of small angle
        #---------------------------------------
        am = np.arange(11) / N
        dx = x2 - x1
        dy = y2 - y1
        xx1 = x1 - dx
        yy1 = y1 - dy
        xx2 = x2 + dx
        yy2 = y2 + dy
        xm = am * xx1 + (1 - am) * xx2
        ym = am * yy1 + (1 - am) * yy2
        xx = np.empty(0)
        yy = np.empty(0)
        for p in np.arange(len(xm)):
            xtmp = a * xv + (1 - a) * xm[p]
            ytmp = a * yv + (1 - a) * ym[p]
            xx = np.append(xx,xtmp)
            yy = np.append(yy,ytmp)

        #Find new positions where angle is larger than minangle
        crosp = (x1-xx)*(x2-xx) + (y1-yy)*(y2-yy)
        l1 = np.sqrt(np.square(x1-xx)+np.square(y1-yy))
        l2 = np.sqrt(np.square(x2-xx)+np.square(y2-yy))
        an = np.arccos(crosp/(l1*l2)) / np.pi * 180.0
        xx = xx[an>minangle]
        yy = yy[an>minangle]
        an = an[an>minangle]

        #Find all neighboring cells compute angles
        nidv = tri[j[n], i[n]]
        xnv, ynv, Rminv = find_nodepos(j[n], nidv, theta, xx, yy, tri, xn, yn)

        #Move node at position 1
        #---------------------------------------
        xx1 = x1 - (y1 - yv)
        yy1 = y1 + (x1 - xv)
        xx2 = x1 + (y1 - yv)
        yy2 = y1 - (x1 - xv)
        am = np.arange(11) / N
        xm = am * xx1 + (1 - am) * xx2
        ym = am * yy1 + (1 - am) * yy2
        xx = np.empty(0)
        yy = np.empty(0)
        for p in np.arange(len(xm)):
            xtmp = a * xv + (1 - a) * xm[p]
            ytmp = a * yv + (1 - a) * ym[p]
            xx = np.append(xx,xtmp)
            yy = np.append(yy,ytmp)

        #Find new positions where angle is larger than minangle
        crosp = (xx-xv)*(x2-xv) + (yy-yv)*(y2-yv)
        l1 = np.sqrt(np.square(xv-xx)+np.square(yv-yy))
        l2 = np.sqrt(np.square(x2-xv)+np.square(y2-yv))
        an = np.arccos(crosp/(l1*l2)) / np.pi * 180.0
        xx = xx[an>minangle]
        yy = yy[an>minangle]
        an = an[an>minangle]

        #Find all neighboring cells compute angles
        i1 = int(i[n]+1-np.floor((i[n]+1)/3)*3)
        nid1 = tri[j[n], i1]
        xn1, yn1, Rmin1 = find_nodepos(j[n], nid1, theta, xx, yy, tri, xn, yn)

        #Move node at position 2
        #---------------------------------------
        xx1 = x2 - (y2 - yv)
        yy1 = y2 + (x2 - xv)
        xx2 = x2 + (y2 - yv)
        yy2 = y2 - (x2 - xv)
        am = np.arange(11) / N
        xm = am * xx1 + (1 - am) * xx2
        ym = am * yy1 + (1 - am) * yy2
        xx = np.empty(0)
        yy = np.empty(0)
        for p in np.arange(len(xm)):
            xtmp = a * xv + (1 - a) * xm[p]
            ytmp = a * yv + (1 - a) * ym[p]
            xx = np.append(xx,xtmp)
            yy = np.append(yy,ytmp)

        #Find new positions where angle is larger than minangle
        crosp = (xx-xv)*(x1-xv) + (yy-yv)*(y1-yv)
        l1 = np.sqrt(np.square(xv-xx)+np.square(yv-yy))
        l2 = np.sqrt(np.square(x1-xv)+np.square(y1-yv))
        an = np.arccos(crosp/(l1*l2)) / np.pi * 180.0
        xx = xx[an>minangle]
        yy = yy[an>minangle]
        an = an[an>minangle]

        #Find all neighboring cells compute angles
        i2 = int(i1+1-np.floor((i1+1)/3)*3)
        nid2 = tri[j[n], i2]
        xn2, yn2, Rmin2 = find_nodepos(j[n], nid2, theta, xx, yy, tri, xn, yn)

        #Determine which of the three nodes to move.
        R = [Rminv, Rmin1, Rmin2]
        xnew = [xnv, xn1, xn2]
        ynew = [ynv, yn1, yn2]
        nid = [nidv, nid1, nid2]
        k = np.where(R==np.min(R))
        k = int(k[0])
        xnode[nid[k]] = xnew[k]
        ynode[nid[k]] = ynew[k]
        
    theta = angles(xnode, ynode, tri)
    nj0 = len(j)
    [i,j] = np.where(theta<atresh)
    print('Number of angles less than ' + str(minangle) + ' reduced from ' + str(nj0) + ' to ' + str(len(j)))

    return xnode, ynode

def find_nodepos(j, nid, theta, xx, yy, tri, xn, yn):
        '''Finds the node position that is nearest to 60 deg angles in all neighboring cells.
           Returns new node position and the minimum of the quadratic function R'''
        cid = neighbors(j, tri)
        t = tri[cid,:]
        R = np.zeros(xx.shape)
        t60 = 60./180.*np.pi
        for c in np.arange(len(cid)):
            if np.any(tri[cid[c],:] == nid):
                cv = np.where(tri[cid[c],:] == nid)
                cv = int(cv[0])
                c1 = int(cv+1-np.floor((cv+1)/3)*3)
                c2 = int(cv+2-np.floor((cv+2)/3)*3)
                x1 = xn[tri[cid[c],c1]]
                y1 = yn[tri[cid[c],c1]]
                x2 = xn[tri[cid[c],c2]]
                y2 = yn[tri[cid[c],c2]]
                AB = np.array([x1-xx, y1-yy])
                BC = np.array([x2-x1, y2-y1])
                CA = np.array([xx-x2, yy-y2])
                lAB   = np.sqrt(np.square(AB[0,:])+np.square(AB[1,:]))
                if np.any(lAB==0.0):
                    ii = np.where(lAB==0.0)
                    lAB[ii] = 1.0
                lBC   = np.sqrt(np.square(BC[0])+np.square(BC[1]))
                if np.any(lBC==0.0):
                    ii = np.where(lBC==0.0)
                    lBC[ii] = 1.0
                lCA   = np.sqrt(np.square(CA[0,:])+np.square(CA[1,:]))
                if np.any(lCA==0.0):
                    ii = np.where(lCA==0.0)
                    lCA[ii] = 1.0
                atv = (-AB[0,:]*CA[0,:] - AB[1,:]*CA[1,:]) / (lAB*lCA)
                atv[atv<-1] = -1.
                atv[atv>1] = 1.
                tv = np.arccos(atv)
                at1 = (-AB[0,:]*BC[0] - AB[1,:]*BC[1]) / (lAB*lBC)
                at1[at1<-1] = -1.
                at1[at1>1] = 1.
                t1 = np.arccos(at1)
                at2 = (-CA[0,:]*BC[0] - CA[1,:]*BC[1]) / (lCA*lBC)
                at2[at2<-1] = -1.
                at2[at2>1] = 1.
                t2 = np.arccos(at2)
                R = R + np.square(tv-t60) + np.square(t1-t60) + np.square(t2-t60)
            else:
                R = R + np.ones(R.shape) * (np.square(theta[0,cid[c]]-t60) + np.square(theta[1,cid[c]]-t60) + np.square(theta[2,cid[c]]-t60))

        ind = np.where(R==np.min(R))
        ind = ind[0]
        xv = xx[ind]
        yv = yy[ind]
        Rv = np.min(R)

        return xv, yv, Rv

        
def neighbors(j, tri):
    '''Finds all cells that is sharing nodes with cell j.'''
    nid = tri[j,:]
    tmp = np.arange(len(tri))
    tmp2 = np.empty(0)
    for t in np.arange(3):
        tmp2 = np.append(tmp2, tmp[np.any(tri==nid[t], axis=1)])
    cid = np.unique(tmp2).astype(int)
    return cid

def write_mesh(xn, yn, tri, fname = 'mesh.txt'):
    fid = open(fname,'w')
    fid.write(str(len(xn)) + '\n')
    for n in np.arange(len(xn)):
        fid.write(str(xn[n]) + ' ' + str(yn[n]) + '\n')
    fid.write(str(len(tri)) + '\n')
    for n in np.arange(len(tri)):
        fid.write(str(tri[n,0]) + ' ' + str(tri[n,1]) + ' ' + str(tri[n,2]) + '\n')
    fid.close()

def mk2dm(xn, yn, tri, new2dm = 'data'):

	# Write the mesh to a .2dm file
	#trangs  = trangs[gtc[0],:]+1 # to get format that SMS is happy with
	newfile = new2dm+'.2dm'
	fid     = open(newfile,'w')

	# write triangulation
	for i in range(len(tri[:,0])):
		fid.write('E3T '+str(i+1)+' '+str(tri[i,0])+' '+str(tri[i,1])+\
				' '+str(tri[i,2])+' 1\n')

	# write node positions
	for i in range(len(xn)):
		fid.write('ND '+str(i+1)+' '+str(xn[i])+' '+str(yn[i])+' 0.00000001\n')

	fid.close()



def plot_mesh(xn, yn, t, minangle = 35.):

    plt.figure()
    plt.triplot(np.squeeze(xn), 
                np.squeeze(yn), 
                t, 
                'g-', markersize=0.2, linewidth=0.2)
    theta = angles(xn, yn, t)
    atresh = minangle/180.*np.pi
    [i,j] = np.where(theta<atresh)
    x = np.array([xn[t[:,0]],xn[t[:,1]],xn[t[:,2]]])
    y = np.array([yn[t[:,0]],yn[t[:,1]],yn[t[:,2]]])
    plt.plot(x[:,j], y[:,j], 'r')
    plt.plot([x[0,j], x[2,j]], [y[0,j],y[2,j]], 'r')
    #plt.plot(x[theta<atresh], y[theta<atresh], 'og')
    plt.axis('equal')
    plt.show()
