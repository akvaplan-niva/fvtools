'''
Python version of the matlab scripts "mk2dm.m" and "trangles.m"  
'''
import numpy as np 
import matplotlib.pyplot as plt 

def write_2dm(datafile, cangle = 5, new2dm='data'):
	'''
	Reads a SMESHING output file and writes the output as a 2dm file
	- optional:
		cangle --> minimum acceptable angle in triangle corner
		new2dm --> name of new 2dm file (standard: new2dm='data')
	'''
	fid    = open(datafile,'r')
	i      = 0  # to know if output is node or element number
	points = [] # list for points
	trangs = [] # list for triangulation

	for line in fid:
		if len(line.split(' '))==1:
			if i == 0:	
				nodeN = int(line.split('/n')[0])
				i     = 1
				mode  = 'node'
				continue

			else:
				elemN = int(line.split('/n')[0])
				mode  = 'elem'
				continue

		if mode == 'node':
			pts = line.split(' ')
			points.append(list(map(float,pts)))

		if mode == 'elem':
			elm = line.split(' ')
			trangs.append(list(map(int,elm)))

	fid.close()

	# I find numpy arrays to be easier to work with than lists...
	points = np.array(points)
	trangs = np.array(trangs)

	# quality control part - please update with more advanced routines!!!
	theta  = trangles(points,trangs)
	plt.figure()
	plt.hist(theta.ravel())
	plt.title('Histogram of ')

	while True:
		# Keep triangles with angles greater than cangle, delete the rest
		gtc    = np.where(theta.min(axis=1)>cangle)
		plt.figure()
		plt.triplot(points[:,0],points[:,1],trangs,c='g',lw=0.2)
		plt.title('raw grid from SMESHING')
		plt.axis('equal')

		plt.figure()
		plt.triplot(points[:,0],points[:,1],trangs[gtc[0],:],c='g',lw=0.2)
		plt.title('modified grid after removing angles less than '+str(cangle))
		plt.axis('equal')
		plt.show()

		print(' ---------------------- ')
		print('Old number of triangles: '+str(len(trangs[:,0])))
		print('New number of triangles: '+str(len(gtc[0])))
		print(' ')
		if input('good enough? y/[n] ')=='y':
			break
		cangle = float(input('enter the new critical angle'))
		print(' ')

	# Write the mesh to a .2dm file
	trangs  = trangs[gtc[0],:]+1 # to get format that SMS is happy with
	newfile = new2dm+'.2dm'
	fid     = open(newfile,'w')

	# write triangulation
	for i in range(len(trangs[:,0])):
		fid.write('E3T '+str(i+1)+' '+str(trangs[i,0])+' '+str(trangs[i,1])+\
				' '+str(trangs[i,2])+' 1\n')

	# write node positions
	for i in range(len(points[:,0])):
		fid.write('ND '+str(i+1)+' '+str(points[i,0])+' '+str(points[i,1])+' 0.00000001\n')

	fid.close()

def trangles(p,t):
	'''
	Reads points and triangulations from SMESHING and removes triangles 
	with too small angles
	    C
	   / \
	  /___\
	 A     B
	Simply solves the equation cos(theta) = a*b/|a||b|
	'''
	xn    = p[:,0]; yn = p[:,1]
	x     = np.array([xn[t[:,0]],xn[t[:,1]],xn[t[:,2]]])
	y     = np.array([yn[t[:,0]],yn[t[:,1]],yn[t[:,2]]])

	# cos(theta) = a * b / |a||b|
	# Store as vectors
	AB    = np.array([x[0,:]-x[1,:], y[0,:]-y[1,:]])
	BC    = np.array([x[1,:]-x[2,:], y[1,:]-y[2,:]])
	CA    = np.array([x[2,:]-x[0,:], y[2,:]-y[0,:]])

	# Get length of each triangle side
	lAB   = np.sqrt(AB[0,:]**2+AB[1,:]**2)
	lBC   = np.sqrt(BC[0,:]**2+BC[1,:]**2)
	lCA   = np.sqrt(CA[0,:]**2+CA[1,:]**2)
	a     = [lAB.min(), lBC.min(), lCA.min()]
	b     = [lAB.max(), lBC.max(), lCA.max()]
	print('Minimum triangle wall length (resolution): '+ str(min(a)))
	print('Maximum triangle wall length (resolution): '+ str(max(b)))

	# Get dot products
	ABAC  = -(AB[0,:]*CA[0,:]+AB[1,:]*CA[1,:])
	BABC  = -(AB[0,:]*BC[0,:]+AB[1,:]*BC[1,:])
	CABC  = -(CA[0,:]*BC[0,:]+CA[1,:]*BC[1,:])

	# Get the angle (in degrees)
	theta = np.array([np.arccos(ABAC/(lAB*lCA)), \
                  	np.arccos(BABC/(lAB*lBC)), \
                  	np.arccos(CABC/(lCA*lBC))])

	theta = theta*360.0/(2*np.pi) # radians to degrees
        
	# Find number of corners < 35*
	th_raveled = np.ravel(theta)
	gt35       = np.where(th_raveled<=35.0)
	print('There are ' + str(len(th_raveled[gt35])) + ' corners less than 35 degrees in this mesh')

	fig, ax = plt.subplots(2,1)
	ax[0].hist(lAB.ravel(),bins=40)
	ax[0].set_title('Histogram of mesh resolution')
	ax[0].set_xlabel('resolution')
	ax[0].set_ylabel('# triangle corners')

	ax[1].hist(theta.ravel(),bins=40)
	ax[1].set_title('Histogram of triangle corner angles')
	ax[1].set_xlabel('angle')
	ax[1].set_ylabel('# triangle corners')
	
	return theta.T
