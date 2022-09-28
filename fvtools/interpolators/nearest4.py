import numpy as np
import matplotlib.pyplot as plt
import progressbar as pb
from numba import njit

class N4:
    '''
    Series of staticmethods, ROMS and AROME interpolators use these to some extent
    '''
    def _compute_interpolation_coefficients(self, target_x, target_y, source_indices, source_points, index, weight, widget_title = 'cell'):
        '''
        Compute interpolation coefficients for the grid points. There is a small bias, but its acceptable
        '''
        widget = [f'    Finding nearest 4 {widget_title} weights: ', pb.Percentage(), pb.BouncingBar()]
        bar = pb.ProgressBar(widgets=widget, maxval=len(target_x))
        bar.start()
        for k, (x, y) in enumerate(zip(target_x, target_y)):
            bar.update(k)
            nearest_source_indices = np.array(source_indices[k])
            index[k,:]  = nearest_source_indices
            weight[k,:] = self._bilinear_coefficients(source_points[nearest_source_indices,0], source_points[nearest_source_indices,1], x, y)
        bar.finish()
        return index.astype(int), weight

    def _bilinear_coefficients(self, source_x, source_y, fvcom_x, fvcom_y):
        '''
        Finds the bilinear interpolation coefficients for a 4 cornered box following http://en.wikipedia.org/wiki/Bilinear_interpolation
        '''
        # Rotate the coordinate system to get a straight system
        source_x, source_y, fvcom_x, fvcom_y = self._rotate_subset(source_x, source_y, fvcom_x, fvcom_y)

        # Identify the "lowest line"
        # ----
        args           = np.argsort(source_y)
        lower          = args[0:2]
        upper          = args[2:]

        coef           = np.zeros((4,))
        denominator    = (max(source_x)-min(source_x))*(max(source_y)-min(source_y))
        coef[lower[0]] = np.abs((source_x[lower[1]]-fvcom_x)*(source_y[upper[0]]-fvcom_y))/denominator
        coef[lower[1]] = np.abs((source_x[lower[0]]-fvcom_x)*(source_y[upper[0]]-fvcom_y))/denominator
        coef[upper[0]] = np.abs((source_x[upper[1]]-fvcom_x)*(source_y[lower[0]]-fvcom_y))/denominator
        coef[upper[1]] = np.abs((source_x[upper[0]]-fvcom_x)*(source_y[lower[0]]-fvcom_y))/denominator
        coef           = coef/np.sum(coef)
        return coef

    @staticmethod
    @njit
    def _rotate_subset(source_x, source_y, fvcom_x, fvcom_y):
        '''
        Since bilinear coefficients is particular about squares being square and straigt
        '''
        # Get the corner to rotate around
        # ----
        first_corner       = np.where(source_y == np.min(source_y))

        # Get the angle to rotate
        # ----
        dist               = np.sqrt((source_x - source_x[first_corner])**2 + (source_x-source_x[first_corner])**2)
        x_tmp1             = source_x[np.argsort(dist)[1:]]
        first_to_the_right = x_tmp1[np.where(x_tmp1 == np.max(x_tmp1))]
        second_corner      = np.where(source_x == first_to_the_right)
        angle              = np.arctan2(source_y[second_corner] - source_y[first_corner], \
                                        source_x[second_corner] - source_x[first_corner])

        # Rotate around origo. (rotating the entire coordinate system should be better...)
        # ----
        x_tmp = (source_x - source_x[first_corner])  
        y_tmp = (source_y - source_y[first_corner])
        fx_t  = (fvcom_x  - source_x[first_corner])
        fy_t  = (fvcom_y  - source_y[first_corner])

        # Rotate source coordinates
        # ----
        source_x = x_tmp*np.cos(-angle) - y_tmp*np.sin(-angle)
        source_y = x_tmp*np.sin(-angle) + y_tmp*np.cos(-angle)

        # Rotate FVCOM
        # ----
        fvcom_x = fx_t*np.cos(-angle) - fy_t*np.sin(-angle)
        fvcom_y = fx_t*np.sin(-angle) + fy_t*np.cos(-angle)
        return source_x, source_y, fvcom_x, fvcom_y

class FindNearest4Points:
    '''
    Work in progress, not used yet...

    Builder for nearest 4 points, which is needed when preparing to use the N4 bi-linear coefficient subset
    Will generalize building of nearest 4- indices for AROME and ROMS input.
    '''
    def __init__(self, x, y, xy_fvcom = None, xy_center = None):
        '''
        Mandatory:
        ---
        x = (m,n) matrix
        y = (m,n) matrix
        xy_fvcom = (i,2) array: Position of FVCOM points

        Optional:
        ---
        xy_center: (k,2) array of points in the center of x,y boxes (not mandatory, otherwise we will find them ourself)
        '''
        self.x = x
        self.y = y

        if xy_center is not None:
            self.xy_center = xy_center

    @property
    def xy_center(self):
        if not hasattr(self, '_xy_center'):
            self._xy_center = self.get_xy_center()
        return self._xy_center

    @xy_center.setter
    def xy_center(self, var):
        self._xy_center = var
    
    @property
    def ball_radius(self):
        dst = np.sqrt((self.x - self.xy_center[0,0])**2 + (self.y - self.xy_center[0,1])**2)
        return 1.3*dst[dst.argsort()[0]] # 1.3 was arbitrary, but turns ot to make sure that we only find the points we need

    def get_xy_center(self):
        '''
        Use the input array to compute xy_center
        '''
        x_center = (self.x[:,1:]+self.x[:,:-1])/2
        x_middle_center = self.x

# debug, just used to show that the interpolation scheme works
def check_squares(fvcom_x, fvcom_y, arome_x, arome_y, coef):
    '''
    Used to check that FVCOM points are actually in squares, and to check that the routine interpolates
    structured grid data to the correct point

    example use:
    ----
    check_squares(FVCOM_grd.x, FVCOM_grd.y, N4.x_arome[N4.nindex], N4.y_arome[N4.nindex], N4.ncoef)
    '''
    for fx, fy, ax, ay, cf in zip(fvcom_x, fvcom_y, arome_x, arome_y, coef):
        plt.clf()
        plt.plot(ax, ay, 'r.')
        plt.scatter(fx, fy, c = 'b')
        xint = np.sum(ax*cf)
        yint = np.sum(ay*cf)
        dst = np.sqrt((fx-xint)**2+(fy-yint)**2)
        plt.scatter(xint, yint, c='g')
        plt.draw()
        plt.pause(0.1)
        print(f'dst = {dst}')