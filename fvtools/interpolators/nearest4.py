import numpy as np
import matplotlib.pyplot as plt
import progressbar as pb
from scipy.spatial import cKDTree as KDTree
from numba import njit

class Nearest4Points:
    '''
    Work in progress, not used yet...

    Builder for nearest 4 points, which is needed when preparing to use the N4 bi-linear coefficient subset
    Will generalize building of nearest 4- indices for AROME and ROMS input.

    xy_source        = (n,2) x,y
    xy_source_center = (m,2) x,y
    xy_fvcom         = (i,2) array: Position of FVCOM points

    The source points (n) are vertices in a structured grid, where the center points (m) are in the center of n-boxes.
        |        |
        |        |
    --- n ------ n ---
        |        |
        |   m    |
        |        |
    --- n ------ n ---
        |        |
        |        |

    We find the m-point that is closest to the FVCOM point, and use that to find the "box" that out FVCOM point is within.
    '''
    @property
    def source_tree(self):
        return KDTree(self.xy_source)

    @property
    def source_center_tree(self):
        return KDTree(self.xy_source_center)

    @property
    def center_closest_to_fvcom(self):
        _, center_index = self.source_center_tree.query(self.xy_fvcom)
        return center_index

    @property
    def ball_radius(self):
        dst = np.sqrt((self.xy_source[:,0] - self.xy_source_center[0,0])**2 + (self.xy_source[:,1] - self.xy_source_center[0,1])**2)
        return 1.3*dst[dst.argsort()[0]] # 1.3 was arbitrary, but turns ot to make sure that we only find the points we need

    @property
    def nearest4inds(self):
        return self.source_tree.query_ball_point(self.source_center_tree.data[self.center_closest_to_fvcom], r = self.ball_radius)

class N4(Nearest4Points):
    '''
    Series of staticmethods, ROMS and AROME interpolators use these to some extent
    '''
    def get_interpolation_matrices(self, xy_source=None, xy_source_center=None, xy_fvcom=None, nfvcom = None, widget_title = None):
        '''
        Find the 4 source indices around this fvcom point, compute bi-linear interpolation coefficients to get *there*
        '''
        assert xy_source is not None
        assert xy_source_center is not None
        assert xy_fvcom is not None

        self.xy_source = xy_source
        self.xy_source_center = xy_source_center
        self.xy_fvcom = xy_fvcom

        print(f'  - Finding nearest 4 {widget_title} weights')
        # - we can't send objects to numba nopython mode, hence we must convert the np.object array to a np.ndarray
        n4indices = np.array([np.array(nearest4, dtype=np.int64) for nearest4 in self.nearest4inds], dtype=np.int64)
        return compute_interpolation_coefficients(n4indices, self.xy_source, xy_fvcom, len(xy_fvcom[:,0]))

@njit
def compute_interpolation_coefficients(source_indices, source_points, xy_fvcom, nfvcom):
    '''
    Compute interpolation coefficients for the grid points. There is a small bias, but its acceptable
    '''
    # Initialize weight and node matrices
    index   = np.zeros((nfvcom, 4), dtype = np.int64)
    weight  = np.zeros((nfvcom, 4), dtype = np.float64)

    # Find the interpolation coefficients
    for k, (x, y) in enumerate(zip(xy_fvcom[:,0], xy_fvcom[:,1])):
        index[k, :]  = source_indices[k]
        weight[k, :] = _bilinear_coefficients(source_points[source_indices[k], 0], source_points[source_indices[k], 1], x, y)
    return index, weight

@njit
def _bilinear_coefficients(source_x, source_y, fvcom_x, fvcom_y):
    '''
    Finds the bilinear interpolation coefficients for a 4 cornered box following http://en.wikipedia.org/wiki/Bilinear_interpolation
    '''
    # Rotate the coordinate system to get a straight system
    source_x, source_y, fvcom_x, fvcom_y = _rotate_subset(source_x, source_y, fvcom_x, fvcom_y)

    # Identify the "lowest line"
    args           = np.argsort(source_y)
    lower          = args[0:2]
    upper          = args[2:]

    coef           = np.zeros((4,1), dtype = np.float64)
    denominator    = (np.max(source_x) - np.min(source_x)) * (np.max(source_y) - np.min(source_y))

    coef[lower[0]] = np.abs((source_x[lower[1]]-fvcom_x) * (source_y[upper[0]]-fvcom_y))/denominator
    coef[lower[1]] = np.abs((source_x[lower[0]]-fvcom_x) * (source_y[upper[0]]-fvcom_y))/denominator
    coef[upper[0]] = np.abs((source_x[upper[1]]-fvcom_x) * (source_y[lower[0]]-fvcom_y))/denominator
    coef[upper[1]] = np.abs((source_x[upper[0]]-fvcom_x) * (source_y[lower[0]]-fvcom_y))/denominator
    coef           = coef/np.sum(coef)
    return coef[:, 0]

@njit
def _rotate_subset(source_x, source_y, fvcom_x, fvcom_y):
    '''
    Since bilinear coefficients is particular about squares being square and straight
    '''
    # Get the corner to rotate around
    first_corner       = np.where(source_y == np.min(source_y))

    # Get the angle to rotate
    dist               = np.sqrt((source_x - source_x[first_corner])**2 + (source_x-source_x[first_corner])**2)
    x_tmp1             = source_x[np.argsort(dist)[1:]]
    first_to_the_right = x_tmp1[np.where(x_tmp1 == np.max(x_tmp1))]
    second_corner      = np.where(source_x == first_to_the_right)
    angle              = np.arctan2(source_y[second_corner] - source_y[first_corner], \
                                    source_x[second_corner] - source_x[first_corner])

    # Rotate around origo
    x_tmp = (source_x - source_x[first_corner])  
    y_tmp = (source_y - source_y[first_corner])
    fx_t  = (fvcom_x  - source_x[first_corner])
    fy_t  = (fvcom_y  - source_y[first_corner])

    # Rotate source coordinates
    source_x = x_tmp*np.cos(-angle) - y_tmp*np.sin(-angle)
    source_y = x_tmp*np.sin(-angle) + y_tmp*np.cos(-angle)

    # Rotate FVCOM
    fvcom_x = fx_t*np.cos(-angle) - fy_t*np.sin(-angle)
    fvcom_y = fx_t*np.sin(-angle) + fy_t*np.cos(-angle)
    return source_x, source_y, fvcom_x, fvcom_y

# debug, just used to show that the interpolation scheme works
# ---
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