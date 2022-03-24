#!/global/apps/python/2.7.3/bin/python
# -*- coding: utf-8 -*-

import os
import numpy as np
from scipy.io import loadmat


class FVCOM_grid():
    'Represents FVCOM grid'

    def __init__(self, pathToFile=os.path.join(os.getcwd(), 'M.mat')):
        '''Create FVCOM-grid object'''
        self.mfile=pathToFile
        self.add_grid_parameters(['x', 'y', 'lon', 'lat', 'h', 'xc', 'yc'])

    

    def add_grid_parameters(self, names):
        '''Read grid attributes from mfile and add them to FVCOM_grid object'''
        grid_mfile = loadmat(self.mfile)
        
        if type(names) is str:
            names=[names]
       
        for name in names:
            setattr(self, name, grid_mfile['Mobj'][0,0][name])
    


    def make_interpolation_matrices_TS(self, interpolation_depths=[-5]):
        ''' Make matrices (numpy arrays) that, when multiplied with fvcom T or S matrix, 
            interpolates data to a given depth.'''
        
        if not hasattr(self, 'siglayz'):
            self.add_grid_parameters(['siglayz'])

        for depth in interpolation_depths:
            
            distance_to_interpolation_depth = np.abs(self.siglayz - depth)
            indices_of_min_distance = np.argmin(distance_to_interpolation_depth, axis=1)
            min_distance = np.min(distance_to_interpolation_depth,axis=1)
            min_distance_z_values = self.siglayz[range(0, self.siglayz.shape[0]), indices_of_min_distance]

            mask_upper = min_distance_z_values > depth
            mask_lower = min_distance_z_values < depth
            mask_exact = min_distance_z_values == depth
            mask_unequal = min_distance_z_values != depth
            mb = self.siglayz[:,-1] > depth
            ma = self.siglayz[:,-1] <= depth
            ms = self.siglayz[:,0] < depth

            ind1 = np.zeros(indices_of_min_distance.shape)
            ind1[mask_upper] = indices_of_min_distance[mask_upper]
            ind1[mask_lower] = indices_of_min_distance[mask_lower]-1
            ind2 = ind1 + 1
            
            ind1[mask_exact]=0
            ind1[mb]=0
            ind2[mask_exact]=1
            ind2[mb]=1

            ind1 = ind1.astype(int)
            ind2 = ind2.astype(int)

            r = np.array(range(0, self.siglayz.shape[0]))
            dz = np.zeros(len(r)) 
            dz = self.siglayz[r, ind1] - self.siglayz[r, ind2]

            interp_matrix = np.zeros(self.siglayz.shape)
            
            interp_matrix[mask_upper, ind1[mask_upper]] = \
            1 - ((self.siglayz[mask_upper, indices_of_min_distance[mask_upper]] - depth) / dz[mask_upper])
            
            interp_matrix[mask_upper, ind2[mask_upper]] = \
            1 - interp_matrix[mask_upper, ind1[mask_upper]]

            interp_matrix[mask_lower, ind2[mask_lower]] = \
            1 - ((depth - self.siglayz[mask_lower, indices_of_min_distance[mask_lower]]) / dz[mask_lower])
            
            interp_matrix[mask_lower, ind1[mask_lower]] = \
            1 - interp_matrix[mask_lower, ind2[mask_lower]]

            interp_matrix[mask_exact, indices_of_min_distance[mask_exact]] = 1
            
            interp_matrix[mb, :] = np.nan
            interp_matrix[ms, 1:] = 0
            interp_matrix[ms, 0] = 1

            setattr(self, 'interpolation_matrix_TS_' + str(abs(int(depth))) + '_m', interp_matrix)
            
            
            
    def make_interpolation_matrices_uv(self, interpolation_depths=[-5]):
        '''Make matrices (numpy arrays) that, when multiplied with fvcom u or v matrix,
            interpolates data to a given depth.'''
        
        if not hasattr(self, 'siglayz_uv'):
            self.add_grid_parameters(['siglayz_uv'])
        
        for depth in interpolation_depths:

            distance_to_interpolation_depth = np.abs(self.siglayz_uv - depth)
            indices_of_min_distance = np.argmin(distance_to_interpolation_depth, axis=1)
            min_distance = np.min(distance_to_interpolation_depth,axis=1)
            min_distance_z_values = self.siglayz_uv[range(0, self.siglayz_uv.shape[0]), indices_of_min_distance]

            mask_upper = min_distance_z_values > depth
            mask_lower = min_distance_z_values < depth
            mask_exact = min_distance_z_values == depth
            mask_unequal = min_distance_z_values != depth
            mb = self.siglayz_uv[:,-1] > depth
            ma = self.siglayz_uv[:,-1] <= depth
            ms = self.siglayz_uv[:,0] < depth

            ind1 = np.zeros(indices_of_min_distance.shape)
            ind1[mask_upper] = indices_of_min_distance[mask_upper]
            ind1[mask_lower] = indices_of_min_distance[mask_lower]-1
            ind2 = ind1 + 1
            
            ind1[mask_exact]=0
            ind1[mb]=0
            ind2[mask_exact]=1
            ind2[mb]=1

            ind1 = ind1.astype(int)
            ind2 = ind2.astype(int)

            r = np.array(range(0, self.siglayz_uv.shape[0]))
            dz = np.zeros(len(r)) 
            dz = self.siglayz_uv[r, ind1] - self.siglayz_uv[r, ind2]

            interp_matrix = np.zeros(self.siglayz_uv.shape)
            
            interp_matrix[mask_upper, ind1[mask_upper]] = \
            1 - ((self.siglayz_uv[mask_upper, indices_of_min_distance[mask_upper]] - depth) / dz[mask_upper])
            
            interp_matrix[mask_upper, ind2[mask_upper]] = \
            1 - interp_matrix[mask_upper, ind1[mask_upper]]

            interp_matrix[mask_lower, ind2[mask_lower]] = \
            1 - ((depth - self.siglayz_uv[mask_lower, indices_of_min_distance[mask_lower]]) / dz[mask_lower])
            
            interp_matrix[mask_lower, ind1[mask_lower]] = \
            1 - interp_matrix[mask_lower, ind2[mask_lower]]

            interp_matrix[mask_exact, indices_of_min_distance[mask_exact]] = 1
            
            interp_matrix[mb, :] = np.nan
            interp_matrix[ms, 1:] = 0
            interp_matrix[ms, 0] = 1

            setattr(self, 'interpolation_matrix_uv_' + str(abs(int(depth))) + '_m', interp_matrix)











