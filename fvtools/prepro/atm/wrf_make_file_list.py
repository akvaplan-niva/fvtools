#!/global/apps/python/2.7.3/bin/python
# -*- coding: utf-8 -*-
#
# Frank Gaardsted, October 2015 



import os
from argparse import ArgumentParser
import datetime

from netCDF4 import Dataset
import netCDF4
import numpy as np
import matplotlib.mlab as mp




def parse_command_line():
    ''' Parse command line arguments'''

    parser = ArgumentParser(description='Make a file which links points in time \
                                         to files and indices in FVCOM results')
    parser.add_argument("-results", "-r",
                        help='path to FVCOM results. If there are multiple directories \
                              the paths should be separated by \
                              comma(s). Eg: results/dir1,results/dir2',
                        default=os.getcwd()) 
    parser.add_argument("-output", "-o", help="where to store the file",
                        default=os.getcwd())
    parser.add_argument("-simulation_name", "-s", help="Simulation name")
    parser.add_argument("-file_list_name", "-f", help="File list name",default='fileList.txt')
    args = parser.parse_args()

    results = args.results.split(',')
    output = args.output
    simulation_name = args.simulation_name
    file_list_name = args.file_list_name
    return results, output, simulation_name, file_list_name




def make_fileList(simulation_name, data_directories):
    ''' Make lists that link a point in time to fvcom result file 
        and index (in corresponding file). Three lists a returned:
        1: list with point in time (fvcom time: days since 1858-11-17 00:00:00)
        2: list with path to files
        3: list with indices'''
        
    # Initialize variables
    fvtime = np.empty(0)
    path = []
    index = []

    # Go through data directories and identyfy relevant data files
    for directory in data_directories:
        print
        print(directory)
       
        #files = [elem for elem in os.listdir(directory) 
         #        if (os.path.isfile(os.path.join(directory,elem)) and len(elem) == 26)]
        
        files = [elem for elem in os.listdir(directory)
                 if (simulation_name in elem)]

        #files = [elem for elem in files if \
        #        os.path.getsize(os.path.join(directory, elem)) > 7918902568]
        
        files.sort()
        #files=files[:-1]
         
        for file in files:
            print file
            nc = Dataset(os.path.join(directory, file), 'r')
            t = nc.variables['Time'][:]
            fvtime = np.append(fvtime,t)
            path.extend([os.path.join(directory, file)] * len(t))
            index.extend(range(len(t)))
        
        # Sort according to time
        sorted_data = sorted(zip(fvtime, path, index), key=lambda x: x[0])
        fvtime = [s[0] for s in sorted_data]
        path = [s[1] for s in sorted_data]
        index = [s[2] for s in sorted_data]


        # Remove overlap
        fvtime_no_overlap = np.empty(0)
        path_no_overlap = []
        index_no_overlap = []
        
        for n in range(1, len(fvtime) -1):
            if fvtime[n] > fvtime[n-1]:
                fvtime_no_overlap = np.append(fvtime_no_overlap, fvtime[n])
                path_no_overlap.append(path[n])
                index_no_overlap.append(index[n])


    return fvtime_no_overlap, path_no_overlap, index_no_overlap




def write_fileList(fvtime, path, index, file_name="fileList.txt", output_directory=os.getcwd()):
    ''' Writes lists (time, path and index) to single file'''

    file=open(os.path.join(output_directory,file_name),'w')
    for t,p,i in zip(fvtime,path,index):
        line=str(t)+'\t'+p+'\t'+str(i)+'\n'
        file.write(line)

    file.close()



def main(results, simulation_name, output=os.getcwd(), file_list_name="fileList.txt"):
    fvtime, path, index = make_fileList(simulation_name, results)
    write_fileList(fvtime, path, index, file_list_name, output)



if __name__ == '__main__':
    result_directories, output_directory, simulation_name, file_list_name = parse_command_line()
    main(result_directories, simulation_name, output_directory, file_list_name)

