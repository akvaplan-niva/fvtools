from netCDF4 import Dataset, num2date
import matplotlib.pyplot as plt
d2 = Dataset('path_to_file.nc')

plt.figure(figsize = [10,20])
speed = np.sqrt(d2['ua'][:]**2 + d2['va'][:]**2)
for t in range(len(d2['time'][:])):
    plt.quiver(d2['xc'][:], d2['yc'][:], d2['ua'][t,:], d2['va'][t,:], speed[t,:])
    plt.title(num2date(d2['time'][t], units = d2['time'].units))
    plt.axis('equal')
    plt.show()
    plt.pause(0.25)
