from bs4 import BeautifulSoup
import urllib
import numpy as np
import geopandas as gp
import datetime
import matplotlib.pyplot as plt
from numba import njit
from functools import cached_property
'''
Download total-runoff (totalavrenning) from NVEs servers. The runoff is the freshwater (psu=0) draining from
land to the ocean via rivers.


Documentation:
----
River runoff from Norwegian catchment areas are computed based on observations being extrapolated to represent catchment areas, see
the description (Norwegian): https://publikasjoner.nve.no/rapport/2012/rapport2012_39.pdf. The extrapolation is tuned to fit model results for
a reference period for the catchment, so eventhough the runoff is observed, one can say that the distribution is "model based" to some extent.

This river runoff represents the natural flow of freshwater to the coast, and does not include any damming (ref. Stein Beldring, pers. comm.)


Improves upon existing scripts in the following way:
----
- Renames variables (ca1, ca2) to their actual function (from_vassdrag, to_vassdrag). Be aware that that is misleading, the range (from_*, to_*) is not always complete
- Uses correct indexing for some catchment areas (by using actual vassdrag-id, not range(num_here), which turned out to be wrong)
- Reports on missing data, both total days without data and info about resolved days
- Compiles a seasonal runoff-pattern to fill gaps in hydrological data rather than using last available value.

Future work:
----
- Fill gaps in-between recordings using a weight function instead of season average? Using time-series forecasting (skforecasting)?
- Distribute runoff to rivers in this routine, rather that in BuildRivers (better way to conserve freshwater to the domain)?
- NVE suggests using https://hydapi.nve.no/ instead
'''


# Set number of days to extract
# ----------------------------------------------------------
def main(days = 8000, 
         kystserie_file = '/data/FVCOM/Setup_Files/Rivers/riverdata_files/Kystserier.csv',
         vassdragsomrade_file = '/data/FVCOM/Setup_Files/Rivers/riverdata_files/Nedborfelt/Nedborfelt_Vassdragsomr.shp',
         archives = [18, 2, 5],
         max_runoff = 4500):
   '''
   Download runoff from NVE servers.
     Vassdrag is Norwegian for "Catchment area", here we are downloading runoff to "vassdragsområder",
     which esentially is "catchment areas areas", hence runoff from a cluster of catchment areas.

   Input:
   ---
   days:                 days to look back 
   kystserie_file:       file referencing kystseries to download (optional input, will look to stokes/data otherwise)
   vassdragsomrade_file: file containing information about land area of each vassdragsområde
   archives to read:     5 is the best-, 2 is the second best- and 18 is the raw river runoff data published by the NVE.
   max_runoff:           maximum runoff that we accept, used as a "clip" filter to avoid ridiculous river runoff
   '''
   # Load the "vassdragsområde" as a geopandas object
   # ----
   vassdrag   = gp.read_file(vassdragsomrade_file)

   # A object referencing vassdrag to links where we download runoff
   # ----
   KystSerie  = KystSerieReader(kystserie_file)

   # Download data from the NVE server
   # ----
   Downloader = RunoffDownloader(days, 
                                 vassdrag['arealLand'].to_numpy(), 
                                 KystSerie)

   Downloader.download_all_kystseries(archives)

   # Show gaps in the downloaded data
   # ----
   show_data_availablity(Downloader.dateobj, Downloader.data, title = 'River runoff availability as downloaded')

   # Remove obvious spikes, fill gaps with season-averages
   # ----
   Filled = DataCheckerAndFiller(Downloader.dateobj, Downloader.data, max_runoff)

   # Fill gaps, show potential gaps in the treated data
   # ----
   show_data_availablity(Downloader.dateobj, Filled.data, title = 'Processed river runoff availabliy to be stored')

   # Store to riverdata.dat file
   # ----
   save_to_riverdata(Downloader.days, Filled.data)

   # Fin.

class KystSerieReader:
   '''
   Reads kystserie file
   '''
   def __init__(self, csvfile):
      '''
      Load the stations that will be used to 
      '''
      f = open(csvfile,'r', errors = 'ignore')
      f.readline() # to skip the header
      self.from_vassdrag = []; self.to_vassdrag = []; self.vassdrags_here = []

      # Read all lines in the Kystserie-file, stop reading when reaching an empty line
      while True:
         text = f.readline()
         if text == '':
            break
         self._read_Kystserie_string(text)
      f.close()

   def _read_Kystserie_string(self, text):
      '''
      Interpret the info given by this 
      '''
      # Separate the comments at the end of the line from the numbers we want
      first_split  = text.split(';;')[0]

      # Separate the numbers from each other: 
      second_split = first_split.split(';')

      # From- to vassdrag
      self.from_vassdrag.append(int(second_split[1]))
      self.to_vassdrag.append(int(second_split[2]))

      # Vassdrags included in this link
      self.vassdrags_here.append(np.array([int(vdr) for vdr in second_split[4:]]))

class RunoffDownloader:
   '''
   Download runoff for segment in Kystseries
   '''
   def __init__(self, days, vassdrag_areas, KystSerie):
      '''
      Download runoff from catchment areas

      input:
      --
      days:           number of days to download
      vassdrag_areas: land area of vassdrag
      KystSerie:      object saying which vassdrags we have in each of the links we will download
      '''
      self.days      = days
      self.data      = self._get_data()
      self.areas     = vassdrag_areas
      self.KystSerie = KystSerie

   @cached_property
   def weight(self):
      '''
      Not all vassdrag have similar areas, we therefore apply a simple weight function to spread the runoff over them.
      This array has shape (Kystserie,), and each kystserie is a list of weights for vassdrags in Kystserie.vassdrags_here[Kystserie,]
      '''
      # vassdrag_id - 1 for python indexing
      return [np.array(self.areas[vassdrag_id-1]/np.sum(self.areas[vassdrag_id-1])) for vassdrag_id in self.KystSerie.vassdrags_here]

   @cached_property
   def datenum(self):
      '''
      to ordinal which is integer days, this is sufficient for these data
      '''
      return np.array([date.toordinal() for date in self.dateobj])

   @cached_property
   def dateobj(self):
      '''
      returns time as list of datetime.datetime objects
      '''
      return [datetime.date.today() - datetime.timedelta(days=x) for x in range(self.days-1, -1, -1)]

   def _get_data(self):
      '''
      Initialize the data output-structure, set the dates.
      '''
      _data = np.nan*np.ones([self.days, 250])
      for day in range(self.days):
         _data[day, 0] = self.dateobj[day].year
         _data[day, 1] = self.dateobj[day].month
         _data[day, 2] = self.dateobj[day].day
      return _data

   def download_all_kystseries(self, archives):
      '''
      Download all kystseries
      '''
      for segment_nr in range(len(self.KystSerie.from_vassdrag)):
         print(f'- From {self.KystSerie.from_vassdrag[segment_nr]} to {self.KystSerie.to_vassdrag[segment_nr]}:')
         self.download_single_kystserie(segment_nr, archives)

   def download_single_kystserie(self, segment_nr, archives):
      '''
      Download full timeseries from a station
      ----
      - segment_nr: the segment of the KystSerie to download
      '''
      for archive in archives:
         read_file = self._get_url(self.KystSerie.from_vassdrag[segment_nr], 
                                       self.KystSerie.to_vassdrag[segment_nr], 
                                       archive)

         print(f' - Reading from: {read_file}')
         xml = XMLreader(read_file)

         self._insert_runoff_to_corresponding_times_in_data_matrix(xml, segment_nr)

      self._check_for_gaps(segment_nr)

   def _check_for_gaps(self, segment_nr):
      '''
      Report number of missing data for this Kystserie
      '''
      nans     = np.max(self.data[:, self.KystSerie.vassdrags_here[segment_nr]+2], axis = 1)
      num_nans = len(np.where(np.isnan(nans))[0])
      if num_nans == self.days:
         print(f'   No available data during the search.\n')
         return

      if num_nans > 0:
         print(f'   Missing data for {num_nans} days (out of {self.days})\n'
               +f'     - First date: {self.dateobj[np.where(~np.isnan(nans))[0][0]]}\n'
               +f'     - Last date:  {self.dateobj[np.where(~np.isnan(nans))[0][-1]]}\n')

      else:
         print(f'   No missing data, this station has a complete timeseries.\n')

   def _insert_runoff_to_corresponding_times_in_data_matrix(self, xml, segment_nr):
      if len(xml.datenum)>0:
         corresponding_times, ind_self, ind_xml = np.intersect1d(self.datenum, np.array(xml.datenum), return_indices = True)
         assert (len(ind_xml) == len(xml.runoff)), f'Unexpected error, not all data could be dumped to riverdata'

         if len(self.weight[segment_nr]) == 1:
            self.data[ind_self, self.KystSerie.vassdrags_here[segment_nr]+2] = xml.runoff*self.weight[segment_nr]
         else:
            self.data[ind_self[:,None], self.KystSerie.vassdrags_here[segment_nr][None,:]+2] = xml.runoff[:,None]*self.weight[segment_nr]

   def _get_url(self, from_vassdrag, to_vassdrag, archive):
      '''
      Read the runoff value from the link for some day and some archive.
      ---
      - from_vassdrag: from this
      - to_vassdrag:   to this (but necessarilly including all in-between)
      - archive:
         Archive: 5  has the best data
         Archive: 2  has sort of good data
         Archive: 18 has raw data
      '''
      return f'http://h-web01.nve.no/ChartServer/ShowData.aspx?req=getchart&ver=1.0&time=-{self.days-1};0&vfmt=xml&chd=ds=htsr,rt=1,da={archive},id=700.{from_vassdrag}.{to_vassdrag}.1001.0'

class XMLreader:
   '''
   Read daily river runoff from the NVE xml-servers.
   '''
   def __init__(self, url):
      self.open_xml(url)
      self.extract_data()
      self.day_month_year()

   @property
   def dateobj(self):
      '''
      is a list of datetime.datetime objects
      '''
      return [datetime.date(year, month, day) for year, month, day in zip(self.year, self.month, self.day)]
   
   @cached_property
   def datenum(self):
      '''
      to ordinal which is integer days, this is sufficient for these data
      '''
      return np.array([date.toordinal() for date in self.dateobj])
   
   def open_xml(self, url):
      url_response = urllib.request.urlopen(url)
      xml_content  = url_response.read()
      self.content = BeautifulSoup(xml_content, 'xml')

   def extract_data(self):
      self.runoff = []
      self.date  = []
      for point in self.content('Point'):
         if len(point('Value')) == 0:
            continue
         self.runoff.append(float(point('Value')[0].getText()))
         self.date.append(point('DateTime')[0].getText())
      self.runoff = np.array(self.runoff)

   def day_month_year(self):
      '''
      split day, month and year from the string
      '''
      self.day   = [int(date[3:5]) for date in self.date]
      self.month = [int(date[:2]) for date in self.date]
      self.year  = [int(date[6:10]) for date in self.date]


class DataCheckerAndFiller:
   def __init__(self, dateobj, data, max_runoff):
      '''
      Class that processes data to remove obviously bad spikes and return a "filtered" dataset
      '''
      data = self.clip_too_big(data, max_runoff)
      data = self.fill_missing_data_with_seasonal_variations(data, dateobj)
      self.data = data

   def clip_too_big(self, data, max_runoff):
      '''
      Some of the values in archive 18 are ridiculous, we therefore clip the worst of them
      '''
      vdrag = np.arange(3,len(data[0, :]))
      for vassdrag_id_plus_two in vdrag:
         nans = np.where(data[:, vassdrag_id_plus_two]>max_runoff)
         data[nans, vassdrag_id_plus_two] = np.nan
      return data 

   def fill_missing_data_with_seasonal_variations(self, data, dateobj):
      '''
      Much of the data is patchy, replace gaps with historical average of data for that day
      - keep in mind though, this is something the NVE should have actual statistics of? 
      '''
      # Find daynumber of year for each timestamp, initialize a "temp_stat" vector
      # ----
      day_num   = np.array([t.timetuple().tm_yday for t in dateobj])
      days      = np.arange(1,max(day_num)+1)

      # Loop over all timeseries, create temperature statistics for given date and replace nans with statistics
      # ----
      for vassdrag_id_plus_two in range(3, len(data[0, :])):            
         flow_stat = np.nan*np.ones((max(day_num),)) # Initialize the climatologi-structure
         for day in days:
            inds             = np.where(day_num == day)[0] # find all measured years with this day
            flow_stat[day-1] = np.nanmean(data[inds, vassdrag_id_plus_two])   # python index starts at 0, day index start at 1

         # We require that this "statistics" has the same average as the rest of the timeseries
         flow_stat = flow_stat*(np.mean(flow_stat)/np.nanmean(data[:, vassdrag_id_plus_two]))

         # replace missing data with the flow_stat
         nans = np.isnan(data[:, vassdrag_id_plus_two])
         days_to_fill = day_num[nans]
         data[nans, vassdrag_id_plus_two] = flow_stat[days_to_fill-1]

      return data

def show_data_availablity(datetime, data, title = ' '):
   '''
   plot to show data availability during period
   '''
   if not np.isnan(data).any():
      print('There are not any holes in the runoff data written to riverdata.dat :)')
   dates = np.tile(datetime, (247,1))
   slots = np.tile(np.arange(247), (len(datetime),1))
   plt.figure()
   plt.pcolor(dates.T, slots, data[:, 3:], vmax = 100000000)
   plt.title(f'{title}. Blue means data, white indicates no data')
   plt.ylabel('Vassdragsområde')
   plt.xlabel('Year')
   plt.show(block = False)

def save_to_riverdata(days, data):
   '''
   Store to same format as 
   '''
   f = open('riverdata.dat', 'w')
   for t in np.arange(days):
       for r in np.arange(3):
           f.write(str(int(data[t,r]))+' ')
       for r in np.arange(3,249):
           f.write(str(data[t,r])+' ')
       f.write(str(data[t,249])+'\n')
   f.close()

class NoAvailableData(Exception):
   pass