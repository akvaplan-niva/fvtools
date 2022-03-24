
from lxml import etree
from io import StringIO
import urllib
import numpy as np

days = 2633
ca1 = 1
ca2 = 0
ind = 3
nca =1
data = np.zeros([days,250])
nodata = np.zeros([days,250])

f = open('Kystserier.csv','r')
f.readline()
text = f.readline()
spl1 = text.split(';;')
spl2 = spl1[0].split(';')
ca1 = int(spl2[1])
ca2 = int(spl2[2])
nca = int(spl2[3])
ind = np.zeros(nca)
for n in np.arange(nca):
   ind[n] = int(spl2[n+4])
ind = ind.astype(int)+2
    
#For the first Kystserie there is only data in archive da=18
url = 'http://h-web01.nve.no/ChartServer/ShowData.aspx?req=getchart&ver=1.0&time=-' + str(days) + ';0&vfmt=xml&chd=ds=htsr,rt=1,da=18,id=700.' + str(ca1) + '.' + str(ca2) + '.1001.0'

tree = etree.parse(urllib.urlopen(url))
root = tree.getroot()
#value = 10.0
for d in np.arange(1, days+1):
    #defstr = np.zeros(250)
    dstr = root[0][d][0].text
    month = int(dstr[0:2])
    day = int(dstr[3:5])
    year = int(dstr[6:10])

    # Time
    data[d-1,0] = year
    data[d-1,1] = month
    data[d-1,2] = day
    # Data
    try:
       value = float(root[0][d][1].text)/nca
    except:
       nodata[d-1,ind] = 1
    data[d-1,ind] = value

#For the rest of the Kystserie we try all archives da= 5 ("sekundaerkontrollerte data"),2 ("primaerkontrollerte data") and 18 ("sanntidsdata").
for v in np.arange(2,69+1):
    text = f.readline()
    spl1 = text.split(';;')
    spl2 = spl1[0].split(';')
    ca1 = int(spl2[1])
    ca2 = int(spl2[2])
    nca = int(spl2[3])
    ind = np.zeros(nca)
    for n in np.arange(nca):
        ind[n] = int(spl2[n+4])
    ind = ind.astype(int)+2
    print('Vassdrag no: ' + str(ind-2))
    #url = 'http://h-web01.nve.no/ChartServer/ShowData.aspx?req=getchart&ver=1.0&time=-' + str(days) + ';0&vfmt=xml&chd=ds=htsr,rt=1,da=5,id=700.' + str(ca1) + '.' + str(ca2) + '.1001.0'

    #tree = etree.parse(urllib.urlopen(url))
    #root = tree.getroot()

    for d in np.arange(1, days+1):
        # Data

        try:
           url = 'http://h-web01.nve.no/ChartServer/ShowData.aspx?req=getchart&ver=1.0&time=-' + str(days) + ';0&vfmt=xml&chd=ds=htsr,rt=1,da=5,id=700.' + str(ca1) + '.' + str(ca2) + '.1001.0'
           tree = etree.parse(urllib.urlopen(url))
           root = tree.getroot()
           value = float(root[0][d][1].text)/nca
           print('Using Archive da=5')           
        except:
           try:
              url = 'http://h-web01.nve.no/ChartServer/ShowData.aspx?req=getchart&ver=1.0&time=-' + str(days) + ';0&vfmt=xml&chd=ds=htsr,rt=1,da=2,id=700.' + str(ca1) + '.' + str(ca2) + '.1001.0'
              tree = etree.parse(urllib.urlopen(url))
              root = tree.getroot()
              value = float(root[0][d][1].text)/nca 
              print('Using Archive da=2')
           except:
              try:
                 url = 'http://h-web01.nve.no/ChartServer/ShowData.aspx?req=getchart&ver=1.0&time=-' + str(days) + ';0&vfmt=xml&chd=ds=htsr,rt=1,da=18,id=700.' + str(ca1) + '.' + str(ca2) + '.1001.0'
                 tree = etree.parse(urllib.urlopen(url))
                 root = tree.getroot()
                 value = float(root[0][d][1].text)/nca
                 print('Using Archive da=18')
              except:
                 nodata[d-1,ind] = 1
                 print('No data in Archives 5,2,18')
        #OBS: should interpolate value where nodata=1, currently just set to previous "value"
        data[d-1,ind] = value

f.close()

#Writing to file
f = open('riverdata.dat', 'w')
for t in np.arange(days):
    for r in np.arange(3):
        f.write(str(int(data[t,r]))+' ')
    for r in np.arange(3,249):
        f.write(str(data[t,r])+' ')
    f.write(str(data[t,249])+'\n')
f.close()
