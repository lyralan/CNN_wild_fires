#!/usr/bin/env python
# coding: utf-8

# In[2]:


from u import *
import PIL
PIL.Image.MAX_IMAGE_PIXELS = 933120000
from PIL import Image


# In[29]:


Base = Path('data')


# In[21]:


# plt.imshow(im[::20,::20])


# In[32]:


fire = Base / 'california_fire_perimeters.kml'
import xml.etree.ElementTree as ET
tree = ET.parse(fire)
root = tree.getroot()


# In[76]:


fire = {}
for child in root[0][1][1:]:
#     print(child.tag, child.attrib)
    for attr in child[1][0]:
        if attr.attrib['name'] == 'GIS_ACRES':
            break
    area = float(attr.text)
    cord = child[2][0][0][0][0].text
    cord = cord.split(' ')
    
    lon_list = []
    lat_list = []
    for x in cord:
        lon, lat = map(float, x.split(','))
        lon_list.append(lon)
        lat_list.append(lat)
    lon = np.mean(lon_list)
    lat = np.mean(lat_list)
    fire[(lon, lat)] = area # acre


# In[84]:


location = np.array(list(fire.keys()))
min_lon, min_lat = location.min(axis=0) 
max_lon, max_lat = location.max(axis=0)


# In[141]:


minx = int(min_lon * 120 + 120 * 180) - 50
maxx = int(max_lon * 120 + 120 * 180) + 1 + 50
maxy = int(90 * 120 - min_lat * 120) + 1 + 50
miny = int(90 * 120 - max_lat * 120) - 50


# In[144]:


mapped_fire = {}
for (lon, lat), area in fire.items():
    x = int(lon * 120 + 120 * 180) 
    y = int(120 * 90 - lat * 120) 
    mapped_fire[(x - minx, y - miny)] = area


# In[157]:


from multiprocessing import Pool
def func(path):
    im = Image.open(path)
    im = np.array(im)
    ca = im[miny:maxy, minx:maxx]
    name = path.replace('tif','npy')
    with open(name, 'wb') as f:
        np.save(f, ca)


# In[158]:


files = (Base /'worldclim').glob('*/*.tif')
with Pool(48) as pool:
    list(pool.imap(func, files))

