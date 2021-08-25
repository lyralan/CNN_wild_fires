#!/usr/bin/env python
# coding: utf-8

# In[9]:


from u import *
import PIL
PIL.Image.MAX_IMAGE_PIXELS = 933120000
from PIL import Image


# In[10]:


Base = Path('data')


# In[76]:


fire = Base / 'california_fire_perimeters.kml'
import xml.etree.ElementTree as ET
tree = ET.parse(fire)
root = tree.getroot()


# # fire

# In[85]:


fire = {}
for child in root[0][1][1:]:
#     print(child.tag, child.attrib)
    area = None
    month = None
    for attr in child[1][0]:
        if attr.attrib['name'] == 'GIS_ACRES':
            area = float(attr.text)
            break
    for attr in child[1][0]:
        if attr.attrib['name'] == 'ALARM_DATE':
            month = int(attr.text.split('/')[1])
            break
    if area == None:
        continue
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
    fire[(lon, lat)] = (area, month) # acre


# In[86]:


location = np.array(list(fire.keys()))
min_lon, min_lat = location.min(axis=0)
max_lon, max_lat = location.max(axis=0)


# In[87]:


minx = int(min_lon * 120 + 120 * 180) - 50
maxx = int(max_lon * 120 + 120 * 180) + 1 + 50
maxy = int(90 * 120 - min_lat * 120) + 1 + 50
miny = int(90 * 120 - max_lat * 120) - 50


# # meteorology

# In[7]:


from multiprocessing import Pool
def func(path):
    print(path)
    name = path.replace('tif','npy')
    if name.exists():
        return
    im = Image.open(path)
    im = np.array(im)
    ca = im[miny:maxy, minx:maxx]
    with open(name, 'wb') as f:
        np.save(f, ca)


# In[16]:


files = (Base /'worldclim').glob('*/*.tif')
# with Pool(48) as pool:
#     list(pool.imap(func, files))
# for file in files:
#     func(file)


# In[70]:


stacked = {}
for mon in range(1, 13):
    files = sorted((Base /'worldclim').glob(f'*/*{mon:02d}.npy'))
    stack = []
    for file in files:
        x = np.load(file)
        if file._up._name == 'temperature_average':
            x[x == x.min()] = 0
            x = x / 25
        elif file._up._name == 'solar':
            x[x == x.max()] = 0
            x = x / x.max()
        elif file._up._name in ['vapor', 'wind', 'precipitation']:
            x[x == x.min()] = 0
            x = x / x.max()
        stack.append(x)
    stack = np.array(stack)
    stacked[mon] = stack


# In[98]:


inputs = []
outputs = []
for (lon, lat), (area, month) in fire.items():
    x = int(lon * 120 + 120 * 180)
    y = int(120 * 90 - lat * 120)
    if month != None:
        c_lat = y - miny
        c_lon = x - minx
        inputs.append(stacked[month][:, c_lat - 25 : c_lat + 26, c_lon - 25 : c_lon + 26])
        outputs.append(area)
inputs = np.array(inputs)
outputs = np.array(outputs)


# In[102]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(inputs, outputs, test_size=0.1, random_state=42)


# In[110]:


import torch
from torch import nn
import torch.nn.functional as F


# In[111]:


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.convs = nn.Sequential(
            nn.Conv2d(5, 32, 4, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(12, stride=None)
        )
        self.fc = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 1))

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        embedding = self.convs(x)
        return self.fc(embedding.squeeze(2).squeeze(2)).squeeze(1)


# In[120]:


x_train_concat = np.concatenate([x_train, x_train[:, :, ::-1, :]])
y_train_concat = np.concatenate([y_train, y_train])


# In[121]:


x_train_concat = np.concatenate([x_train_concat,
                                 np.rot90(x_train_concat, 1, axes=(2,3)),
                                 np.rot90(x_train_concat, 2, axes=(2,3)),
                                 np.rot90(x_train_concat, 3, axes=(2,3))])


# In[122]:


y_train_concat = np.concatenate([y_train_concat, y_train_concat, y_train_concat, y_train_concat])


# In[128]:


net = ConvNet().cuda()


# In[130]:


optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)


# In[139]:


"""
for i in range(n_steps):
    Sample a batch of size n_batch x and y from the training set
    Convert x_batch and y_batch to torch.tensor
    feed x_batch_tensor into net to get y_pred_batch
    compute (MSE) loss between y_pred_batch and y_batch
    compute gradient for the net parameters given the loss
    optimizer update the parameters given the gradient
"""

def evaluate(net, x_test, y_test):
    with torch.no_grad():
        x_test_tensor = torch.tensor(x_test, dtype=torch.float32, device='cuda:0')
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32, device='cuda:0')
        y_pred = net(x_test_tensor)
        loss = F.mse_loss(y_pred, y_test_tensor)
        return loss

n_steps = 1000
n_eval_steps = 50
n_batch = 10000
for i in range(n_steps):
    indices = np.random.choice(len(x_train_concat), size=n_batch)
    x_batch = x_train_concat[indices]
    y_batch = y_train_concat[indices]
    x_batch_tensor = torch.tensor(x_batch, dtype=torch.float32, device='cuda:0')
    y_batch_tensor = torch.tensor(y_batch, dtype=torch.float32, device='cuda:0')
    y_batch_pred = net(x_batch_tensor)
    loss = F.mse_loss(y_batch_pred, y_batch_tensor)
    optimizer.zero_grad() # clear previous gradient
    loss.backward() # back propagation to get the gradient
    optimizer.step() # optimizer applies gradient to the parameters

    if (i + 1) % n_eval_steps == 0:
        loss = evaluate(net, x_test, y_test)
        print(f'Step {i+1} validation loss = {loss}', flush=True)


# In[ ]:




