"""
This is the source code for volume visualization

Computer Vision Center
Universitat Autonoma de Barcelona

__author__ = "Debora Gil, Guillermo Torres, Carles Sanchez, Pau Cano"
__license__ = "GPL"
__email__ = "debora,gtorres,csanchez,pcano@cvc.uab.es"
__year__ = "2023"
"""

# --- IMPORT PY LIBRARIES
# Python Library 2 manage volumetric data
import numpy as np
# Python standard Visualization Library
import matplotlib.pyplot as plt
# Python standard IOs Library
import os

# --- IMPORT SESSION FUNCTIONS
# --- Session Code Folder (change to your path)
SessionPyFolder = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
os.chdir(SessionPyFolder)  # Change Dir 2 load session functions
# .nii Read Data
from NiftyIO import readNifty
# Volume Visualization
from VolumeCutBrowser import VolumeCutBrowser

# --- LOAD DATA

# -- Data Folders (change to your path)
SessionDataFolder = os.path.join(SessionPyFolder, 'dataset')
os.chdir(SessionDataFolder)

CaseFolder = 'CT'
TypeFolder = 'sample'
ImageFile = 'LIDC-IDRI-0001.nii.gz'

# --- 2. 定义你想使用的掩码文件名 (这是你实际可用的掩码文件名)
MaskFile = 'LIDC-IDRI-0001_R_1.nii.gz'

# ----------------- Load Intensity Volume (CT 图像) -----------------
# 注意这里使用了 ImageFile
NiiFile = os.path.join(SessionDataFolder, TypeFolder, CaseFolder, 'image', ImageFile)
niivol, niimetada = readNifty(NiiFile)

# ----------------- Load Nodule Mask (掩码文件) -----------------
# 注意这里使用了 MaskFile
NiiFile = os.path.join(SessionDataFolder, TypeFolder, CaseFolder, 'nodule_mask', MaskFile)
niimask, niimetada = readNifty(NiiFile)


# --- VOLUME METADATA
print('Voxel Resolution (mm): ', niimetada.spacing)
print('Volume origin (mm): ', niimetada.origen)
print('Axes direction: ', niimetada.direction)
# --- VISUALIZE VOLUMES

# --- Interactive Volume Visualization
# Short Axis View
VolumeCutBrowser(niivol)
VolumeCutBrowser(niivol, IMSSeg=niimask)
# Coronal View
VolumeCutBrowser(niivol, Cut='Cor')
# Sagital View
VolumeCutBrowser(niivol, Cut='Sag')

# --- Short Axis (SA) Image
# Define SA cut
k = int(niivol.shape[2] / 2)  # Cut at the middle of the volume
SA = niivol[:, :, k]
# Image
fig1 = plt.figure()
plt.imshow(SA, cmap='gray')
plt.close(fig1)  # close figure fig1

# Cut Level Sets
levels = [400]
fig1 = plt.figure()
ax1 = fig1.add_subplot(111, aspect='equal')
ax1.imshow(SA, cmap='gray')
plt.contour(SA, levels, colors='r', linewidths=2)
plt.close("all")  # close all plt figures
