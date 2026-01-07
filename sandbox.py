"""
NOTE: this file is NOT for final scripts; instead it should be used for testing modules 
      or blocks of code whithin the CovNNet file structure. DO NOT write code in
      this script that will effect other scripts/other scripts will rely on.
      Essentually, this script should be used as the equivilent of scrap paper.
"""
"""
imports
----------------------------------------------------------------------------------------------------------------------------------------"""
import numpy as np
import matplotlib.pyplot as plt

import h5py, os, optuna, torch
from scipy.spatial import KDTree
from scipy.optimize import curve_fit
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch_geometric as torchg
import torch_scatter as torchs
import random

"""
SANDBOX
----------------------------------------------------------------------------------------------------------------------------------------"""