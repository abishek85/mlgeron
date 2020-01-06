#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 11:40:24 2019

@author: abishekk
"""

# common imports
import os
import tarfile
import pandas as pd
import numpy as np
from six.moves import urllib

"""
Download data from given url into specified directory
"""
def fetch_data_from_url(data_url, data_dir):
    # create directory if it does not exist
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)
        
    # download tar file
    # TODO: add if check
    ip_tarfile = os.path.join(data_dir, "data.tgz")
    urllib.request.urlretrieve(data_url, ip_tarfile)
    
    # extract tar if needed
    tardata = tarfile.open(ip_tarfile)
    tardata.extractall(data_dir)
    tardata.close()

"""
Load CSV  data using Pandas
"""
def load_csv_data(data_dir, filename):
        
    return pd.read_csv(os.path.join(data_dir, filename))

"""
Create test/train splits
"""
def create_test_train_splits(data, test_ratio):
    
    np.random.seed(42)
    
    # total num of samples
    num_data_pts = len(data)
    
    # list of shuffled indices
    shuffled_ind = np.random.permutation(num_data_pts)
    
    # fraction to set aside for testing
    num_test_pts = np.ceil(num_data_pts * test_ratio)
    
    test_data = shuffled_ind[:num_test_pts]
    
    train_data = shuffled_ind[num_test_pts:]
    
    return data.iloc[train_data], data.iloc[test_data]
