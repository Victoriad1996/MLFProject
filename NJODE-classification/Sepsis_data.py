import csv
import math
import os
import pathlib
import tarfile
import numpy as np
import torch
import urllib.request
import zipfile
import pandas as pd
from torchvision.datasets.utils import download_url

import common

class Sepsis(object):
    here = pathlib.Path(__file__).resolve().parent

    base_base_loc = here / 'data'
    base_loc = base_base_loc / 'sepsis'
    loc_Azip = base_loc / 'training_setA.zip'
    loc_Bzip = base_loc / 'training_setB.zip'

    #def __init__(self, root, train=True, download=False, n_samples=None, device=torch.device("cpu")):
    def __init__(self, download=False, n_samples=None, device=torch.device("cpu")):


        #self.root = root
        #self.train = train
        #self.reduce = "average"

        if download:
            self.download()



    def download(self):
        if not os.path.exists(self.loc_Azip):
            if not os.path.exists(self.base_base_loc):
                os.mkdir(self.base_base_loc)
            if not os.path.exists(self.base_loc):
                os.mkdir(self.base_loc)
            urllib.request.urlretrieve('https://archive.physionet.org/users/shared/challenge-2019/training_setA.zip',
                                       str(self.loc_Azip))
            urllib.request.urlretrieve('https://archive.physionet.org/users/shared/challenge-2019/training_setB.zip',
                                       str(self.loc_Bzip))

            with zipfile.ZipFile(self.loc_Azip, 'r') as f:
                f.extractall(str(self.base_loc))
            with zipfile.ZipFile(self.loc_Bzip, 'r') as f:
                f.extractall(str(self.base_loc))
            for folder in ('training', 'training_setB'):
                for filename in os.listdir(self.base_loc / folder):
                    if os.path.exists(self.base_loc / filename):
                        raise RuntimeError
                    os.rename(self.base_loc / folder / filename, self.base_loc / filename)


    def process_data(self):
        patients = []
        total = 0
        X_times = []
        X_static = []
        y = []
        counter = 0
        for filename in os.listdir(self.base_loc):
            if counter%1000 == 0: print(counter)
            if filename.endswith('.psv'):
                with open(self.base_loc / filename) as file:
                    time = []
                    label = 0.0
                    reader = csv.reader(file, delimiter='|')
                    reader = iter(reader)
                    next(reader)  # first line is headings
                    prev_iculos = 0
                    for line in reader:
                        assert len(line) == 41
                        *time_values, age, gender, unit1, unit2, hospadmtime, iculos, sepsislabel = line
                        iculos = int(iculos)
                        if iculos > 72:  # keep at most the first three days
                            break
                        for iculos_ in range(prev_iculos + 1, iculos):
                            time.append([float('nan') for value in time_values])
                        prev_iculos = iculos
                        time.append([float(value) for value in time_values])
                        label = max(label, float(sepsislabel))
                    unit1 = float(unit1)
                    unit2 = float(unit2)
                    unit1_obs = not math.isnan(unit1)
                    unit2_obs = not math.isnan(unit2)
                    if not unit1_obs:
                        unit1 = 0.
                    if not unit2_obs:
                        unit2 = 0.
                    hospadmtime = float(hospadmtime)
                    if math.isnan(hospadmtime):
                        hospadmtime = 0.  # this only happens for one record
                    static = [float(age), float(gender), unit1, unit2, hospadmtime]
                    if len(time) > 2:
                        X_times.append(time)
                        X_static.append(static)
                        y.append(label)
                counter += 1
        final_indices = []
        for time in X_times:
            final_indices.append(len(time) - 1)
        maxlen = max(final_indices) + 1
        for time in X_times:
            for _ in range(maxlen - len(time)):
                time.append([float('nan') for value in time_values])

        X_times = torch.tensor(X_times)
        X_static = torch.tensor(X_static)
        y = torch.tensor(y)

        final_indices = torch.tensor(final_indices)
        #record_id = torch.linspace(1,120000,1)
        times = torch.linspace(1, X_times.size(1), X_times.size(1))
        mask = []
        for i in range(X_times.size()[0]):
            m2 =[]
            for j in range(X_times.size()[1]):
                m1 = pd.DataFrame(X_times[i,j,])
                m1 = (m1.notnull()).astype('int')
                m1.columns =['v']
                m1 = m1['v'].to_list()
                m2.append(m1)

            mask.append(m2)

        mask = torch.tensor(mask)
        for i in range(X_times.size()[0]):
            x = X_times[i,:,:]
            m = mask[i,:,:]
            yy = y[i]
            patients.append((i,times,x,m,yy))
        #check dir and create it if not exists
        if not os.path.isdir(self.base_loc / 'processed_data'):
            print("Dir not here. Creating it")
            os.mkdir(self.base_loc / 'processed_data')

        torch.save(patients, self.base_loc / 'processed_data' / 'data.pt')



if __name__ == "__main__":
    #put all your tests here
    sepsis = Sepsis()
    sepsis.download()
    sepsis.process_data()
    print("test")