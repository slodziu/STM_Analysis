# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 14:35:16 2024

Code to calculate dlnI-dlnV graphs from sample data
"""

import csv
import matplotlib.pyplot as plt
import numpy as np

def extract_csv_columns(file_path):
    first_column = []
    second_column = []
    with open(file_path, mode='r') as file:
        csv_reader = csv.reader(file, delimiter=';')
        for i, row in enumerate(csv_reader):
            if i == 0:
                first_column = [float(value) for value in row]
            elif i == 1:
                second_column = [float(value) for value in row]
            else:
                break  
                
    return first_column, second_column

file_path = 'RawData\IV_Spectra\spectrum6.csv'  
voltages, currents = extract_csv_columns(file_path)
lnvolt = np.log(voltages)
lncur = np.log(currents)
derivatives = np.gradient(lncur/ lnvolt)

plt.figure(figsize=(8, 6))
plt.plot(voltages, currents, marker='o', linestyle='none', color='b')
plt.xlabel('V, V')
plt.ylabel('I, A')
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(voltages, derivatives, marker='o', linestyle='none', color='r')
plt.xlabel('V, V')
plt.ylabel('d(lnI) / d(lnV)')
plt.grid(True)
plt.show()
