import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import os
import pandas


file_name = "/scratch/micheli/perfectIC-highpt-EB-training.root"
tmp = file_name.split("/")
out_dir = tmp[-2] + "_" + tmp[-1].replace("-training.root", "_test20190122_2")
filecsv = out_dir+"/summary.csv" 

df = pandas.read_csv(filecsv)
plt.plot(df['epoch'],df['test-effrms-mean'],color='blue',label='test')
plt.plot(df['epoch'],df['train-effrms-mean'],color='orange',label='train')
plt.legend(loc="upper left")
plt.savefig(out_dir+'/effrms_vs_iter.png')
plt.ylim(0.010, 0.015)
plt.savefig(out_dir+'/effrms_vs_iter_zoom.png')
plt.close()

plt.plot(df['epoch'],df['n_estimators'],color='blue',label='nestimators')
plt.legend(loc="upper left")
plt.savefig(out_dir+'/n_estimators_vs_iter.png')
plt.close()
