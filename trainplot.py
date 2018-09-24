import matplotlib.pyplot as plt
import numpy as np
import pickle as cucumber
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--filename', type=str, default = '-1')
args = parser.parse_args()
infilename = args.filename
output_filename  = infilename[10:-7] 

training_error_file = infilename
title = 'Train Error On Config '+ infilename[10:-7]

with open(training_error_file,'r') as f:
  train_error = cucumber.load(f);

train_error_fixed = train_error[0];
for i in range(1,len(train_error)):
  if (i%59):
    train_error_fixed = np.append(train_error_fixed, train_error[i]);

x_vals = range(0,len(train_error_fixed));

plt.plot(x_vals, train_error_fixed)

axes = plt.gca()
axes.set_xlim([0,len(x_vals)])
axes.set_ylim([0,100])

plt.xlabel('Training Examples (x1000)')
plt.ylabel('Error')
plt.title(title)
plt.grid(False)
plt.savefig("outputPlots/" + output_filename)
