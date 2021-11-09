import matplotlib.pyplot as plt
import numpy as np
from statistics import variance, mean
from numpy import asarray
from numpy import arange
from numpy import meshgrid
import torch


def plots(i, samples):
    print("Program", i)
    plt.switch_backend('agg')
    
    num_samples = len(samples)

    if i in [1,2]:
        for _ in range(num_samples):
            samples = [float(x) for x in samples]
        
        plt.figure(figsize=(5,4))

        if i == 1:
            plt.xlabel("n")
        else:
            plt.xlabel("mu")

        plt.ylabel("frequency")
        plt.title("Histogram program " + str(i))
        plt.hist(samples, bins=20)
        figstr = "histograms/program_"+str(i)
        plt.savefig(figstr)

        means = "{:.5f}".format(mean(np.array(samples, dtype=float)))  
        vars = "{:.5f}".format(variance(np.array(samples, dtype=float)))  

    elif i==3:

        for n in range(num_samples):
            samples[n] = [int(x) for x in samples[n]]

        variables = np.array(samples,dtype=object).T.tolist()

        for d in range(len(variables)):
            counts = [0,0,0]
            for element in variables[d]:
                counts[element] += 1
            plt.figure(figsize=(5,4))
            plt.bar([0,1,2],counts)
            plt.xlabel("observations["+str(d)+"]")
            plt.ylabel("frequency")
            figstr = "histograms/program_"+str(i)+"_var_"+str(d)
            state_dist = [x/num_samples for x in counts]
            print("state distribution after", d, "steps", state_dist)
            plt.savefig(figstr)
        print("\n")
        
        means = ["{:.5f}".format(mean(variables[d])) for d in range(len(variables))]
        vars = ["{:.5f}".format(variance(variables[d])) for d in range(len(variables))]

    print("mean", means)
    print("variance", vars)
    print("\n\n\n")

