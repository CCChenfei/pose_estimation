import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import math
import pylab
import sys
import argparse
import re
from pylab import figure, show, legend, ylabel

from mpl_toolkits.axes_grid1 import host_subplot


if __name__ == "__main__":

  parser = argparse.ArgumentParser(description='makes a plot from Caffe output')
  parser.add_argument('output_file', help='file of captured stdout and stderr')
  args = parser.parse_args()
  
  f = open(args.output_file, 'r')

  training_iterations = []
  training_loss = []

  test_iterations = []
  test_loss = []
  
  #add for heatmap
  training_heatmap_loss1 = []  
  training_heatmap_loss2 = []  
  training_heatmap_loss3 = []  
 # training_fusion_loss = []
#  training_fusion2_loss = []


  check_test = False
  for line in f:

    if check_test:
      if 'Test net output #0' in line and 'loss_heatmap1' in line:
        sumloss = float(line.strip().split(' ')[-2])
      if 'Test net output #1' in line and 'loss_heatmap2' in line:
        sumloss += float(line.strip().split(' ')[-2])
      if 'Test net output #2' in line and 'loss_heatmap3' in line:
        test_loss.append(float(line.strip().split(' ')[-2])+sumloss)
        check_test = False
    if '] Iteration ' in line and 'loss = ' in line:
      arr = re.findall(r'ion \b\d+\b,', line)
      training_iterations.append(int(arr[0].strip(',')[4:]))
      training_loss.append(float(line.strip().split(' = ')[-1]))
    
    # add for heatmap
    if 'Train net output #0: loss_heatmap1' in line:
      training_heatmap_loss1.append(float(line.strip().split(' ')[-2]))
    if 'Train net output #1: loss_heatmap2' in line:
      training_heatmap_loss2.append(float(line.strip().split(' ')[-2]))
    if 'Train net output #2: loss_heatmap3' in line:
      training_heatmap_loss3.append(float(line.strip().split(' ')[-2]))

    ##
    if '] Iteration ' in line and 'Testing net' in line:
      arr = re.findall(r'ion \b\d+\b,', line)
      test_iterations.append(int(arr[0].strip(',')[4:]))
      check_test = True

  print 'train iterations len: ', len(training_iterations)
  print 'train loss len: ', len(training_loss)
  print 'heat loss1 len: ', len(training_heatmap_loss1)
  print 'heat loss2 len: ', len(training_heatmap_loss2)
  print 'heat loss3 len: ', len(training_heatmap_loss3)
  print 'test loss len: ', len(test_loss)
  print 'test iterations len: ', len(test_iterations)

  f.close()
#  plt.plot(training_iterations, training_loss, '-', linewidth=2)
#  plt.show()
  
  host = host_subplot(111)#, axes_class=AA.Axes)
  plt.subplots_adjust(right=0.75)

  par1 = host.twinx()

  host.set_xlabel("iterations")
  host.set_ylabel("log loss")
  
  p1, = host.plot(training_iterations, training_loss, label="training loss")
  p2, = host.plot(training_iterations,training_heatmap_loss1,label= "heatmap loss1")
  p3, = host.plot(training_iterations,training_heatmap_loss2, label = "heatmap loss2")
  p4, = host.plot(test_iterations, test_loss, label="test loss")
  p5, = host.plot(training_iterations,training_heatmap_loss3, label = "heatmap loss3")

  host.legend(loc=2)

  host.axis["left"].label.set_color(p1.get_color())

  plt.draw()
  plt.show()

