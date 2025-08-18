#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 07:25:04 2022

@author: dori
"""
#import matplotlib.pyplot as plt
#from mayavi import mlab
import numpy as np
from aggregation import generator, rotator, aggregate, dendrite, crystal
from aggregation import mcs
from aggregation.rotator import Rotator
import os

from extended_aggregation import Hexprism, GenericDendrite, NullRotator

def a_from_L(L):
        """Relation between different dimensions of the crystal.

        Args:            
            L: the height of the hexagonal prism.

        Returns:
            a, The side of the hexagon.
        """
        if L < 100e-6:
            return 0.35*L
        else:
            return 3.48*(L*1e6)**0.5 * 1e-6

rot = NullRotator()
grid_res=1.0e-6
rho_i = 917.6

D_vec = np.arange(10,100,5)*1e-6

D_vec = np.append(D_vec,np.arange(100,1000,100)*1e-6)
D_vec = np.append(D_vec,np.arange(1000,5500,500)*1e-6)
savePath = '../output/needle/'#'/data/optimice/aggregate_model/Jussis_aggregates/monomers/
property_file = savePath+'needle_properties.txt'
if os.path.isfile(property_file):
	file = np.loadtxt(property_file,skiprows=1)
	
	if len(file) < len(D_vec):
		create_new = True
		if file.ndim > 1:
			i_vec = np.arange(len(file),len(D_vec))
		else:
			print(len(file))
			
			if len(file)>1:
				i_vec = np.arange(1,len(D_vec))
			else:
				i_vec = np.arange(0,len(D_vec))
		print(i_vec)
	else:
		create_new = False
else:
	create_new = True
	i_vec = np.arange(0,len(D_vec))
#with open(property_file,'a+') as f:
	np.savetxt(property_file, np.vstack(("mass", "area", "Dmax","ar")).T, fmt="%s")
if create_new:
	for i in i_vec:
	#for i,D in enumerate(D_vec):
		D = D_vec[i]
		#grid_res = grid_res_vec[i]
		grid_res = 0.5*D*1e-2
		print('now at ',i+1,' of total ',len(D_vec))
		den0 = crystal.Needle(D)
		#cry = Hexprism(D, ar)
		
		gen = generator.MonodisperseGenerator(den0, rot, grid_res)
		
		agg = aggregate.Aggregate(gen)		        
	    
	    # save shapefile
		if not os.path.exists(savePath):
			os.makedirs(savePath)
		savefile = 'needle_size_{:.6e}_res_{:.2e}.txt'.format(D,grid_res)
		grid = agg.grid()
		np.savetxt(savePath+savefile, grid, fmt="%d") # in multiples of grid_res
		savefile = 'align_' + savefile
		# align and then save shapefile
		agg.align()
		grid = agg.grid()
		np.savetxt(savePath+savefile, grid, fmt="%d")

		# calculate properties and add to properties file
		mass = rho_i*agg.X.shape[0]*agg.grid_res**3
		area = agg.vertical_projected_area()
		#D_max = mcs.minimum_covering_sphere(agg.X)[1]*2
		#thickness = agg.grid_res*(1 + np.max(agg.X[:, 2]) - np.min(agg.X[:, 2]))
		aspect_ratio = agg.aspect_ratio()
		
		with open(property_file,'a+') as f:
			np.savetxt(f, np.vstack((mass, area, D,aspect_ratio**-1)).T, fmt="%.6e")
			f.close()
	#quit()


