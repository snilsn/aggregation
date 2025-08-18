#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 07:25:04 2022

@author: dori
"""
#import matplotlib.pyplot as plt
#from mayavi import mlab
import numpy as np
from aggregation import generator, rotator, aggregate, dendrite, crystal, riming_runs
from aggregation import mcs
from aggregation.rotator import Rotator
import os
class NullRotator(Rotator):
    def rotate(self, X):
        return X

dataPath = '/data/optimice/aggregate_model/Jussis_aggregates_monomers/'
		

rot = NullRotator()
grid_res=1.0e-6
rho_i = 917.6
D = 5.0e-3 # should be scalable
D_vec = np.arange(10,100,5)*1e-6
l = len(D_vec)
D_vec = np.append(D_vec,np.arange(100,1000,100)*1e-6)
D_vec = np.append(D_vec,np.arange(1000,5500,500)*1e-6)
D_vec = D_vec
#print(len(D_vec))
#quit()
#grid_res_vec = D_vec/D_vec
#grid_res_vec[0:11] = 1e-7
#grid_res_vec[11:-1] = 1e-6
#grid_res_vec[l:-1] = 1e-5
#grid_res_vec[-1] = 1e-6

#den0 = crystal.Dendrite(D, alpha=1.0, beta=0.35, gamma=0.001,
#                        num_iter=5000, grid_size=400)

#gamma=0.001
alpha = 1.0
beta_vec = np.arange(0.2,0.6,0.1)#[0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95]
gamma_vec = 10**np.arange(-4,-2,0.2)
print(['{:.4e}'.format(g) for g in gamma_vec])
#quit()
#gamma = 0.0001
#for beta in beta_vec[4:5]:
beta=0.5
for i_g,gamma in enumerate(gamma_vec):
	print(beta)
	print(i_g,'of total gamma ',len(gamma_vec))
	print(gamma)
	property_file = dataPath+'dendrite_beta{:.4e}_gamma{:.4e}_adapted_gridres_aspectratio_properties.txt'.format(beta,gamma)
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
		np.savetxt(property_file, np.vstack(("mass", "area", "D_max","ar")).T, fmt="%s")
	if create_new:
		for i in i_vec:
		#for i,D in enumerate(D_vec):
			D = D_vec[i]
			#grid_res = grid_res_vec[i]
			grid_res = 0.25*D*1e-2#0.25*D*1e-2
			print('now at ',i+1,' of total ',len(D_vec))
			den0 = crystal.Dendrite(D, alpha=alpha, beta=beta, gamma=gamma,
			        				num_iter=100000, grid_size=400)
			
			gen = generator.MonodisperseGenerator(den0, rot, grid_res)
			
			agg = aggregate.Aggregate(gen)		        
			#agg = aggregate.RimedAggregate(gen,ident=0)
			#agg_rime = riming.generate_rimed_aggregate(agg, N=1, align=True, riming_lwp=0.5)
			#gen = riming_runs.gen_monomer(psd="monodisperse", size=D, mono_type="dendrite_reiter",alpha=alpha,beta=beta,gamma=gamma,grid_res=grid_res, rimed=True)
			#agg = riming_runs.generate_rimed_aggregate(gen, N=1, align=False, riming_lwp=0.0, riming_mode="subsequent")
			
		    # save shapefile
			savePath = dataPath+'dendrite_beta{:.4e}_gamma{:.4e}/'.format(beta,gamma)
			if not os.path.exists(savePath):
				os.makedirs(savePath)
			savefile = 'dendrite_beta{:.4e}_gamma{:.4e}_size_{:.6e}_res_{:.2e}.txt'.format(beta,gamma,D,grid_res)
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
			D_max = mcs.minimum_covering_sphere(agg.X)[1]*2
			#thickness = agg.grid_res*(1 + np.max(agg.X[:, 2]) - np.min(agg.X[:, 2]))
			aspect_ratio = agg.aspect_ratio()
			
			
			with open(property_file,'a+') as f:
				np.savetxt(f, np.vstack((mass, area, D_max,aspect_ratio)).T, fmt="%.6e")
				f.close()
		#quit()


