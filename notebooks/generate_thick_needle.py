'''
generate columnar particles with desired size and aspect ratio. Size and aspect ratio are taken from McSnow single particle simulation
'''


import numpy as np
import sys
import os
try:
  import cPickle as pickle
except:
  import pickle
import multiprocessing
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import xarray as xr
import random

import aggregation
from aggregation import generator, aggregate,mcs
from extended_aggregation import Hexprism, GenericDendrite, NullRotator

dendrite_file = open(aggregation.__path__[0]+"/dendrite_grid.dat", 'rb')
kwargs = {"encoding": "latin1"} if sys.version_info[0] >= 3 else {}
dend_grid = pickle.load(dendrite_file, **kwargs) # Throws an error if out of scope

#######################################################################
# Setup the generator
#######################################################################
def a_from_L(L,habit='column'):
	"""Relation between different dimensions of the crystal.

	Args:            
	L: the height of the hexagonal prism.

	Returns:
	a, The side of the hexagon.
	"""
	if habit == 'column':
		if L < 100e-6:
			return 0.35*L
		else:
			return 3.48*(L*1e6)**0.5 * 1e-6
	elif habit=='needle':
		return 3.527e-2 * (L*1e2)**0.437 / 2.0 * 1e-2 

def gen_save_crystal(size, habit='needle', ar=None,
                     rotate_90=True, # if column or needle the simmetry axis has to be either x or y, so need a hard swap of axis set to True for doing this
                     ):
    if ar is None: # standard habits
        if habit=='needle':
            cry = aggregation.crystal.Needle(size)
        elif habit=='dendrite':
            cry = aggregation.crystal.Dendrite(size)
        ar = -99 # just to write something 
    else: # ar is not None, we need the generic
        if habit=='dendrite':
            cry = GenericDendrite(size, ar=ar, hex_grid=dend_grid) 
        elif habit=='hexprism':
            cry = Hexprism(size, ar)
    gen = generator.MonodisperseGenerator(cry, rot, grid_res)
    if rimed:
        agg = aggregate.RimedAggregate(gen, ident=ident)
    else:
        agg = aggregate.Aggregate(gen, ident=ident)
    if rotate_90:
        agg.X = np.roll(agg.X, 1, axis=1) # z becomes x, y-> z and x->y
        agg.update_extent()
    return agg
def get_properties(agg, resolution=1):
	mass = ice_density*agg.X.shape[0]*resolution**3
	Dmax = mcs.minimum_covering_sphere(agg.X*resolution)[1]*2 # seems to be wrong for needles, not sure why!!
	areaCalc = agg.vertical_projected_area()
	#thickness = resolution*(1 + np.max(agg[:, 2]) - np.min(agg[:, 2]))
	aspect = agg.aspect_ratio()
	return Dmax, mass, areaCalc, aspect

#######################################################################
# Some definitions
#######################################################################

#grid_res = 1e-6
rot = NullRotator()
ice_density = 917.6
rimed = False
ident = 0
colors = ['#4477AA', '#EE6677', '#228833', '#CCBB44', '#66CCEE', '#AA3377', '#BBBBBB']

#######################################################################
# Run
#######################################################################

sizes = np.linspace(5e-6, 4e-3, 100) # sizes [meters] of the particles
#sizes = np.linspace(1.5e-3, 4.6e-3, 5)
#nprocs = 1 In principle it could be parallel
#monomer = 'needle'


#McSnowfile = '/project/meteo/work/L.Terzi/McSnowoutput/habit/trajectories/1d_habit_trajectories_habit1_IGF2_xi1_nz200_iwc1_nugam10_mugam10_dtc5_nrp1_vt3_coll_kern0_at0_stick2_colleffi1_dt1_bndtype3_ba500_domtop5000._atmo1_ssat50'
McSnowfile = '/project/meteo/work/L.Terzi/McSnowoutput/habit/trajectories/1d_habit_trajectories_habit1_IGF2_xi1_nz200_iwc3_nugam3.5_mugam3.5_dtc5_vt3_bndtype3_ba500_domtop5000._atmo1_ssat50'
McSnow = xr.open_dataset(McSnowfile+'/mass2fr.nc')
McSnow = McSnow.to_dataframe()
mcTable = McSnow[(McSnow['sPhi']>1)]
#mcTable = mcTable.set_index(pd.Index(range(len(mcTable))))
x = random.sample(range(1,len(mcTable)),50)
# fig,ax = plt.subplots()
# mcTablesel = mcTable.iloc[x]
# ax.loglog(mcTable.dia,mcTable.mTot,marker='.',ls='None',c=colors[0])
# ax.loglog(mcTablesel.dia,mcTablesel.mTot,marker='.',ls='None',c=colors[1])
# ax.set_ylabel('mass [kg]',fontsize=16)
# ax.set_xlabel('Dmax [m]',fontsize=16)
# ax.tick_params(which='both',labelsize=14)
# plt.grid()
# plt.tight_layout()
# plt.savefig('test_subsample_mcSnow.png')
# plt.show()
#quit()
propertyfile = 'thick_needles_properties.txt'
with open(propertyfile,'w') as f:
	np.savetxt(f,np.vstack(('mass','Dmax','ar','area')).T,fmt='%s')
	f.close()
with open(propertyfile,'a+') as f:
	for i,size,ar in zip(range(50),mcTablesel.dia,mcTablesel.sPhi):
		print(i)
		grid_res = 0.5*size*1e-2
		agg = gen_save_crystal(size, habit='hexprism', ar=ar, rotate_90=True)
		Dmax,mass,area,aspect_ratio = get_properties(agg,grid_res)
		np.savetxt(f,np.vstack((mass,size,aspect_ratio**-1,area)).T,fmt='%.6e')
		f.flush()
		savefile = 'needle_size_{:.6e}_aspect_{:.2f}_adjusted_res_{:.2e}.txt'.format(size, aspect_ratio**-1,grid_res)
		grid = agg.grid()
		np.savetxt('needle/'+savefile, grid, fmt="%d") # in multiples of grid_res
		agg.align()
		grid = agg.grid()
		savefile = 'align_'+savefile
		np.savetxt('needle/'+savefile, grid, fmt="%d") # in multiples of grid_res








