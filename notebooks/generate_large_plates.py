from aggregation import riming_runs # type: ignore
from matplotlib import pyplot as plt
import numpy as np
from aggregation.riming import gen_polydisperse_monomer # type: ignore
from aggregation import fallvelocity # type: ignore
from aggregation import mcs # type: ignore
import os
from sys import argv
from scipy import stats

scriptname, i_size = argv
#rime_lwp='0.2'
i_size = int(i_size)
#rimeelwp = float(rime_lwp)
savepath = '/scratch/l/L.Terzi/plate_aggregates/'
rho_i = 917.6
if not os.path.exists(savepath): #create directory if it does not exists
    os.makedirs(savepath)

size = np.loadtxt('size_params_plate_dendrite_aggs_cluster.txt')[i_size]
#print(size)
#quit()
print('%E'%(size))

#print(size)
dir_shape = savepath+'/size_%E' % float(size)
if not os.path.exists(dir_shape):
    os.makedirs(dir_shape)
    print("Directory " , dir_shape ,  " Created ")
min_size = 10.0e-6; max_size = 3000.0e-6; grid_res = 50e-6; rimed = True; align = True; riming_lwp = 0.0; riming_mode="subsequent"; lwp_div=500; compact_dist=62/100.0
mono_type = 'plate'

propertyfile = savepath+'plate_regulargrid_%E_res_%E_properties.txt' %(float(size),float(grid_res))
propertyfile = propertyfile.replace(' ','')

with open(propertyfile,'wb') as f: #
    np.savetxt(f, np.vstack(("size", "min_size", "max_size",  "rimed", "grid_res", "align", "riming_lwp", "riming_mode", "lwp_div", "compact_dist")).T, fmt="%s") #, "compact_dist"
    np.savetxt(f, np.vstack((size, min_size, max_size, rimed, grid_res, align, riming_lwp, riming_mode,lwp_div,compact_dist)).T, fmt="%s")      #,compact_dist
    np.savetxt(f, np.vstack(("Nmono", "mass", "area", "Dmax", "vel_HW", "vel_KC")).T, fmt="%s")
    f.close()

polygen = riming_runs.gen_monomer(psd="exponential", size=size, min_size=min_size, max_size=max_size, mono_type=mono_type, rimed=True, grid_res=grid_res)
print('Polygen done')
with open(propertyfile,"a+") as f:
    for Nmono in np.array([100,600,700,800,900,1000]):
        print(Nmono)
        agg = riming_runs.generate_rimed_aggregate(polygen,N=Nmono,align=align,riming_lwp=riming_lwp,riming_mode=riming_mode,lwp_div=lwp_div,compact_dist=compact_dist,iter=False) #compact_dist=62/100.0,compact_dist=compact_dist,
        print('done with agg')
        # save shapefile
        agg.align()
        savefile = dir_shape+"/plate_regulargrid_%E_Nmono_%d_res_%E.txt" %(float(size),(Nmono),float(grid_res))
        savefile = savefile.replace(" ", "")
        print('done with agg')
        #print(agg.grid())
        mass = rho_i*agg.X.shape[0]*agg.grid_res**3
        print('mass',mass)
        area = agg.vertical_projected_area()
        D_max = mcs.minimum_covering_sphere(agg.X)[1]*2
        vel_HW = fallvelocity.fall_velocity(agg, method="HW")
        vel_KC = fallvelocity.fall_velocity(agg, method="KC")
        np.savetxt(f, np.vstack((Nmono, mass, area, D_max, vel_HW, vel_KC)).T, fmt="%.6e")
        print('Dmax',D_max)
        print('done with calculations')
        np.savetxt(savefile, agg.grid(), fmt="%d")
        #np.savetxt(f, np.vstack((Nmono, mass, area, D_max)).T, fmt="%.6e")

        f.flush()
        plt.scatter(agg.X[:,0], agg.X[:,2], lw=(0,), s=3)
        # set plot limits
        plt.gca().set_xlim((agg.X[:,0].min(), agg.X[:,0].max()))
        plt.gca().set_ylim((agg.X[:,2].min(), agg.X[:,2].max()))
        figName = dir_shape+"/plate_regulargrid_%E_Nmono_%d_res_%E.png" %(float(size),(Nmono),float(grid_res))
        figName = figName.replace(" ", "")
        plt.savefig(figName)
        plt.close()
        #quit()
'''
for N in np.arange(1500,2500,100):
	agg = riming_runs.generate_rimed_aggregate(gen, N=N, align=align, riming_lwp=0.1, riming_mode="simultaneous")

	#plt.scatter(agg.X[:,0], agg.X[:,2], lw=(0,), s=3)
	# set plot limits
	#plt.gca().set_xlim((agg.X[:,0].min(), agg.X[:,0].max()))
	#plt.gca().set_ylim((agg.X[:,2].min(), agg.X[:,2].max()))
	#plt.show()
	#calculate the properties for each particle
	mass = [rho_i*agg.X.shape[0]*agg.grid_res**3]
	area = [agg.vertical_projected_area()]
	D_max = [mcs.minimum_covering_sphere(agg.X)[1]*2]
	print('mass',mass,'area',area,'Dmax',D_max)
	with open(prop_file,'a+') as f: #
		np.savetxt(f, np.vstack((N, mass, area, D_max)).T, fmt="%.6e")
		f.close()
	agg.align()
	grid = agg.grid()
	savefile = '{}_size_{:.4e}_minsize_{:.4e}_maxsize{:.4e}_Nmono_{}_gridres_{:.4e}.txt'.format(mono_type,size,min_size,max_size,N,grid_res)
	np.savetxt('../output/aggregates/'+savefile, grid, fmt="%d")
'''