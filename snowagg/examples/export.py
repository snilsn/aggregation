"""Saving a snowflake: as numbers, as a voxel grid, as a 3D model.

* np.savez: the element coordinates, the monomer identifiers and the element
  size, readable without snowagg.
* pickle: the whole Aggregate (it can be loaded again and grown further, for
  example rimed).
* Aggregate.grid(): integer grid coordinates, one per element (the input of
  scattering codes that work on voxels).
* STL: a surface model of the voxels, for 3D viewers and 3D printers.

    python export.py [output directory]
"""
import os
import pickle
import sys

import numpy as np

from snowagg import riming, stl


def main(out="."):
    gen = riming.gen_monomer(psd="exponential", size=1e-3, min_size=0.1e-3, max_size=3e-3,
                             mono_type="dendrite", grid_res=20e-6, rimed=True)
    agg = riming.generate_rimed_aggregate(gen, N=5, riming_lwp=0.05, riming_mode="subsequent", seed=7)
    os.makedirs(out, exist_ok=True)

    # 1. numbers
    np.savez_compressed(os.path.join(out, "snowflake.npz"), X=agg.X, ident=agg.ident, grid_res=agg.grid_res)
    data = np.load(os.path.join(out, "snowflake.npz"))
    print(f"snowflake.npz: {len(data['X']):,} elements of {float(data['grid_res']) * 1e6:g} um")

    # 2. the Aggregate itself
    with open(os.path.join(out, "snowflake.pkl"), "wb") as f:
        pickle.dump(agg, f)
    with open(os.path.join(out, "snowflake.pkl"), "rb") as f:
        again = pickle.load(f)
    print(f"snowflake.pkl: an Aggregate with {again.num_elements:,} elements")

    # 3. voxels: integer coordinates in units of grid_res
    G = agg.grid()
    print(f"grid(): {len(G):,} voxels in a box of {np.ptp(G, axis=0) + 1} cells")

    # 4. a 3D model of the voxels
    stl.snowflake_grid_to_stl(G, os.path.join(out, "snowflake.stl"))
    print(f"snowflake.stl: {os.path.getsize(os.path.join(out, 'snowflake.stl')) / 1e6:.1f} MB")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
