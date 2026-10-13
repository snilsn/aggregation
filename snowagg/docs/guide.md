# User guide

[← README](../../README.md) · **User guide** · [Examples](examples.md) · [Technical details](../README.md)

snowagg builds 3D models of snowflakes: single ice crystals, aggregates of crystals, and rimed
snowflakes and graupel. This guide explains how to use it, step by step, with short code. The
[examples](examples.md) are complete scripts with figures.

1. [The model in a nutshell](#1-the-model-in-a-nutshell)
2. [A first snowflake](#2-a-first-snowflake)
3. [Monomers: single crystals](#3-monomers-single-crystals)
4. [Aggregation](#4-aggregation)
5. [Riming](#5-riming)
6. [Watching a particle grow](#6-watching-a-particle-grow)
7. [Deposition and sublimation](#7-deposition-and-sublimation)
8. [Properties of a particle](#8-properties-of-a-particle)
9. [Saving a particle](#9-saving-a-particle)
10. [Many snowflakes: seeds, threads, ensembles](#10-many-snowflakes-seeds-threads-ensembles)
11. [Coming from `aggregation`](#11-coming-from-aggregation)
12. [Function reference](#12-function-reference)

## 1. The model in a nutshell

A particle is a cloud of points: the centres of small cubes of ice, the **volume elements**, all
with the same edge length `grid_res` (for example 10 µm). A particle of `n` elements has the
mass `n × grid_res³ × 916.7 kg m⁻³`. Everything else follows from where the elements are:
size, projected area, shape, fall speed.

A snowflake is built in steps (Leinonen and Moisseev, 2015; Leinonen and Szyrmer, 2015):

1. **Monomers.** Single crystals: a crystal shape (plate, column, dendrite, …) is filled with
   the points of a cubic grid and turned into a random orientation.
2. **Aggregation.** Pairs of particles collide, with a probability that mimics differential
   sedimentation, and stick where they first touch, until one particle is left.
3. **Riming.** Supercooled droplets freeze onto the particle where they hit it. How many
   follows from the liquid water the particle falls through.
4. **Deposition** (optional): water vapour freezes onto the particle, or ice sublimates.

All quantities are in SI units: m, kg, kg m⁻², K, Pa.

## 2. A first snowflake

```python
from snowagg import fallvelocity, mcs, riming

# monomers: dendrites, sizes from an exponential distribution, 10 µm elements
gen = riming.gen_monomer(psd="exponential", size=1e-3, min_size=0.1e-3, max_size=3e-3,
                         mono_type="dendrite", grid_res=10e-6, rimed=True)

# aggregate 10 of them, then rime the aggregate with 0.1 kg/m² of liquid water
agg = riming.generate_rimed_aggregate(gen, N=10, riming_lwp=0.1, riming_mode="subsequent",
                                      lwp_div=10, seed=1)

X = agg.X                                                     # (n, 3) element centres [m]
mass = riming.rho_i * len(X) * agg.grid_res**3                # [kg]
D_max = 2 * mcs.minimum_covering_sphere(agg)[1]               # maximum dimension [m]
area = agg.projected_area(dim=2)                              # seen from above [m²]
v = fallvelocity.fall_velocity(agg, T=263.15, P=1000e2)       # fall speed [m/s]
```

* `gen_monomer` returns a **monomer generator**: a function that makes a new crystal each time
  it is called. `rimed=True` makes crystals that can be rimed.
* `generate_rimed_aggregate` runs the whole model and returns the final particle, an
  `Aggregate`. With `riming_lwp=0` it only aggregates.
* `seed=1` makes the run reproducible: the same seed gives the same snowflake.

The result (`examples/first_snowflake.py`): 466,000 elements, 0.43 mg, 3.9 mm across, falling at
1.2 m s⁻¹.

![A first snowflake](img/examples/first_snowflake.png)

## 3. Monomers: single crystals

### With `gen_monomer`

```python
gen = riming.gen_monomer(psd="monodisperse", size=1e-3, mono_type="plate", grid_res=10e-6)
crystal = gen()          # an Aggregate made of one crystal
```

| Argument | Meaning | Default |
|---|---|---|
| `mono_type` | `"plate"`, `"column"`, `"needle"`, `"dendrite"`, `"rosette"`, `"bullet"` or `"spheroid"` | `"dendrite"` |
| `psd` | size distribution: `"monodisperse"` (all crystals have `size`) or `"exponential"` | `"monodisperse"` |
| `size` [m] | the size of the crystals, or for `"exponential"` the mean of the distribution | 1 mm |
| `min_size`, `max_size` [m] | sizes drawn outside these limits are drawn again (`"exponential"`) | 0.1 mm, 20 mm |
| `grid_res` [m] | edge of a volume element | 20 µm |
| `rimed` | `True`: crystals that can be rimed (`RimedAggregate`) | `False` |

The crystal types, all of `size` 1 mm (`examples/crystals.py`):

![The seven crystal types](img/examples/crystals.png)

| Type | Shape | `size` is | The rest follows from |
|---|---|---|---|
| `plate` | hexagonal plate | the diameter (corner to corner) | thickness: Hong (2007) |
| `column` | hexagonal column | the length | width from the length |
| `needle` | thin hexagonal column | the length | width: Pruppacher and Klett |
| `dendrite` | a 2D dendrite (Reiter's model) of the thickness of a plate | the diameter | thickness: Pruppacher and Klett |
| `rosette` | six bullets from one centre | the maximum diameter | |
| `bullet` | a single bullet | the maximum diameter | |
| `spheroid` | a sphere (a frozen drop) | the diameter | |

**Choosing `grid_res`.** Small elements resolve thin crystals, but the number of elements, and
so memory and time, grows with `1/grid_res³` for rimed particles. A plate of 1 mm is about 50 µm
thick, so 10 µm elements give it five layers; 40 µm elements give it one. A rule of thumb: at
least two elements across the thinnest part of your crystals. An element takes about
100 bytes, so 10 million elements need about 1 GB.

**Mixtures of crystal types.** `gen_polydisperse_monomer` picks each monomer from one of several
generators:

```python
gen = riming.gen_polydisperse_monomer(
    monomers=[dict(psd="exponential", size=1e-3, mono_type="dendrite", grid_res=10e-6, rimed=True),
              dict(psd="exponential", size=0.5e-3, mono_type="column", grid_res=10e-6, rimed=True)],
    ratios=[0.7, 0.3])
```

**Building crystals yourself.** A monomer generator is any function `gen(ident=0)` that returns
a new `Aggregate`. The building blocks are a crystal shape (module `crystal`), a generator that
fills it with elements and turns it (`generator.MonodisperseGenerator`, with a rotator from
`rotator`), and the `Aggregate` (or `RimedAggregate`) made from it:

```python
from snowagg import aggregate, crystal, generator, rotator

shape = crystal.Spheroid(0.8e-3, 0.5)            # an oblate spheroid, axis ratio 0.5
fill = generator.MonodisperseGenerator(shape, rotator.UniformRotator(), 10e-6)

def gen(ident=0):
    return aggregate.RimedAggregate(fill, ident=ident)
```

Dendrites of other shapes come from Reiter's model, `dendrite.generate_dendrite(alpha, beta,
gamma)`: `crystal.Dendrite(D, hex_grid=dendrite.generate_dendrite(1.0, 0.4, 0.001))` (the
default shape, `riming.dendrite_grid()`, is the one shipped with the original).

## 4. Aggregation

```python
agg = riming.generate_rimed_aggregate(gen, N=20, seed=5)     # 20 crystals, no riming
```

How it works: the model first makes `N` monomers. Then it repeatedly picks two particles, with
a probability that grows with their cross sections and with the difference of their
mass-to-size ratios (a proxy for the difference in fall speed), drops the one on the other at a
random horizontal position, and lets it stick where they first touch (it may penetrate up to
80 µm). After each merge the new particle is turned into a falling orientation. This repeats
until one particle is left.

| Argument | Meaning | Default |
|---|---|---|
| `N` | number of monomers | 5 |
| `align` | `True`: particles fall horizontally aligned (longest axes horizontal, canting angles of about 40°, randomly flipped); `False`: they tumble (uniformly random orientations) | `True` |
| `seed` | random seed; `None`: a different snowflake each time | `None` |

The final particle is in a falling orientation (roughly horizontal, with some canting and a
random azimuth). To put it into the frame of its principal axes (longest along x, shortest
along z), call `agg.align()`.

To merge two particles yourself: `agg.add_particle(particle=other, required=True,
pen_depth=80e-6)`.

## 5. Riming

```python
agg = riming.generate_rimed_aggregate(gen, N=10, riming_lwp=0.3, riming_mode="subsequent",
                                      lwp_div=30, seed=1)
```

`riming_lwp` is the **liquid water path** the particle falls through [kg m⁻²]: the mass of
droplets it collects per area of its cross section. A droplet is one volume element, and it
freezes where it hits the particle. 0.1 kg m⁻² rimes a snowflake lightly; with 1 kg m⁻² it
becomes graupel-like (`examples/riming.py`):

![The same snowflake rimed with more and more water](img/examples/riming.png)

| Argument | Meaning | Default |
|---|---|---|
| `riming_lwp` [kg m⁻²] | liquid water path | 0 |
| `riming_mode` | `"subsequent"`: aggregate first, then rime; `"simultaneous"`: after every merge, every particle collects `riming_lwp / (N − 1)` (aggregation and riming at the same time) | `"simultaneous"` |
| `lwp_div` | number of riming steps: the particle turns between steps | 10 |
| `rime_pen_depth` [m] | a droplet moves to the deepest free spot within this distance of where it first touches; 0: it sticks where it first touches (lighter rime) | 120 µm |
| `riming_eff` | collection efficiency: the fraction of droplets that stick | 1 |
| `compact_dist` | compaction of the rime (0: off) | 0 |

* **Steps.** Within a riming step all droplets come from the same direction, and light rime grows
  in spikes towards them. A falling snowflake turns many times while it collects a layer of
  droplets, so use enough steps: about 10 g m⁻² per step (`lwp_div = riming_lwp / 0.01`) or
  less. Every step adds at least 10 droplets.
* **Which elements are rime?** Their identifier is −1: `rime_fraction = (agg.ident == -1).mean()`.
* **Riming a particle you already have** (it must be a `RimedAggregate`):
  `riming.generate_rime(agg, rotator.PartialAligningRotator(exp_sig_deg=40, random_flip=True), 0.2, lwp_div=20)`.
* **A single rimed crystal:** `N=1` rimes the monomer.

## 6. Watching a particle grow

With `iter=True`, `generate_rimed_aggregate` returns the stages of the run one by one, each as
the list of particles at that stage: first the monomers, then the particles after every merge.

```python
for stage, particles in enumerate(riming.generate_rimed_aggregate(gen, N=6, riming_lwp=0.2, seed=3,
                                                                    iter=True)):
    largest = max(particles, key=lambda p: p.num_elements)
```

Six plates aggregating and riming at the same time (`examples/growth_stages.py`):

![Growth stages](img/examples/growth_stages.png)

## 7. Deposition and sublimation

```python
from snowagg import deposition

volume = agg.num_elements * agg.grid_res**3      # the particle's ice volume [m³]
deposition.grow_ice(agg, 0.5 * volume)           # add half of it as deposited ice
deposition.grow_ice(agg, -0.2 * volume)          # sublimate (negative volume)
```

Molecules diffuse in from far away with a random walk and stick where they first touch, so
exposed tips grow fastest; sublimation removes elements the same way. Deposited elements have
the identifier −2 (`deposition.DEPOSITION_IDENT`). The random walk is slow for large
particles (`examples/deposition.py`: 30 s for a 1 mm dendrite).

![Deposition and sublimation](img/examples/deposition.png)

## 8. Properties of a particle

| What | How |
|---|---|
| element centres [m] | `agg.X`, an (n, 3) array (read-only; z is vertical) |
| number of elements | `agg.num_elements` or `len(agg)` |
| element size [m] | `agg.grid_res` |
| where each element came from | `agg.ident`: the monomer's number (0 … N−1), −1 for rime, −2 for deposition |
| mass [kg] | `riming.rho_i * agg.num_elements * agg.grid_res**3` |
| maximum dimension [m] | `2 * mcs.minimum_covering_sphere(agg)[1]` (the diameter of the smallest sphere around it) |
| projected area [m²] | `agg.projected_area(dim=2)`: seen along z, from above (pixels of `grid_res`); `dim=0`, `1`: along x, y |
| projected image | `agg.project_on_dim(dim=2)`: a 2D array of element counts |
| aspect ratio | `agg.aspect_ratio()`: height over width of the principal axes (≤ 1); `agg.projected_aspect_ratio(dim)`: of a projection |
| principal axes | `agg.principal_axes()`: (3, 3) array, one axis per column, longest first |
| extent [m] | `agg.extent`: [[x_min, x_max], [y_min, y_max], [z_min, z_max]] |
| fall speed [m/s] | `fallvelocity.fall_velocity(agg, T=263.15, P=1000e2, method="HW")`: Heymsfield and Westbrook (2010); `method="KC"`: Khvorostyanov and Curry (2005) |

Turning a particle: `agg.align()` (principal axes along x, y, z) and `agg.rotate(rotator)` with
`rotator.UniformRotator()` (random orientation), `rotator.HorizontalRotator()` (random azimuth)
or `rotator.PartialAligningRotator(exp_sig_deg=40)` (falling orientation with canting).

A quick picture with matplotlib (the examples use a slightly longer version):

```python
import matplotlib.pyplot as plt
plt.imshow(agg.project_on_dim(dim=1).T > 0, origin="lower", cmap="Blues")   # side view
```

## 9. Saving a particle

```python
import numpy as np, pickle
from snowagg import stl

np.savez("snowflake.npz", X=agg.X, ident=agg.ident, grid_res=agg.grid_res)   # plain numbers
pickle.dump(agg, open("snowflake.pkl", "wb"))                               # the Aggregate itself
G = agg.grid()                         # integer voxel coordinates, in units of grid_res
stl.snowflake_grid_to_stl(G, "snowflake.stl")                                 # a 3D model
```

A pickled `Aggregate` can be loaded and grown further (for example rimed). `grid()` gives every
element its own voxel, the input of scattering codes that work on voxels. The STL file opens in
3D viewers and slicers for 3D printing (`examples/export.py`).

## 10. Many snowflakes: seeds, threads, ensembles

* **Seeds.** `seed=` (or `np.random.seed()` before a run) makes a run reproducible on the same
  machine. snowagg uses numpy's random numbers, as the original does; with
  `snowagg.set_rng_mode("own", seed)` it uses its own generator instead and leaves numpy's
  untouched.
* **Threads.** Operations on whole particles use several threads (by default half the CPUs, at
  most 8); the results don't depend on the number. `snowagg.set_num_threads(n)` or
  `OMP_NUM_THREADS` changes it.
* **Ensembles.** Many independent runs are fastest as one single-threaded process each, with
  their own seeds: `examples/parallel_ensemble.py` runs one per CPU with `multiprocessing`.
  `examples/mass_size.py` fits mass–size relations to a small ensemble:

  ![Mass-size relation](img/examples/mass_size.png)

* **Command line.** One run to files, as with the original's script:

  ```bash
  python -m snowagg.riming_runs --psd exponential --mono_type dendrite --mono_size 1e-3 \
      --mono_min_size 0.1e-3 --mono_max_size 3e-3 --num_monos 10 --grid_res 10e-6 \
      --riming_mode subsequent --riming_lwp 0.1 --riming_eff 1 --rime_pen_depth 120e-6 --output flake.gz
  ```

  writes the voxel grid (`flake.gz`) and the properties (`flake.gz.meta`, JSON: mass, maximum
  diameter, radius of gyration, …).
* **Time.** Aggregating 20 dendrites with 10 µm elements takes about 0.1 s; riming costs about
  2 µs per rime element plus the turning of the particle in every step. The
  [examples](examples.md) run in 1–30 s each.

## 11. Coming from `aggregation`

Code written for the original runs unchanged after replacing `from aggregation import ...` by
`from snowagg import ...`: the modules and functions have the same names, arguments and
results. The same seed gives a statistically equivalent, but not the identical snowflake
(principal axes and rotations are computed in C++); in a verification configuration complete
runs agree bit for bit. Small differences (for example `Aggregate.X` is read-only, aggregates
can be pickled) are listed in the [technical details](../README.md#differences-to-the-original).

## 12. Function reference

| Module | Function or class | What it does |
|---|---|---|
| `riming` | `gen_monomer(...)` | a monomer generator ([§3](#3-monomers-single-crystals)) |
| | `gen_polydisperse_monomer(monomers, ratios)` | a generator for a mixture of monomer types |
| | `generate_rimed_aggregate(gen, N, ...)` | the whole model: aggregation and riming ([§4](#4-aggregation), [§5](#5-riming)) |
| | `generate_rime(agg, rotator, lwp, ...)` | riming of an existing particle |
| | `dendrite_grid()`, `rho_i` | the shipped dendrite shape; the ice density (916.7 kg m⁻³) |
| `aggregate` | `Aggregate`, `RimedAggregate` | a particle: `X`, `ident`, `align()`, `rotate()`, `projected_area()`, `add_particle()`, `add_rime_particles()`, `grid()`, … ([§8](#8-properties-of-a-particle)) |
| `crystal` | `Plate`, `Column`, `Needle`, `Dendrite`, `Rosette`, `Bullet`, `Spheroid` | crystal shapes |
| `generator` | `MonodisperseGenerator(crystal, rotator, grid_res)` | fills a shape with elements and turns it |
| `rotator` | `UniformRotator`, `HorizontalRotator`, `PartialAligningRotator`, `Rotator` | orientations |
| `dendrite` | `generate_dendrite(alpha, beta, gamma)` | Reiter's 2D dendrite growth |
| `deposition` | `grow_ice(agg, volume)` | deposition and sublimation ([§7](#7-deposition-and-sublimation)) |
| `mcs` | `minimum_covering_sphere(agg)` | centre and radius of the smallest sphere around the particle |
| `fallvelocity` | `fall_velocity(agg, T, P, method)` | fall speed |
| `stl` | `snowflake_grid_to_stl(G, file)` | STL export of a voxel grid |
| `riming_runs` | command line | one run to a file ([§10](#10-many-snowflakes-seeds-threads-ensembles)) |
| `snowagg` | `set_num_threads(n)`, `set_rng_mode(mode, seed)` | threads and random numbers |

The docstrings have the details: `help(riming.generate_rimed_aggregate)`.

References: J. Leinonen and D. Moisseev (2015), *J. Geophys. Res. Atmos.*, 120, 229–239;
J. Leinonen and W. Szyrmer (2015), *Earth and Space Science*, 2, 346–358; A. J. Heymsfield and
C. D. Westbrook (2010), *J. Atmos. Sci.*, 67, 2469–2482; V. I. Khvorostyanov and J. A. Curry
(2005), *J. Atmos. Sci.*, 62, 4343–4357; C. Reiter (2005), *Chaos, Solitons & Fractals*, 23,
1111–1119; G. Hong (2007), *J. Geophys. Res.*, 112, D11208.
