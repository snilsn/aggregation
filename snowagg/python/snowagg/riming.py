"""Aggregation and riming driver (drop-in replacement for aggregation.riming).

This is the original Python driver with the heavy work done by the C++
core. Differences to the original code are marked with "snowagg:".
"""
import os
import pickle

import numpy as np
from numpy import random
from scipy import stats

from . import aggregate, crystal, generator, mcs, rotator  # noqa: F401

rho_w = 1000.0
rho_i = 916.7


def get_N_rime_particles(agg, rot, riming_lwp, riming_eff=1.0, align=True,
                         num_area_samples=10, debug=False):
    """Calculate the number of rime particles for a given LWP.

    Returns:
        Tuple (N_particles, area), where N_particles is the number of
        rime particles needed, and area is the averaged projected area [m^2].
    """
    area_list = []
    for i in range(num_area_samples):
        area_list.append(agg.vertical_projected_area())
        if align:
            agg.align()
        agg.rotate(rot)
    area = np.mean(area_list)

    vol = riming_lwp * area / rho_w
    N_particles = int(round(riming_eff * vol / agg.grid_res**3))
    if debug:
        print(riming_lwp, area, vol, N_particles)
    return (N_particles, area)


def lwp_from_N(agg, N, area):
    """The liquid water path [kg/m^2] corresponding to N rime particles."""
    vol = N * agg.grid_res**3
    return vol * rho_w / area


def generate_rime(agg, rot, riming_lwp, riming_eff=1.0, align=True,
                  pen_depth=120e-6, lwp_div=10.0, compact_dist=0., iter=False):
    """Generate rime on an aggregate (see the original documentation).

    Returns:
        A generator if iter=True, None otherwise.
    """

    def gen():
        remaining_lwp = riming_lwp

        while remaining_lwp > 0:
            (N_particles, area) = get_N_rime_particles(
                agg, rot, min(riming_lwp / lwp_div, remaining_lwp), riming_eff, align=align)
            N_to_add = max(N_particles, 10)
            agg.add_rime_particles(N=N_to_add, pen_depth=pen_depth, compact_dist=compact_dist)
            remaining_lwp -= lwp_from_N(agg, N_to_add, area)
            if align:
                agg.align()
            agg.rotate(rot)
            yield agg

    if iter:
        return gen()
    else:
        for i in gen():
            pass


def generate_rimed_aggregate(*args, **kwargs):
    """Generate a rimed aggregate particle.

    Args:
        monomer_generator: Callable returning a new monomer Aggregate
            (see gen_monomer).
        N: Number of ice crystals to aggregate.
        align: If True, the aggregate is kept horizontally aligned between
            iterations.
        riming_lwp: Liquid water path that the aggregate is assumed to fall
            through [kg/m^2].
        riming_eff: Riming efficiency (between 0 and 1, default 1).
        riming_mode: "simultaneous" or "subsequent".
        rime_pen_depth: The distance that rime particles are allowed to
            penetrate into the particle.
        seed: Random seed (np.random.seed(seed) is called).
        lwp_div: See generate_rime.
        compact_dist: See RimedAggregate.add_rime_particles.
        debug: If True, print additional debugging information.
        iter: If True, return an iterator over the stages of aggregation and
            riming; each iteration gives the list of particles at that stage.

    Returns:
        The final Aggregate, or a generator if iter=True.
    """
    if "iter" in kwargs:
        iter = kwargs["iter"]
        del kwargs["iter"]
    else:
        iter = False

    if iter:
        def generator():
            for aggs in generate_rimed_aggregate_iter(*args, **kwargs):
                yield aggs
        return generator()
    else:
        aggs = None
        for aggs in generate_rimed_aggregate_iter(*args, **kwargs):
            pass
        return aggs[0]


def generate_rimed_aggregate_iter(monomer_generator, N=5, align=True,
                                  riming_lwp=0.0, riming_eff=1.0, riming_mode="simultaneous",
                                  rime_pen_depth=120e-6, seed=None, lwp_div=10, compact_dist=0.,
                                  debug=False):
    """Generate a rimed aggregate particle (see generate_rimed_aggregate)."""

    random.seed(seed)

    align_rot = rotator.PartialAligningRotator(exp_sig_deg=40, random_flip=True)
    uniform_rot = rotator.UniformRotator()

    agg = [monomer_generator(ident=i) for i in range(N)]
    yield agg

    while len(agg) > 1:
        r = np.array([((a.extent[0][1] - a.extent[0][0]) +
                       (a.extent[1][1] - a.extent[1][0])) / 4.0 for a in agg])
        # snowagg: element count without copying X
        m_r = np.sqrt(np.array([a.num_elements for a in agg]) / r)
        r_mat = (np.tile(r, (len(agg), 1)).T + r)**2
        mr_mat = abs(np.tile(m_r, (len(agg), 1)).T - m_r)
        p_mat = r_mat * mr_mat
        p_mat /= p_mat.max()
        collision = False
        while not collision:

            i = random.randint(len(agg))
            j = i
            while j == i:
                j = random.randint(len(agg))
            rnd = random.rand()
            if rnd < p_mat[i][j]:
                if debug:
                    print(i, j)
                agg_top = agg[i] if (m_r[i] > m_r[j]) else agg[j]
                agg_btm = agg[i] if (m_r[i] <= m_r[j]) else agg[j]
                agg_btm.rotate(uniform_rot)

                # snowagg: pass the aggregate itself (elements and idents are
                # taken from it without copying through Python)
                collision = agg_top.add_particle(
                    particle=agg_btm, required=True, pen_depth=80e-6,
                    add_N_monomers=agg_btm.monomer_number,
                    add_id_branch=agg_btm.id_tree)
                if collision:
                    if align:
                        agg_top.align()
                        agg_top.rotate(align_rot)
                    else:
                        agg_top.rotate(uniform_rot)
                    agg.pop(i if (m_r[i] <= m_r[j]) else j)

        if riming_mode == "simultaneous":
            for a in agg:
                generate_rime(a, align_rot if align else uniform_rot,
                              riming_lwp / float(N - 1), riming_eff=riming_eff,
                              pen_depth=rime_pen_depth, lwp_div=lwp_div,
                              compact_dist=compact_dist)

        if len(agg) > 1:
            yield agg

    if (riming_mode == "subsequent") or (N == 1):
        for a in generate_rime(agg[0], align_rot if align else uniform_rot,
                               float(riming_lwp), riming_eff=riming_eff,
                               pen_depth=rime_pen_depth, lwp_div=lwp_div, iter=True,
                               compact_dist=compact_dist):
            yield [a]

    if align:
        agg[0].align()
        agg[0].rotate(align_rot)
    agg[0].rotate(rotator.HorizontalRotator())

    yield agg


def gen_polydisperse_monomer(monomers=[], ratios=[]):
    """Monomer generator that picks from several monomer distributions.

    Args:
        monomers: list of dicts with gen_monomer arguments for each type.
        ratios: relative contributions of each monomer type.
    """
    if len(monomers) != len(ratios):
        raise AttributeError('The length of the list of monomers must  match' +
                             'the length of the list of ratios')

    if sum(ratios) != 1.0:
        print('Warning! Distro ratios do not sum up to 1.0 ... normalizing')

    genlist = [gen_monomer(**i) for i in monomers]
    cumsum = np.cumsum(ratios)

    def polygen(ident=0):
        rnd = np.random.uniform()
        i = np.arange(len(genlist))[cumsum > rnd][0]
        return genlist[i](ident)

    return polygen


_dendrite_grid = None


def dendrite_grid():
    """The pregenerated dendrite grid shipped with the original package."""
    # snowagg: loaded once instead of for every monomer
    global _dendrite_grid
    if _dendrite_grid is None:
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "dendrite_grid.dat")
        with open(path, 'rb') as f:
            _dendrite_grid = pickle.load(f, encoding="latin1")
    return _dendrite_grid


def gen_monomer(psd="monodisperse", size=1e-3, min_size=0.1e-3,
                max_size=20e-3, mono_type="dendrite", grid_res=0.02e-3,
                rimed=False, debug=False):
    """Make a monomer crystal generator.

    Args:
        psd: "monodisperse" or "exponential".
        size: If psd="monodisperse", the diameter of the ice crystals; if
            psd="exponential", the inverse of the slope parameter.
        min_size, max_size: Minimum and maximum allowed crystal size.
        mono_type: "dendrite", "plate", "needle", "rosette", "bullet",
            "column" or "spheroid".
        grid_res: The volume element size.
        rimed: True if a rimed aggregate should be generated.
        debug: If True, debug information will be printed.
    """

    def make_cry(D):
        if mono_type == "dendrite":
            cry = crystal.Dendrite(D, hex_grid=dendrite_grid())
        elif mono_type == "plate":
            cry = crystal.Plate(D)
        elif mono_type == "needle":
            cry = crystal.Needle(D)
        elif mono_type == "rosette":
            cry = crystal.Rosette(D)
        elif mono_type == "bullet":
            cry = crystal.Bullet(D)
        elif mono_type == "column":
            cry = crystal.Column(D)
        elif mono_type == "spheroid":
            cry = crystal.Spheroid(D, 1.0)
        return cry

    rot = rotator.UniformRotator()

    def gen(ident=0):
        if psd == "monodisperse":
            D = size
        elif psd == "exponential":
            psd_f = stats.expon(scale=size)
            D = max_size + 1
            while (D < min_size) or (D > max_size):
                D = psd_f.rvs()

        cry = make_cry(D)
        if debug:
            print(D, mono_type)

        gen = generator.MonodisperseGenerator(cry, rot, grid_res)
        if rimed:
            agg = aggregate.RimedAggregate(gen, ident=ident)
        else:
            agg = aggregate.Aggregate(gen, ident=ident)
        return agg

    return gen


def visualize_crystal(mono_type):
    gen = gen_monomer(mono_type=mono_type, size=2e-3, grid_res=40e-6)
    cry = gen()
    cry.align()
    cry.visualize(bgcolor=(1, 1, 1))
