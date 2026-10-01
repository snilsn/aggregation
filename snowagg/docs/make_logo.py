"""The snowagg logo, drawn with snowagg itself.

Three dendrites (the dendrite shape shipped with the model) are rendered on a
common pixel grid, as the volume elements of the model are, and joined at
their branch tips into a small aggregate. Each monomer has its own shade,
like the monomer labels (`ident`) of an aggregate, and a few rime particles
sit on the branch tips. The wordmark is Ubuntu Medium/Bold, converted to
outlines so the SVG looks the same everywhere.

Writes docs/img/logo-light.svg, docs/img/logo-dark.svg (for light and dark
backgrounds) and docs/img/icon.svg.

usage: python docs/make_logo.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "python"))

from snowagg import crystal, generator, riming, rotator  # noqa: E402

OUT = os.path.join(HERE, "img")
FONTS = ["/usr/share/fonts/truetype/ubuntu/Ubuntu-M.ttf", "/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf"]

THEMES = {
    # monomer shades (largest first), rime, wordmark ("snow", "agg"), tagline
    "light": dict(mono=["#1c5cab", "#3987e5", "#6da7ec"], rime="#eb6834",
                  text=["#0b0b0b", "#2a78d6"], tag="#52514e"),
    "dark": dict(mono=["#86b6ef", "#b7d3f6", "#5598e7"], rime="#f08a5d",
                 text=["#ffffff", "#6da7ec"], tag="#c3c2b7"),
}


def voxel_image(hex_grid, npx, rot_deg, sub=4, threshold=0.3):
    """Top view of a dendrite on a grid of npx pixels across: a pixel is set
    when the projected elements (sub x sub per pixel) cover enough of it."""
    D = 1e-3
    res = D / npx
    cry = crystal.Dendrite(D, hex_grid=hex_grid)
    X = np.asarray(generator.MonodisperseGenerator(cry, rotator.Rotator(), res / sub).generate()).T
    a = np.deg2rad(rot_deg)
    P = X[:, :2] @ np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]).T
    fine = np.unique(np.floor(P / (res / sub)).astype(int), axis=0)
    coarse, counts = np.unique(np.floor_divide(fine, sub), axis=0, return_counts=True)
    keep = coarse[counts >= threshold * sub * sub]
    return keep - np.round(keep.mean(0)).astype(int)  # (n, 2) pixel coordinates, centred


def tips(px, n_arms=6, rot_deg=90):
    """Outermost pixel of each branch."""
    ang = np.arctan2(px[:, 1], px[:, 0])
    out = []
    for k in range(n_arms):
        a = np.deg2rad(rot_deg + 60 * k)
        d = np.abs(np.angle(np.exp(1j * (ang - a))))
        cand = px[d < np.deg2rad(8)]
        out.append(cand[np.argmax((cand ** 2).sum(1))])
    return np.array(out)


def compose():
    g = riming.dendrite_grid()
    big = voxel_image(g, 55, 90, threshold=0.45)
    mid = voxel_image(g, 29, 105, threshold=0.35)
    small = voxel_image(g, 19, 75, threshold=0.3)
    t = tips(big)
    # attach the smaller monomers beyond two branch tips (lower right, upper left)
    def attach(px, tip, direction, overlap=5):
        u = np.array([np.cos(np.deg2rad(direction)), np.sin(np.deg2rad(direction))])
        reach = (px @ u).min()  # how far the monomer extends towards the tip
        return px + np.round(tip - u * (reach + overlap)).astype(int)
    mid = attach(mid, t[4], -30)    # tip at 90 + 4*60 = 330 deg
    small = attach(small, t[2], 210)  # tip at 210 deg
    monomers = [big, mid, small]
    # rime: a few single elements just outside the branch tips of all monomers
    rng = np.random.default_rng(7)
    occupied = {tuple(p) for m in monomers for p in m}
    rime = []
    for m, n in zip(monomers, (14, 7, 4)):
        edge = [p for p in m if any((p[0] + dx, p[1] + dy) not in occupied
                                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)))]
        edge = np.array(edge)
        r = np.sqrt(((edge - m.mean(0)) ** 2).sum(1))
        far = edge[r > np.percentile(r, 80)]
        for p in far[rng.choice(len(far), size=min(n, len(far)), replace=False)]:
            free = [(p[0] + dx, p[1] + dy) for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
                    if (p[0] + dx, p[1] + dy) not in occupied]
            if not free:
                continue
            q = free[rng.integers(len(free))]
            occupied.add(q)
            rime.append(q)
    # later monomers do not cover earlier ones (each element belongs to one monomer)
    seen, layers = set(), []
    for m in monomers:
        layer = [tuple(p) for p in m if tuple(p) not in seen]
        seen.update(layer)
        layers.append(np.array(layer))
    return layers, np.array(rime)


def squares_path(px, y_top, x_left, gap=0.14):
    s = 1 - gap
    return "".join(f"M{x - x_left + gap / 2:g} {y_top - y + gap / 2:g}h{s:g}v{s:g}h-{s:g}z" for x, y in px)


def text_path(text, font_file, size, x, y):
    """SVG path data of text (baseline at y) from the font's outlines."""
    from matplotlib.font_manager import FontProperties
    from matplotlib.path import Path
    from matplotlib.textpath import TextPath
    tp = TextPath((0, 0), text, size=size, prop=FontProperties(fname=font_file))
    d, verts, codes = [], tp.vertices, tp.codes
    i = 0
    while i < len(verts):
        c = codes[i]
        vx, vy = verts[i]
        if c == Path.MOVETO:
            d.append(f"M{x + vx:.2f} {y - vy:.2f}")
            i += 1
        elif c == Path.LINETO:
            d.append(f"L{x + vx:.2f} {y - vy:.2f}")
            i += 1
        elif c == Path.CURVE3:
            (x2, y2) = verts[i + 1]
            d.append(f"Q{x + vx:.2f} {y - vy:.2f} {x + x2:.2f} {y - y2:.2f}")
            i += 2
        elif c == Path.CURVE4:
            (x2, y2), (x3, y3) = verts[i + 1], verts[i + 2]
            d.append(f"C{x + vx:.2f} {y - vy:.2f} {x + x2:.2f} {y - y2:.2f} {x + x3:.2f} {y - y3:.2f}")
            i += 3
        elif c == Path.CLOSEPOLY:
            d.append("Z")
            i += 1
        else:
            i += 1
    return "".join(d), tp.get_extents().width


def write_svg(path, layers, rime, theme, wordmark=True):
    allpx = np.vstack(layers + ([rime] if len(rime) else []))
    x0, y0 = allpx[:, 0].min(), allpx[:, 1].max()
    w_icon = allpx[:, 0].max() - x0 + 1
    h_icon = y0 - allpx[:, 1].min() + 1
    pad = 2
    parts = []
    for layer, colour in zip(layers, theme["mono"]):
        parts.append(f'<path fill="{colour}" d="{squares_path(layer, y0, x0)}"/>')
    if len(rime):
        parts.append(f'<path fill="{theme["rime"]}" d="{squares_path(rime, y0, x0)}"/>')
    width, height = w_icon, h_icon
    if wordmark:
        size = 0.42 * h_icon
        base = 0.5 * h_icon + 0.2 * size
        x = w_icon + 0.12 * h_icon
        d1, w1 = text_path("snow", FONTS[0], size, x, base)
        d2, w2 = text_path("agg", FONTS[1], size, x + w1 + 0.02 * size, base)
        d3, w3 = text_path("aggregation · riming · deposition, in C++", FONTS[0], 0.115 * h_icon,
                           x + 0.03 * size, base + 0.36 * size)
        parts.append(f'<path fill="{theme["text"][0]}" d="{d1}"/>')
        parts.append(f'<path fill="{theme["text"][1]}" d="{d2}"/>')
        parts.append(f'<path fill="{theme["tag"]}" d="{d3}"/>')
        width = x + max(w1 + w2 + 0.02 * size, w3) + 0.05 * h_icon
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{-pad} {-pad} {width + 2 * pad:.2f} '
           f'{height + 2 * pad:.2f}" role="img" aria-label="snowagg">\n'
           f'<title>snowagg</title>\n' + "\n".join(parts) + "\n</svg>\n")
    with open(path, "w") as f:
        f.write(svg)
    print("wrote", path, f"({len(svg) / 1024:.0f} kB)")


def main():
    os.makedirs(OUT, exist_ok=True)
    layers, rime = compose()
    for name, theme in THEMES.items():
        write_svg(os.path.join(OUT, f"logo-{name}.svg"), layers, rime, theme)
    write_svg(os.path.join(OUT, "icon.svg"), layers, rime, THEMES["light"], wordmark=False)


if __name__ == "__main__":
    main()
