"""
Top-down (face-on) view of the Edenhofer et al. (2024) 3D dust map,
with an overlaid sightline in the direction of the Galactic Center.
 
Reproduces the geometry of Fig. 5 (top panel) of the paper: heliocentric
Galactic Cartesian X-Y, integrating the extinction density along Z.
 
Requires: numpy, astropy, healpy, matplotlib
Data:     mean_and_std_healpix.fits  (Zenodo 8187943, ~3.2 GB)
"""
 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import healpy as hp
from astropy.io import fits
 
# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
from pathlib import Path

FNAME = Path(__file__).resolve().parent.parent / "Data_And_Samplers" / "mean_and_std_healpix.fits"
 
E_TO_AV = 2.8          # ZGR23 E -> A_V  (Edenhofer et al. 2024 recommendation)
R_MAX   = 1250.0       # pc, outer edge of the reconstruction
Z_HALF  = 400.0        # pc, half-thickness of the integrated slab (paper: 400)
 
# Draft settings run in ~15 s; final settings take a few minutes.
DRAFT = True
PIX    = 4.0 if DRAFT else 2.0    # pc per image pixel
Z_STEP = 4.0 if DRAFT else 2.0    # pc, sampling along z
 
# Clip to the cylinder where every column has the full path length,
# so brightness reflects dust and not the shrinking chord of the sphere.
CLIP_TO_CYLINDER = False
 
# The sightline(s) to draw: (longitude in deg, label)
SIGHTLINES = [(0.0, "")]
 
# ----------------------------------------------------------------------
# Load
# ----------------------------------------------------------------------
with fits.open(FNAME) as hdul:
    hdr = hdul["MEAN"].header
    dens = hdul["MEAN"].data.astype(np.float32)        # (n_r, n_pix), E per pc
    r_bounds = np.asarray(
        hdul["RADIAL PIXEL BOUNDARIES"].data.field(0), dtype=np.float64
    )
 
n_r, n_pix = dens.shape
nside = hp.npix2nside(n_pix)
nest = hdr.get("ORDERING", "NESTED").strip().upper().startswith("NEST")
assert r_bounds.size == n_r + 1, "boundaries should have one more entry than shells"
print(f"nside={nside}  shells={n_r}  r=[{r_bounds[0]:.1f}, {r_bounds[-1]:.1f}] pc  nest={nest}")
 
# ----------------------------------------------------------------------
# Resample onto a Cartesian X-Y grid, integrating over z
# ----------------------------------------------------------------------
n = int(round(2 * R_MAX / PIX))
edges = np.linspace(-R_MAX, R_MAX, n + 1)
cen = 0.5 * (edges[:-1] + edges[1:])
X, Y = np.meshgrid(cen, cen, indexing="xy")     # X to the right, Y upward
 
lon = np.degrees(np.arctan2(Y, X))              # Galactic longitude, deg
rxy2 = X**2 + Y**2
 
col = np.zeros_like(X)                          # accumulated column, in E units
z_samples = np.arange(-Z_HALF + 0.5 * Z_STEP, Z_HALF, Z_STEP)
 
for k, z in enumerate(z_samples):
    d = np.sqrt(rxy2 + z * z)                   # heliocentric distance, pc
    ri = np.searchsorted(r_bounds, d) - 1       # radial shell index
    good = (ri >= 0) & (ri < n_r)               # drops inner 69 pc and beyond 1250 pc
    if not good.any():
        continue
    lat = np.degrees(np.arcsin(z / d[good]))    # Galactic latitude, deg
    ipix = hp.ang2pix(nside, lon[good], lat, nest=nest, lonlat=True)
    col[good] += dens[ri[good], ipix] * Z_STEP
    if k % 25 == 0:
        print(f"  z-plane {k+1}/{z_samples.size}")
 
col *= E_TO_AV                                  # -> A_V in magnitudes
 
# Mask what we do not want to show
r_full = np.sqrt(R_MAX**2 - Z_HALF**2)          # ~1184 pc
r_show = r_full if CLIP_TO_CYLINDER else R_MAX
col[rxy2 > r_show**2] = np.nan
 
print(f"A_V range: {np.nanmin(col):.3f} to {np.nanmax(col):.2f} mag")
 
# ----------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------
plt.rcParams.update({
    "font.size": 15,            # poster-legible
    "axes.labelsize": 17,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
})
 
fig, ax = plt.subplots(figsize=(8.5, 8.0))
 
vmax = np.nanpercentile(col, 99.9)
im = ax.imshow(
    col,
    origin="lower",
    extent=(-R_MAX, R_MAX, -R_MAX, R_MAX),
    cmap="Greys",               # white = no dust, black = dusty (paper convention)
    vmin=0.0,
    vmax=vmax,
    interpolation="nearest",
)
 
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, extend="max")
cb.set_label(rf"$A_V$ integrated over $|z| < {Z_HALF:.0f}$ pc  [mag]")
 
RED = "#e8000b"
 
 
def draw_sightline(ax, l_deg, d_start=69.0, d_end=None, b_deg=0.0,
                   label=None, color=RED, lw=2.6):
    """Draw a sightline at Galactic longitude l_deg, from d_start to d_end (pc).
 
    A sightline at latitude b is foreshortened in this projection: its
    apparent length is d * cos(b).
    """
    if d_end is None:
        d_end = r_show
    l = np.radians(l_deg)
    proj = np.cos(np.radians(b_deg))
    xs = np.array([d_start, d_end]) * proj * np.cos(l)
    ys = np.array([d_start, d_end]) * proj * np.sin(l)
 
    # A white halo keeps the line readable over both white voids and black clouds
    (line,) = ax.plot(
        xs, ys, color=color, lw=lw, solid_capstyle="round", zorder=5,
        path_effects=[pe.withStroke(linewidth=lw + 2.4, foreground="white")],
    )
    # Arrowhead at the far end
    ax.annotate(
        "", xy=(xs[1], ys[1]),
        xytext=(xs[0] + 0.92 * (xs[1] - xs[0]), ys[0] + 0.92 * (ys[1] - ys[0])),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                        mutation_scale=22, shrinkA=0, shrinkB=0),
        zorder=6,
    )
    if label:
        # Offset the text perpendicular to the line so it does not sit on it
        perp = np.array([-np.sin(l), np.cos(l)]) * 55.0
        mid = np.array([0.55 * (xs[0] + xs[1]), 0.55 * (ys[0] + ys[1])]) + perp
        ax.text(
            mid[0], mid[1], label, color=color, fontsize=15, fontweight="bold",
            ha="center", va="center", rotation=l_deg, rotation_mode="anchor",
            zorder=6,
            path_effects=[pe.withStroke(linewidth=3.0, foreground="white")],
        )
    return line
 
 
for l_deg, label in SIGHTLINES:
    draw_sightline(ax, l_deg, label=label)
 
# The Sun
ax.plot(0, 0, marker="o", ms=8, mfc="#ffd400", mec="k", mew=1.2, zorder=7)
 
ax.set_xlabel(r"$X$  [pc]")
ax.set_ylabel(r"$Y$  [pc]")
ax.set_xlim(-R_MAX, R_MAX)
ax.set_ylim(-R_MAX, R_MAX)
ax.set_aspect("equal")
 
fig.tight_layout()
fig.savefig("edenhofer_topdown.pdf")                 # vector text, for the poster
fig.savefig("edenhofer_topdown.png", dpi=400)
print("wrote edenhofer_topdown.pdf / .png")
 