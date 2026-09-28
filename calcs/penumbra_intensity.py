"""
Penumbra intensity for LEO sunshade elements.

There is no umbra unless the shade element is wider than the
penumbral spreading (2 x 4.652 mrad x slant distance). The relevant quantity
is the local blocking fraction: what percent of the solar disk a ground
observer sees occulted.

Model: long thin ribbon of width w (length >> width), slant distance d from
shade element to shadow point. The ribbon's angular half-width from the
ground is x = (w/2)/d, in units of the solar angular radius
ALPHA_S = 4.652 mrad (0.2665 deg). A strip of angular half-width x offset by
y from the center of the unit solar disk blocks the disk-area fraction:

    f(x, y) = [A(min(1, y+x)) - A(max(-1, y-x))] / pi,
    A(t) = t*sqrt(1-t^2) + arcsin(t)   (integral of chord 2*sqrt(1-t^2))

Peak blocking is at y=0. Penumbra full width on the ground (perpendicular to
the ribbon, before incidence-angle stretching) = w + 2*ALPHA_S*d.
Energy check: integral of f across the shadow = w (the ribbon's projected
width) - total blocked power is conserved no matter how diluted the shadow.

Incidence: projecting onto tilted ground stretches the footprint by
1/sin(sun elevation) and dilutes local W/m^2 by the same factor, but does
not change total blocked watts.
"""

import numpy as np
from leo_shade_calcs import sso_inclination_deg, shadow_points, OBLIQ

ALPHA_S = 4.652e-3   # rad, solar angular RADIUS
S0 = 1361.0          # W/m^2


def A(t):
    t = np.clip(t, -1.0, 1.0)
    return t * np.sqrt(1.0 - t * t) + np.arcsin(t)


def strip_block_fraction(x, y=0.0):
    """Fraction of solar disk area blocked by a strip of angular half-width
    x at offset y (both in units of solar angular radius)."""
    return max(0.0, (A(y + x) - A(y - x)) / np.pi)


def ribbon_case(w_m, d_km):
    d = d_km * 1e3
    x = (w_m / 2.0) / d / ALPHA_S
    peak = strip_block_fraction(x, 0.0)
    pen_width_km = (w_m + 2 * ALPHA_S * d) / 1e3
    # average blocking across the penumbra (energy / width)
    avg = (w_m / (pen_width_km * 1e3))
    umbra = w_m > 2 * ALPHA_S * d
    return peak, avg, pen_width_km, umbra


if __name__ == "__main__":
    print("Solar angular radius: %.3f mrad; penumbral spreading = "
          "%.2f km per 1000 km of slant range" % (ALPHA_S * 1e3,
                                                  2 * ALPHA_S * 1000))

    print("\nNADIR-ish geometry (slant ~ altitude): noon-midnight carpet")
    print(f"{'width':>8} {'alt km':>7} {'penumbra km':>12} {'peak %':>8} "
          f"{'avg %':>7} {'umbra?':>7}")
    for w in [100, 1000, 5000, 10000]:
        for d in [274, 350, 575, 1000]:
            peak, avg, pw, umb = ribbon_case(w, d)
            print(f"{w:7d}m {d:7d} {pw:12.2f} {100*peak:8.2f} "
                  f"{100*avg:7.2f} {str(umb):>7}")

    print("\nDAWN-DUSK geometry: slant range from first-principles sweep")
    for h in [274, 575]:
        i = sso_inclination_deg(h)
        lat, elev, sub, frac, slant = shadow_points(h, i, 18.0, OBLIQ)
        print(f"  h={h} km solstice: slant range {slant.min():.0f}.."
              f"{slant.max():.0f} km (median {np.median(slant):.0f} km)")
        d_med = float(np.median(slant))
        for w in [100, 1000, 5000, 10000]:
            peak, avg, pw, umb = ribbon_case(w, d_med)
            print(f"    w={w:6d} m at median slant: penumbra {pw:7.1f} km, "
                  f"peak block {100*peak:6.2f}%, avg {100*avg:5.2f}%")

    print("\nEnergy bookkeeping (per km of ribbon length, w=1 km):")
    print("  blocked power = S0 * w * 1km = %.2f MW per km of ribbon,"
          % (S0 * 1000 * 1000 / 1e6))
    print("  independent of slant range; dilution only spreads it.")
    print("  BUT at grazing incidence the blocked beam was arriving at")
    print("  sun elevation eps; the energy that ground would have ABSORBED")
    print("  is S0 * w * (1 - albedo(eps)), and albedo rises steeply at")
    print("  low eps (ocean: ~0.05 overhead, >0.3 below ~10 deg, >0.5")
    print("  below ~5 deg). Handle in radiative-effectiveness section.")
