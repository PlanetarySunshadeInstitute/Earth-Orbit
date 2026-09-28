"""
Noon-midnight SSO sunshade: first-principles geometry.

SSO with local time of ascending node 12:00
(or 00:00). The orbit plane contains (approximately) the Earth-Sun line, so
the satellite crosses near the subsolar point on every orbit.

Part 1: shade-time fraction per orbit, year round (vs dawn-dusk's
        seasonal on/off behaviour).                          -> report 3.2
Part 2: attitude coupling - the geometric reason noon-midnight forces a
        choice between sun-facing (max shade, catastrophic ram drag) and
        horizontal/nadir-facing "magic carpet" (always edge-on to ram,
        shade effectiveness reduced by cos of the sun angle). The cosine
        projection derived here is what constellation_sizing.py uses.
                                                        -> report 3.2, 4.4
Part 3: eclipse fraction, on the same geometry (the power problem).
                                                             -> report 3.2

Run: python3 noon_midnight_calcs.py
"""

import numpy as np
from leo_shade_calcs import (sso_inclination_deg, sun_hat, raan_for_ltan,
                             OBLIQ, R_E, D2R, declination_series)


def orbit_state(h_km, ltan_hours, delta_deg, alpha_sun_deg=90.0, n_u=20000):
    """Positions and unit velocity directions around one circular orbit."""
    a = R_E + h_km * 1e3
    i = sso_inclination_deg(h_km) * D2R
    O = raan_for_ltan(ltan_hours, alpha_sun_deg) * D2R
    u = np.linspace(0, 2 * np.pi, n_u, endpoint=False)
    # basis vectors in orbit plane
    e1 = np.array([np.cos(O), np.sin(O), 0.0])               # node direction
    e2 = np.array([-np.sin(O) * np.cos(i), np.cos(O) * np.cos(i), np.sin(i)])
    r = a * (np.outer(np.cos(u), e1) + np.outer(np.sin(u), e2))
    v_hat = -np.outer(np.sin(u), e1) + np.outer(np.cos(u), e2)
    s = sun_hat(delta_deg, alpha_sun_deg)
    return a, r, v_hat, s


def shade_mask(a, r, s):
    rs = r @ s
    disc = rs**2 - (np.sum(r * r, axis=1) - R_E**2)
    return (disc >= 0) & (rs > 0), rs, disc


def eclipse_mask(a, r, s):
    rs = r @ s
    rho2 = np.sum(r * r, axis=1) - rs**2
    return (rs < 0) & (rho2 < R_E**2)


if __name__ == "__main__":
    print("PART 1: shade-time fraction per orbit, through the year")
    print("  (noon-midnight LTAN 12:00; compare dawn-dusk which is zero")
    print("   near equinoxes and grazing when nonzero)")
    for h in [274, 350, 575, 1000]:
        days, dec = declination_series(366)
        fr = []
        for d in dec:
            a, r, v, s = orbit_state(h, 12.0, d, n_u=4000)
            m, _, _ = shade_mask(a, r, s)
            fr.append(m.mean())
        fr = np.array(fr)
        print(f"  h={h:4d} km: shade fraction {100*fr.min():.1f}%"
              f" - {100*fr.max():.1f}% of every orbit, all 365 days")

    print("\nPART 2: attitude coupling (the forced choice)")
    print("  Evaluated at 1000 km, averaged over the year,")
    print("  on the same basis as constellation_sizing.py.")
    h = 1000
    days, dec = declination_series(366)
    ram_sun, proj_arc, duty = [], [], []
    lo, hi = 1.0, 0.0
    for d in dec:
        a, r, v_hat, s = orbit_state(h, 12.0, d, n_u=4000)
        m, _, _ = shade_mask(a, r, s)
        # (i) sun-facing sail: full shade projection, but the ram projection
        #     |v.s| swings face-on to the airstream twice per orbit.
        ram_sun.append(np.abs(v_hat @ s).mean())
        # (ii) horizontal carpet (normal = local vertical): edge-on to ram
        #      always; shade projection = cos(solar zenith angle at the craft)
        r_hat = r / np.linalg.norm(r, axis=1, keepdims=True)
        p = np.clip(r_hat @ s, 0.0, None)
        proj_arc.append((p * m).mean())
        duty.append(m.mean())
        lo, hi = min(lo, p[m].min()), max(hi, p[m].max())
    ram_sun = float(np.mean(ram_sun))
    cos_proj = float(np.mean(proj_arc)) / float(np.mean(duty))
    print(f"  Sun-facing sail: full shade projection during the shading arc,")
    print(f"    but mean |v.s| ram projection over the orbit = "
          f"{ram_sun:.3f} of full sail area")
    print(f"  Horizontal 'magic carpet': edge-on to ram ALWAYS (ram")
    print(f"    projection ~ flatness/attitude error only); shade")
    print(f"    projection while shading: mean = {cos_proj:.3f}, "
          f"range {lo:.3f}..{hi:.3f}")
    print("  -> the horizontal carpet is the flyable option: it gives up")
    print(f"     {100*(1-cos_proj):.0f}% of shade effectiveness to stay flyable.")
    print(f"     Effective blocked-area fraction eta = duty x cos projection")
    print(f"     = {np.mean(duty):.3f} x {cos_proj:.3f} = "
          f"{np.mean(duty)*cos_proj:.3f}")

    print("\nPART 3: eclipse fraction (power)")
    for h in [274, 350, 575, 1000]:
        days, dec = declination_series(366)
        fr = []
        for d in dec:
            a, r, v, s = orbit_state(h, 12.0, d, n_u=4000)
            fr.append(eclipse_mask(a, r, s).mean())
        fr = np.array(fr)
        print(f"  h={h:4d} km: in Earth's shadow {100*fr.min():.1f}%"
              f" - {100*fr.max():.1f}% of every orbit, year round")
    print("  Dawn-dusk SSO by comparison: ~0-2% eclipse. The eclipse arc")
    print("  coincides with the shading arc, so storage and array margin")
    print("  must be carried within the 21 g/m^2 areal budget.")
