"""
Noon-midnight SSO shade fraction at the report's sample altitudes.

For each altitude, the fraction of every orbit during which the craft sits
between the Sun and the Earth disk (and so casts shade on Earth), evaluated
for every day of the year. Reuses the geometry in noon_midnight_calcs.py
(orbit_state, shade_mask), so the values match that script's Part 1
exactly at the altitudes the two share.

Also prints the SSO inclination at each altitude and the matching
eclipse fraction (identical to shade fraction for noon-midnight).

Run from inside this directory: python3 shade_fraction_table.py
"""

import numpy as np
from leo_shade_calcs import sso_inclination_deg, declination_series
from noon_midnight_calcs import orbit_state, shade_mask, eclipse_mask

ALTITUDES_KM = [274, 350, 500, 575, 800, 1000, 2000]
LTAN_HOURS = 12.0      # noon-midnight
N_U = 4000             # samples per orbit (same as noon_midnight_calcs.py)


def shade_fraction_over_year(h_km):
    _, dec = declination_series(366)
    shade, ecl = [], []
    for d in dec:
        a, r, _, s = orbit_state(h_km, LTAN_HOURS, d, n_u=N_U)
        m, _, _ = shade_mask(a, r, s)
        shade.append(m.mean())
        ecl.append(eclipse_mask(a, r, s).mean())
    return np.array(shade), np.array(ecl)


if __name__ == "__main__":
    print("Noon-midnight SSO (LTAN 12:00): shade fraction per orbit,")
    print("evaluated on every day of the year\n")
    print(f"{'Altitude':>10} {'SSO incl.':>10} {'Shade frac. (mean)':>19}"
          f" {'Year range':>17} {'Eclipse (mean)':>15}")
    rows = []
    for h in ALTITUDES_KM:
        fr, ec = shade_fraction_over_year(h)
        rows.append((h, fr.mean()))
        print(f"{h:>7,d} km {sso_inclination_deg(h):>9.1f}° "
              f"{100*fr.mean():>18.1f}% "
              f"{100*fr.min():>8.1f}–{100*fr.max():.1f}% "
              f"{100*ec.mean():>14.1f}%")

    print("\nMarkdown table for the report:\n")
    print("| Orbit type | Altitude | Shade fraction |")
    print("| --- | --- | --- |")
    for h, f in rows:
        print(f"| circular (noon-midnight SSO) | {h:,d} km | {100*f:.1f}% |")
