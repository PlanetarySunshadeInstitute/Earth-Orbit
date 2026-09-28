"""
Shade area required for a given insolation reduction.
1000 km noon-midnight SSO, horizontal ("magic carpet") attitude.

Earth intercepts sunlight over a disk of area pi * R_E^2. A shade element
removes flux equal to its area projected normal to the Sun line, but only
while its shadow falls on Earth. The year-averaged blocked area of one
craft is A_craft * eta, with

    eta = < max(n_hat . s_hat, 0) * 1[shading] >

where n_hat is the sail normal and s_hat points from Earth to the Sun.
For the horizontal carpet n_hat is the local vertical, so eta combines the
orbital duty cycle with the cosine projection of the sail onto the Sun line.

The fleet required for a fractional insolation reduction f is

    N = f * pi * R_E^2 / (A_craft * eta)

Penumbral spreading does not enter: the flux removed from the beam is
conserved regardless of how the shadow is smeared, provided it lands on
Earth. See penumbra_intensity.py for the intensity distribution.
"""

import numpy as np

from leo_shade_calcs import (R_E, D2R, sun_hat, sso_inclination_deg,
                             raan_for_ltan, declination_series)

ALT_KM = 1000.0
A_CRAFT = 100.0 * 10_000.0        # m^2, 100 m x 10 km = 1.0 km^2
LTAN_H = 12.0                     # noon-midnight
ALPHA_SUN_DEG = 90.0


def orbit_positions(h_km, delta_deg, n_u=4000):
    """Positions around one circular SSO, and the Sun direction."""
    a = R_E + h_km * 1e3
    i = sso_inclination_deg(h_km) * D2R
    O = raan_for_ltan(LTAN_H, ALPHA_SUN_DEG) * D2R
    u = np.linspace(0.0, 2.0 * np.pi, n_u, endpoint=False)
    e1 = np.array([np.cos(O), np.sin(O), 0.0])
    e2 = np.array([-np.sin(O) * np.cos(i), np.cos(O) * np.cos(i), np.sin(i)])
    r = a * (np.outer(np.cos(u), e1) + np.outer(np.sin(u), e2))
    return r, sun_hat(delta_deg, ALPHA_SUN_DEG)


def shading_mask(r, s):
    """True where the craft is sunward of Earth and its shadow meets the disk."""
    rs = r @ s
    disc = rs ** 2 - (np.sum(r * r, axis=1) - R_E ** 2)
    return (disc >= 0) & (rs > 0)


def eta_year(h_km, n_days=366):
    """Year-averaged blocked-area fraction, and the duty cycle alone."""
    _, dec = declination_series(n_days)
    duty = np.empty(n_days)
    eff = np.empty(n_days)
    for k, d in enumerate(dec):
        r, s = orbit_positions(h_km, d)
        m = shading_mask(r, s)
        r_hat = r / np.linalg.norm(r, axis=1, keepdims=True)
        eff[k] = (np.clip(r_hat @ s, 0.0, None) * m).mean()
        duty[k] = m.mean()
    return eff.mean(), duty.mean()


def fleet_for_fraction(f, eta, a_craft=A_CRAFT):
    """Craft count and total film area for insolation reduction f."""
    n = f * np.pi * R_E ** 2 / (a_craft * eta)
    return n, n * a_craft


if __name__ == "__main__":
    a_disk = np.pi * R_E ** 2
    eta, duty = eta_year(ALT_KM)

    print(f"Earth intercept disk      {a_disk/1e6:,.0f} km^2")
    print(f"shade element             {A_CRAFT/1e6:.1f} km^2 (100 m x 10 km)")
    print(f"altitude                  {ALT_KM:.0f} km, LTAN {LTAN_H:.0f}:00")
    print()
    print(f"duty cycle                {100*duty:.1f}% of each orbit")
    print(f"cosine projection         {eta/duty:.3f}")
    print(f"eta                       {100*eta:.1f}%")
    print()
    print(f"  {'insolation cut':>14s}  {'blocking area km^2':>19s}  "
          f"{'craft':>10s}  {'film area km^2':>15s}")
    for f in [1e-3, 1e-2]:
        n, area = fleet_for_fraction(f, eta)
        print(f"  {100*f:13.1f}%  {f*a_disk/1e6:19,.0f}  {n:10,.0f}  "
              f"{area/1e6:15,.0f}")
