"""
LEO sunshade geometry, from first principles.
Shared constants and helpers for the other scripts in this directory.

Conventions:
- ECI-like frame, Earth center at origin, Z = north pole.
- Sun direction s_hat at solar declination delta and right ascension alpha:
  s_hat = (cos d cos a, cos d sin a, sin d). Points FROM Earth TOWARD Sun.
- Spherical Earth, R_E = 6371 km for shadow geometry (volumetric mean).
  R_eq = 6378.137 km used only in the J2 precession formula.
- Sun treated as a point source for shade-existence geometry; penumbral
  spreading handled separately (0.53 deg solar angular diameter).

Part 1: SSO inclination vs altitude (J2 nodal precession).  -> report 3.1
Part 2: Shading season duration vs altitude, dawn-dusk.     -> report 3.2

This is also the shared geometry module for the rest of the directory:
noon_midnight_calcs.py, constellation_sizing.py and
penumbra_intensity.py import sun_hat, sso_inclination_deg,
raan_for_ltan, declination_series and shadow_points from here.

Run: python3 leo_shade_calcs.py
"""

import numpy as np

# Constants
MU = 3.986004418e14        # m^3/s^2
J2 = 1.08262668e-3
R_EQ = 6378.137e3          # m, equatorial (for J2 formula)
R_E = 6371.0e3             # m, mean (for shadow geometry)
YEAR_S = 365.2422 * 86400.0
SSO_RATE = 2.0 * np.pi / YEAR_S   # rad/s, required nodal precession (eastward)
OBLIQ = 23.44              # deg
SUN_HALF_ANGLE = 0.2665    # deg, solar angular radius from Earth

D2R = np.pi / 180.0


# ---------- Part 1: SSO inclination vs altitude ----------

def sso_inclination_deg(h_km):
    """Inclination (deg) for a circular sun-synchronous orbit at altitude h_km."""
    a = R_EQ + h_km * 1e3   # use equatorial radius datum, standard practice
    n = np.sqrt(MU / a**3)
    cos_i = -SSO_RATE / (1.5 * J2 * n * (R_EQ / a) ** 2)
    return np.degrees(np.arccos(cos_i))


# ---------- Part 2: orbit plane vs sun geometry ----------

def sun_hat(delta_deg, alpha_deg):
    d, al = delta_deg * D2R, alpha_deg * D2R
    return np.array([np.cos(d) * np.cos(al), np.cos(d) * np.sin(al), np.sin(d)])


def orbit_normal(i_deg, raan_deg):
    """Angular momentum direction for inclination i, RAAN (from X axis)."""
    i, O = i_deg * D2R, raan_deg * D2R
    return np.array([np.sin(O) * np.sin(i), -np.cos(O) * np.sin(i), np.cos(i)])


def raan_for_ltan(ltan_hours, alpha_sun_deg):
    """RAAN such that ascending node sits at local solar time ltan_hours."""
    return alpha_sun_deg + (ltan_hours - 12.0) * 15.0


def tilt_from_terminator_deg(i_deg, ltan_hours, delta_deg, alpha_sun_deg=90.0):
    """Dihedral angle between orbit plane and terminator plane (deg).
    Terminator plane normal = s_hat. Tilt = angle(n_hat, s_hat), folded to <=90."""
    s = sun_hat(delta_deg, alpha_sun_deg)
    n = orbit_normal(i_deg, raan_for_ltan(ltan_hours, alpha_sun_deg))
    ang = np.degrees(np.arccos(np.clip(abs(np.dot(n, s)), -1, 1)))
    return ang


# ---------- Part 3: shade existence ----------

def shade_exists(h_km, i_deg, ltan_hours, delta_deg, alpha_sun_deg=90.0):
    """A point-sun shadow from somewhere on the orbit reaches Earth iff the
    orbit's minimum impact parameter about the sun line < R_E, on the day side.
    rho_min = a * cos(theta), theta = tilt from terminator."""
    a = R_E + h_km * 1e3
    theta = tilt_from_terminator_deg(i_deg, ltan_hours, delta_deg, alpha_sun_deg)
    return a * np.cos(theta * D2R) < R_E, theta


# ---------- shadow footprint (used by penumbra_intensity.py) ----------

def shadow_points(h_km, i_deg, ltan_hours, delta_deg, alpha_sun_deg=90.0, n_u=20000):
    """Sweep the orbit; for each satellite position whose anti-sun ray hits
    Earth, return shadow point latitude (deg) and sun elevation there (deg)."""
    a = R_E + h_km * 1e3
    s = sun_hat(delta_deg, alpha_sun_deg)
    O = raan_for_ltan(ltan_hours, alpha_sun_deg) * D2R
    i = i_deg * D2R
    u = np.linspace(0, 2 * np.pi, n_u, endpoint=False)
    r = a * np.stack([
        np.cos(u) * np.cos(O) - np.sin(u) * np.cos(i) * np.sin(O),
        np.cos(u) * np.sin(O) + np.sin(u) * np.cos(i) * np.cos(O),
        np.sin(u) * np.sin(i)], axis=1)
    rs = r @ s
    # ray: p = r - t*s, t>0. Solve |p|=R_E: t^2 - 2 t (r.s) + a^2 - R_E^2 = 0
    disc = rs**2 - (a**2 - R_E**2)
    hit = (disc >= 0) & (rs > 0)
    t = rs[hit] - np.sqrt(disc[hit])          # first intersection (sun side)
    p = r[hit] - t[:, None] * s
    lat = np.degrees(np.arcsin(np.clip(p[:, 2] / R_E, -1, 1)))
    sun_elev = 90.0 - np.degrees(np.arccos(np.clip((p @ s) / R_E, -1, 1)))
    sublat = np.degrees(np.arcsin(np.clip(r[hit, 2] / a, -1, 1)))
    return lat, sun_elev, sublat, hit.sum() / n_u, t / 1e3  # t in km


# ---------- Part 2: shading season ----------

def declination_series(n=3653):
    """Approx solar declination over a year (deg), and alpha (deg) - we keep
    alpha fixed at 90 and vary only declination, which is equivalent for this
    geometry because the SSO plane tracks the sun in right ascension."""
    days = np.linspace(0, 365.2422, n)
    dec = np.degrees(np.arcsin(np.sin(OBLIQ * D2R) *
                               np.sin(2 * np.pi * (days - 80.0) / 365.2422)))
    return days, dec


def shade_days_per_year(h_km, ltan_hours):
    i = sso_inclination_deg(h_km)
    days, dec = declination_series()
    ok = np.array([shade_exists(h_km, i, ltan_hours, d)[0] for d in dec])
    frac = ok.mean()
    return frac * 365.2422


# ---------- Output ----------

if __name__ == "__main__":
    print("=" * 70)
    print("PART 1: SSO inclination vs altitude")
    print("  Sun-synchronicity couples altitude to inclination; all are")
    print("  retrograde of polar.")
    for h in [274, 350, 500, 575, 700, 894, 1100, 1300, 1681]:
        print(f"  h={h:5d} km  i={sso_inclination_deg(h):7.2f} deg")

    print("\n" + "=" * 70)
    print("PART 2: days per year with any shade, dawn-dusk (best LTAN)")
    print("  Point-sun shade existence: the shadow reaches Earth only when")
    print("  the orbit plane tilts far enough off the terminator, which")
    print("  happens seasonally and only near one pole.")
    for h in [200, 274, 350, 500, 575, 800, 1000, 1200]:
        d6 = shade_days_per_year(h, 6.0)
        d18 = shade_days_per_year(h, 18.0)
        print(f"  h={h:5d} km  LTAN6: {d6:6.1f} d/yr   LTAN18: {d18:6.1f} d/yr")
    print("  Noon-midnight, by contrast, shades on every orbit all year;")
    print("  see noon_midnight_calcs.py.")
