"""
Elliptical 'heliotropic' sunshade case.

Concept (suggested in the literature, cf. Bewick et al. heliotropic dust
orbits): an elliptical orbit with apogee pointed at the Sun. The satellite
moves slowly through the distant sunward arc (casting shade near the
subsolar region, at good radiative incidence) and quickly through a low
perigee on the night side (minimizing time in the dense atmosphere).
The apsidal line must rotate ~0.9856 deg/day to keep tracking the Sun.

Part 1: shade-time fraction vs (perigee, apogee), exact Kepler solution,
        compared with circular orbits. Includes the closed form
        f ~= (R_E/(pi*a)) * sqrt(r_a/r_p), which shows f is MAXIMIZED at
        r_a = r_p: eccentricity always REDUCES total shade time relative
        to a circular orbit at the same perigee.
        -> report 2.4, the shade-fraction table
Part 2: whether the heliotropic orbit can be FLOWN at all, once SRP on a
        21 g/m^2 film is taken seriously. Two results:
          (a) The configuration is over-determined. Apogee points at the Sun
              only if the Sun lies in the orbit plane, so the node must be
              sun-synchronous AND the apsis must advance at +0.9856 deg/day.
              That is two conditions on one free parameter (i), because a
              and e are already pinned by the shading requirement. No
              (rp, ra) of interest closes both, and above ~10000 km apogee
              no sun-synchronous inclination exists at all.
          (b) SRP OPPOSES the requirement. With apogee at the Sun the SRP
              acceleration is anti-sunward, hence parallel to the
              eccentricity vector, so <de/dt> = 0 exactly but the apsis
              REGRESSES at ~1 deg/day. Closing the residual needs more
              acceleration than the sail can vector sideways (max 0.385 of
              its own SRP force, at 35.26 deg tilt, costing 18% of shade
              area), so the balance must come from propellant, forever.
        The contrast case - a circular noon-midnight SSO, where the apsis
        precesses freely and BOUNDS the SRP-driven eccentricity excursion -
        is computed in drag_reboost.py Part 4, not repeated here.
                                    -> report 2.4, the SRP acceleration and
                                       the "worse under SRP" claim

Run: python3 elliptical_heliotropic.py
"""

import functools

import numpy as np

MU = 3.986004418e14
J2 = 1.08262668e-3
R_EQ = 6378.137e3
R_E = 6371.0e3
SUN_RATE = 360.0 / 365.2422          # deg/day apsidal tracking requirement
S0 = 1361.0
D2R = np.pi / 180.0

C_LIGHT = 2.99792458e8
SIGMA = 0.021                        # kg/m^2 areal density (21 g/m^2 film)
P_SRP = 2 * S0 / C_LIGHT             # N/m^2, specular reflection, normal inc.
A_SRP = P_SRP / SIGMA                # m/s^2 characteristic SRP acceleration
STEER_FRAC = 2.0 / (3.0 * np.sqrt(3.0))   # 0.3849: max transverse fraction of
                                     # a flat specular sail's own SRP force,
                                     # achieved at 35.26 deg tilt
TILT_BEST = np.degrees(np.arcsin(1.0 / np.sqrt(3.0)))   # 35.264 deg
OBLIQ = 23.44                        # deg, max |solar declination|


def kepler_sweep(rp_km, ra_km, n=400001):
    rp, ra = rp_km * 1e3 + R_E, ra_km * 1e3 + R_E
    a, e = 0.5 * (rp + ra), (ra - rp) / (ra + rp)
    M = np.linspace(0, 2 * np.pi, n, endpoint=False)   # time-uniform
    E = M.copy()
    for _ in range(60):                                # Newton
        E -= (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
    nu = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2),
                        np.sqrt(1 - e) * np.cos(E / 2))
    nu = np.mod(nu + np.pi, 2 * np.pi) - np.pi   # wrap to (-pi, pi]
    r = a * (1 - e * np.cos(E))
    return a, e, nu, r


def shade_fraction_elliptical(rp_km, ra_km):
    """Apogee on the Earth-Sun line. Shade when the satellite is sunward
    (angle psi from sun line < 90 deg) with impact parameter r*sin(psi)<R_E.
    psi = pi - |nu| (nu measured from perigee)."""
    a, e, nu, r = kepler_sweep(rp_km, ra_km)
    psi = np.pi - np.abs(nu)
    shade = (psi < np.pi / 2) & (r * np.sin(psi) < R_E)
    return shade.mean(), a, e


def shade_fraction_circular(h_km):
    return np.arcsin(R_E / (R_E + h_km * 1e3)) / np.pi


def j2_apsidal_rate_deg_day(rp_km, ra_km, i_deg):
    rp, ra = rp_km * 1e3 + R_E, ra_km * 1e3 + R_E
    a, e = 0.5 * (rp + ra), (ra - rp) / (ra + rp)
    p = a * (1 - e * e)
    n = np.sqrt(MU / a**3)
    i = i_deg * D2R
    w_dot = 0.75 * n * J2 * (R_EQ / p) ** 2 * (5 * np.cos(i) ** 2 - 1)
    O_dot = -1.5 * n * J2 * (R_EQ / p) ** 2 * np.cos(i)
    # inertial drift of the apsis direction ~ w_dot + O_dot*cos(i)
    return np.degrees(w_dot) * 86400, np.degrees(O_dot) * 86400


# ----------------------------------------------------------------------
# Part 3 helpers: SRP as a perturbation on the osculating elements.
# ----------------------------------------------------------------------

_NG = 20001                                       # samples per orbit
_M_UNIFORM = np.linspace(0, 2 * np.pi, _NG, endpoint=False)


@functools.lru_cache(maxsize=128)
def _true_anomaly(e):
    """Time-uniform (mean-anomaly-uniform) sample of true anomaly."""
    E = _M_UNIFORM.copy()
    for _ in range(40):
        E -= (E - e * np.sin(E) - _M_UNIFORM) / (1 - e * np.cos(E))
    return 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2),
                          np.sqrt(1 - e) * np.cos(E / 2))


def gauss_avg(a, e, beta, acc):
    """Orbit-averaged de/dt [1/s] and dw/dt [rad/s] for a constant inertial
    acceleration of magnitude `acc` pointing ANTI-sunward, with the Sun at
    in-plane angle `beta` measured from the perigee direction.

    beta = pi is the heliotropic configuration (apogee at the Sun).
    Gauss variational equations, radial R and transverse T components."""
    nu = _true_anomaly(round(e, 9))
    n = np.sqrt(MU / a**3)
    s = np.sqrt(1 - e * e)
    R = -acc * np.cos(nu - beta)
    T = acc * np.sin(nu - beta)
    dedt = s / (n * a) * (R * np.sin(nu)
                          + T * (np.cos(nu)
                                 + (e + np.cos(nu)) / (1 + e * np.cos(nu))))
    dwdt = s / (n * a * e) * (-R * np.cos(nu)
                              + T * np.sin(nu) * (2 + e * np.cos(nu))
                              / (1 + e * np.cos(nu)))
    return dedt.mean(), dwdt.mean()


def wdot_sensitivity(a, e):
    """Max |dw/dt| [deg/day] obtainable per unit in-plane acceleration
    [m/s^2]. The averaged response is sinusoidal in the applied direction
    beta, so two quadrature evaluations give the envelope exactly."""
    _, w0 = gauss_avg(a, e, 0.0, 1.0)
    _, w90 = gauss_avg(a, e, np.pi / 2, 1.0)
    return np.degrees(np.hypot(w0, w90)) * 86400


def sso_inclination(a, e):
    """Inclination [deg] whose J2 nodal regression equals +SUN_RATE, i.e.
    the orbit plane keeps up with the Sun. NaN if no solution exists."""
    p = a * (1 - e * e)
    n = np.sqrt(MU / a**3)
    c = -(SUN_RATE * D2R / 86400) / (1.5 * n * J2 * (R_EQ / p) ** 2)
    return np.degrees(np.arccos(c)) if abs(c) <= 1 else np.nan


def elements(rp_km, ra_km):
    rp, ra = rp_km * 1e3 + R_E, ra_km * 1e3 + R_E
    return 0.5 * (rp + ra), (ra - rp) / (ra + rp)


if __name__ == "__main__":
    print("PART 1: shade-time fraction (apogee locked on Sun)")
    print("  Circular reference:")
    for h in [500, 1000, 2000, 10000]:
        print(f"    circular {h:5d} km: f_shade = "
              f"{100*shade_fraction_circular(h):5.1f}%")
    print("  Elliptical (perigee km x apogee km):")
    for rp in [500, 1000, 2000]:
        for ra in [3000, 8000, 20000, 35786]:
            f, a, e = shade_fraction_elliptical(rp, ra)
            T_h = 2 * np.pi * np.sqrt(a**3 / MU) / 3600
            print(f"    {rp:4d} x {ra:5d}: e={e:5.3f}  T={T_h:5.2f} h  "
                  f"f_shade={100*f:5.1f}%")
    print("  Closed form: f ~ (R_E/(pi a))*sqrt(r_a/r_p);")
    print("  d f/d r_a = 0 at r_a = r_p -> eccentricity never beats the")
    print("  circular orbit at the same perigee on total shade time.")

    ELLIPTICALS = [(500, 3000), (500, 8000), (500, 20000), (1000, 20000)]

    print("\nPART 2: SRP and the cost of holding a heliotropic orbit")
    print(f"  film {1e3*SIGMA:.0f} g/m^2 -> A/m = {1/SIGMA:.1f} m^2/kg")
    print(f"  P_srp = 2 S0/c = {P_SRP:.3e} N/m^2 (specular, normal incidence)")
    print(f"  a_srp = {A_SRP:.3e} m/s^2 = {A_SRP*86400:.1f} m/s per day")
    print(f"  a flat specular sail can vector at most {STEER_FRAC:.4f} of that")
    print(f"  sideways, at {TILT_BEST:.2f} deg tilt, which costs "
          f"{100*(1-np.cos(TILT_BEST*D2R)):.1f}% of projected shade area.")

    print("\n  2A. The heliotropic orbit is over-determined.")
    print("      Apogee can only point at the Sun if the Sun lies IN the")
    print("      orbit plane, so the node must be sun-synchronous AND the")
    print("      apsis must advance at +%.4f deg/day. Two conditions;"
          % SUN_RATE)
    print("      a and e are already fixed by shading, leaving only i.")
    print(f"  {'rp x ra':>13} {'e':>6} {'i_sso':>8} {'wdot_J2':>9} "
          f"{'residual':>9}")
    for (rp, ra) in ELLIPTICALS:
        a, e = elements(rp, ra)
        i_s = sso_inclination(a, e)
        if np.isnan(i_s):
            print(f"  {rp:5d} x {ra:<5d} {e:6.3f} {'--':>8} {'--':>9} "
                  f"{'NO SUN-SYNC i EXISTS':>9}")
            continue
        w, _ = j2_apsidal_rate_deg_day(rp, ra, i_s)
        print(f"  {rp:5d} x {ra:<5d} {e:6.3f} {i_s:8.2f} {w:+9.3f} "
              f"{SUN_RATE - w:+9.3f}")
    print("      No row closes. J2 cannot satisfy both conditions at once.")

    print("\n  2B. Control burden to force the apsidal sun-lock.")
    print("      wdot_SRP is the apsis rate SRP itself produces at beta=pi.")
    print(f"  {'rp x ra':>13} {'i':>6} {'e':>6} {'wJ2':>8} {'wSRP':>8} "
          f"{'need':>8} {'a/a_srp':>8} {'m/s/day':>8}  verdict")
    for (rp, ra) in ELLIPTICALS:
        a, e = elements(rp, ra)
        _, w_srp_rad = gauss_avg(a, e, np.pi, A_SRP)
        w_srp = np.degrees(w_srp_rad) * 86400
        for i in [0.0, 23.44, 63.43]:
            w_j2, _ = j2_apsidal_rate_deg_day(rp, ra, i)
            need = SUN_RATE - w_j2 - w_srp
            a_req = abs(need) / wdot_sensitivity(a, e)
            if i < OBLIQ:
                verdict = "GEOM INVALID (Sun not in plane)"
            elif a_req > STEER_FRAC * A_SRP:
                verdict = "SAIL CANNOT: needs propellant"
            else:
                verdict = "sail could steer this"
            print(f"  {rp:5d} x {ra:<5d} {i:6.2f} {e:6.3f} {w_j2:+8.3f} "
                  f"{w_srp:+8.3f} {need:+8.3f} {a_req/A_SRP:8.2f} "
                  f"{a_req*86400:8.1f}  {verdict}")
    print("      SRP opposes the required precession (wSRP < 0 everywhere),")
    print("      so it is a headwind, not a resource. i < %.2f deg rows are"
          % OBLIQ)
    print("      geometrically invalid: an equatorial plane contains the Sun")
    print("      line only at the equinoxes.")
    print("      a/a_srp assumes the control accel is applied in the single")
    print("      most efficient in-plane direction, so it is a BEST CASE")
    print("      lower bound. Any real steering law does worse.")

    print("\n  Conclusion. At %.0f g/m^2 SRP breaks BOTH geometries, but"
          % (1e3 * SIGMA))
    print("  not in the same way. The circular SSO fails passively and with")
    print("  a bounded excursion (drag_reboost.py Part 4), and mass buys the")
    print("  fix outright. The elliptical heliotropic orbit fails as a")
    print("  CONTROL problem that mass does not fix, since a heavier sail")
    print("  loses steering authority in the same proportion. It also demands")
    print("  continuous off-Sun tilt, which spends shade area to buy")
    print("  station-keeping. Eccentricity already lost on shade fraction in")
    print("  Part 1; Part 2 says it also cannot be flown.")
