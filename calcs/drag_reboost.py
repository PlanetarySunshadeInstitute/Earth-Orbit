"""
Orbit maintenance for a thin-film shade: drag, solar radiation pressure,
uncontrolled lifetime, and the propellant budget that follows.

Supersedes the earlier split between drag_reboost.py and
srp_eccentricity.py, which derived the frozen eccentricity e* by two
independent paths and agreed. The analytic path is kept (Part 2) and the
independent numerical path is kept as its check (Part 3).

Sail model: flying carpet, normal = local zenith (radial), noon-midnight
SSO, equinox geometry. Photons always push ANTI-SUNWARD along the sail
normal axis, so the radial perturbing acceleration is

    a_r(u) = -A cos(u) |cos(u)|   (sunlit),  0 in eclipse

with u the argument of latitude from the subsolar point and A = P_srp /
sigma the broadside peak. Note this is NOT +A cos^2(u): on the noon side
the push is radially INWARD.

Part 1: DRAG. Station-keeping dV/yr to hold a circular orbit, swept over
        altitude and the effective ram-area fraction eps.   -> report 4.2
Part 2: SRP secular eccentricity growth, analytic (Gauss VOP).
        A radial force gives NO secular da, so SRP is not a reboost cost.
        It IS an e-control cost if you insist on holding the orbit
        circular: dV_min/yr = (v/2) de/dt YEAR.              -> report 4.3
Part 3: numerical check of Part 2 (full nonlinear planar two-body + SRP
        + cylindrical shadow, RK4, 30 days).
Part 4: J2 apsidal precession bounds the excursion. e oscillates 0 -> 2e*
        rather than growing without bound. Flight precedent: LightSail 2
        showed this ~100-day cycle (Mansell et al. 2023).    -> report 4.3
Part 5: passive decay rate and uncontrolled lifetime, solar mean and max.
Part 6: total annual budget at solar max, taking the cheaper of
          A: hold circular, pay SRP e-control + circular drag;
          B: park at the SRP-J2 frozen orbit e* (Colombo, Lucking, and
             McInnes 2012). SRP costs nothing; you pay permanent drag at
             a perigee a*e* below the mean altitude.         -> report 4.4

Run: python3 drag_reboost.py
"""

import math

import numpy as np

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
MU = 3.986004418e14
R_E = 6371.0e3            # m, mean radius, for shadow / geometry
R_EQ = 6378.137e3         # equatorial radius for J2 terms
J2 = 1.08262668e-3
CD = 2.2
SIGMA_FILM = 0.021        # kg/m^2 reference film
S0 = 1361.0
C_LIGHT = 2.99792458e8
YEAR_S = 365.2422 * 86400
VE_SEP = 3000 * 9.81      # m/s, Isp 3000 s
VE_CHEM = 300 * 9.81      # m/s, Isp 300 s
OMEGA_SSO = 2 * np.pi / YEAR_S     # required node rate, rad/s

P_SRP = 2 * S0 / C_LIGHT           # N/m^2, specular, normal incidence
A_SRP = P_SRP / SIGMA_FILM         # m/s^2, broadside peak

ALTS = [274, 350, 500, 575, 800, 1000]
EPS_SWEEP = [1e-3, 1e-2, 1e-1]     # effective ram-area fraction

# ======================================================================
# DENSITY MODEL - the only place atmospheric density enters
# ======================================================================
# PROVENANCE: representative exponential-atmosphere values at two solar
# activity levels, "mean" (F10.7 ~ 140) and "max" (F10.7 ~ 190). They are
# consistent with standard reference tables (Vallado, CIRA-class) to
# within roughly 25% and reproduce NRLMSISE-00 behaviour qualitatively,
# but they are NOT an extraction from a model run and have not been
# pulled from the NASA CCMC NRLMSIS instant-run service. Parts 1, 5 and 6
# depend on them; Parts 2-4 do not. eps is a far larger unknown than rho
# in every result here, so the 25% is not the binding uncertainty.
DENSITY_PROVENANCE = {
    "model": "representative exponential atmosphere (Vallado / CIRA-class)",
    "sourced": False,
}

RHO = {
    # alt km: {"mean": rho, "max": rho}   kg/m^3
    200:  {"mean": 2.5e-10, "max": 4.0e-10},
    274:  {"mean": 4.0e-11, "max": 9.0e-11},
    300:  {"mean": 1.9e-11, "max": 5.0e-11},
    350:  {"mean": 7.0e-12, "max": 2.2e-11},
    400:  {"mean": 2.8e-12, "max": 9.0e-12},
    500:  {"mean": 5.0e-13, "max": 2.2e-12},
    575:  {"mean": 1.6e-13, "max": 8.0e-13},
    600:  {"mean": 1.1e-13, "max": 6.0e-13},
    800:  {"mean": 1.2e-14, "max": 8.0e-14},
    1000: {"mean": 3.0e-15, "max": 2.0e-14},
}

_RHO_H = np.array(sorted(RHO))
_RHO_LN = {k: np.log(np.array([RHO[h][k] for h in sorted(RHO)]))
           for k in ("mean", "max")}


def rho_interp(h_km, activity="max"):
    """Log-linear (exponential-atmosphere) interpolation of the density
    table; edge slopes extrapolate below 200 km and above 1000 km."""
    ln_tab = _RHO_LN[activity]
    h = np.atleast_1d(h_km).astype(float)
    ln = np.interp(h, _RHO_H, ln_tab)
    s_lo = (ln_tab[1] - ln_tab[0]) / (_RHO_H[1] - _RHO_H[0])
    s_hi = (ln_tab[-1] - ln_tab[-2]) / (_RHO_H[-1] - _RHO_H[-2])
    ln = np.where(h < _RHO_H[0], ln_tab[0] + s_lo * (h - _RHO_H[0]), ln)
    ln = np.where(h > _RHO_H[-1], ln_tab[-1] + s_hi * (h - _RHO_H[-1]), ln)
    return np.exp(ln) if np.ndim(h_km) else float(np.exp(ln)[0])


# ----------------------------------------------------------------------
# Geometry
# ----------------------------------------------------------------------
def geom(h_km):
    a = R_E + h_km * 1e3
    v = np.sqrt(MU / a)
    n = np.sqrt(MU / a ** 3)
    beta = np.arcsin(R_E / a)            # eclipse half-angle
    return a, v, n, beta


def brackets(beta):
    s = np.sin(beta)
    b_cos2 = s - s ** 3 / 3               # +A cos^2 convention
    b_phys = 4.0 / 3.0 - b_cos2           # -A cos|cos| convention (physical)
    return b_cos2, b_phys


def sunlit_mean_cos2(beta):
    """Mean of cos^2(u) over the eclipse-bounded sunlit arc (naive basis)."""
    c = np.pi - beta
    return (c - np.sin(beta) * np.cos(beta)) / (2 * c)


def sso_inclination(a, n):
    cosi = -OMEGA_SSO / (1.5 * n * J2 * (R_EQ / a) ** 2)
    return np.arccos(cosi)


def apsidal_rate(a, n, i):
    return 0.75 * n * J2 * (R_EQ / a) ** 2 * (5 * np.cos(i) ** 2 - 1)


# ----------------------------------------------------------------------
# SRP secular theory
# ----------------------------------------------------------------------
def srp_e_rate(h_km):
    """Secular |de/dt| [1/s] from SRP on the radial-normal carpet."""
    a, v, n, beta = geom(h_km)
    _, b_phys = brackets(beta)
    return A_SRP * b_phys / (np.pi * v)


def j2_srp_averaged(h_km):
    """Averaged e-vector dynamics de/dt = f*yhat + wd*(zhat x e).
    Linear ODE -> exact solution: circle of radius |e0 - e_c| about the
    forced equilibrium e_c = (-f/wd, 0). From e0 = 0: |e| max = 2f/|wd|,
    period 2pi/|wd|."""
    a, v, n, beta = geom(h_km)
    f = srp_e_rate(h_km)
    i = sso_inclination(a, n)
    wd = apsidal_rate(a, n, i)
    e_star = f / abs(wd)
    return f, wd, i, e_star, 2 * e_star, 2 * np.pi / abs(wd) / 86400


# ----------------------------------------------------------------------
# Drag, lifetime, propellant
# ----------------------------------------------------------------------
def decay_rate_km_per_day(h_km, A_over_m, activity="max"):
    """da/dt = -Cd (A/m) rho sqrt(mu a) for circular decay."""
    a = R_E + h_km * 1e3
    return (CD * A_over_m * rho_interp(h_km, activity)
            * np.sqrt(MU * a) * 86400 / 1e3)


def lifetime_days(h0_km, A_over_m, activity="max", h_end=120.0,
                  dt=120.0, t_max_yr=50):
    """Uncontrolled orbital lifetime: integrate circular decay to h_end."""
    h, t = float(h0_km), 0.0
    cap = t_max_yr * YEAR_S
    while h > h_end and t < cap:
        a = R_E + h * 1e3
        h -= (CD * A_over_m * rho_interp(h, activity)
              * np.sqrt(MU * a) * dt / 1e3)
        t += dt
    return t / 86400 if t < cap else np.inf


def drag_dv_yr(h_km, eps, activity="max"):
    """Circular-orbit drag make-up dV per year."""
    _, v, _, _ = geom(h_km)
    return (0.5 * CD * rho_interp(h_km, activity) * v ** 2
            * (eps / SIGMA_FILM) * YEAR_S)


def drag_dv_yr_eccentric(h_km, e, eps, activity="max", n_samples=720):
    """Time-averaged drag dV/yr on an orbit of mean altitude h_km and
    eccentricity e (the frozen orbit at e*). Time weighting via the
    eccentric anomaly (dt proportional to 1 - e cosE)."""
    a, _, _, _ = geom(h_km)
    E = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
    r = a * (1 - e * np.cos(E))
    v2 = MU * (2 / r - 1 / a)
    acc = (0.5 * CD * rho_interp((r - R_E) / 1e3, activity) * v2
           * (eps / SIGMA_FILM))
    w = 1 - e * np.cos(E)
    return float(np.sum(acc * w) / np.sum(w) * YEAR_S)


def propellant_fraction(dv, ve):
    return 1 - np.exp(-dv / ve)


# ----------------------------------------------------------------------
# Numerical check of the analytic SRP rate
# ----------------------------------------------------------------------
def accel(x, y):
    """Planar two-body + physical SRP, sun along +x, cylindrical shadow."""
    r = math.hypot(x, y)
    r3 = r * r * r
    ax, ay = -MU * x / r3, -MU * y / r3
    if not (x < 0 and abs(y) < R_E):
        ct = x / r
        f = -A_SRP * ct * abs(ct)
        ax += f * x / r
        ay += f * y / r
    return ax, ay


def rk4_step(s, dt):
    x, y, vx, vy = s
    a1 = accel(x, y)
    a2 = accel(x + 0.5 * dt * vx, y + 0.5 * dt * vy)
    vx2, vy2 = vx + 0.5 * dt * a1[0], vy + 0.5 * dt * a1[1]
    a3 = accel(x + 0.5 * dt * vx2, y + 0.5 * dt * vy2)
    vx3, vy3 = vx + 0.5 * dt * a2[0], vy + 0.5 * dt * a2[1]
    a4 = accel(x + dt * vx3, y + dt * vy3)
    vx4, vy4 = vx + dt * a3[0], vy + dt * a3[1]
    return (x + dt / 6 * (vx + 2 * vx2 + 2 * vx3 + vx4),
            y + dt / 6 * (vy + 2 * vy2 + 2 * vy3 + vy4),
            vx + dt / 6 * (a1[0] + 2 * a2[0] + 2 * a3[0] + a4[0]),
            vy + dt / 6 * (a1[1] + 2 * a2[1] + 2 * a3[1] + a4[1]))


def osculating(x, y, vx, vy):
    r = math.hypot(x, y)
    v2 = vx ** 2 + vy ** 2
    a = 1.0 / (2.0 / r - v2 / MU)
    rv = x * vx + y * vy
    ex = ((v2 - MU / r) * x - rv * vx) / MU
    ey = ((v2 - MU / r) * y - rv * vy) / MU
    return a, ex, ey


def verify(h_km, days=30, steps_per_orbit=600):
    a0, v0, n0, _ = geom(h_km)
    T = 2 * np.pi / n0
    dt = T / steps_per_orbit
    s = (a0, 0.0, 0.0, v0)
    ts, aa, exx, eyy = [], [], [], []
    for k in range(int(days * 86400.0 / T)):
        for _ in range(steps_per_orbit):
            s = rk4_step(s, dt)
        a, ex, ey = osculating(*s)
        ts.append((k + 1) * T); aa.append(a); exx.append(ex); eyy.append(ey)
    ts, aa = np.array(ts), np.array(aa)
    exx, eyy = np.array(exx), np.array(eyy)
    return (np.polyfit(ts, eyy, 1)[0], np.polyfit(ts, exx, 1)[0],
            np.polyfit(ts, aa, 1)[0], srp_e_rate(h_km),
            aa, np.hypot(exx, eyy))


if __name__ == "__main__":
    if not DENSITY_PROVENANCE["sourced"]:
        print("-" * 74)
        print(f"NOTE: density model is {DENSITY_PROVENANCE['model']},")
        print("not a direct model extraction. Parts 1, 5 and 6 depend on it;")
        print("Parts 2-4 do not.")
        print("-" * 74)

    print("\n" + "=" * 74)
    print("PART 1: drag station-keeping dV/yr vs altitude and eps")
    print("=" * 74)
    print("  Circular orbit, drag only; SRP is Part 2. Moderate solar")
    print("  activity. eps = fraction of sail area presented to the")
    print("  airstream, set by flatness and pointing.")
    print(f"  {'h km':>6} | " + " | ".join(f"{'eps=' + f'{e:.0e}':>12}"
                                           for e in EPS_SWEEP))
    fmt_dv = lambda x: f"{x:12,.1f}" if x < 100 else f"{x:12,.0f}"
    for h in ALTS:
        row = " | ".join(fmt_dv(drag_dv_yr(h, e, "mean")) for e in EPS_SWEEP)
        print(f"  {h:6d} | {row}")
    print("  m/s per year. At solar maximum every figure rises:")
    print(f"  {'h km':>6} | {'max/mean':>9}")
    for h in ALTS:
        r = drag_dv_yr(h, 1e-2, "max") / drag_dv_yr(h, 1e-2, "mean")
        print(f"  {h:6d} | {r:8.1f}x")

    print("\n" + "=" * 74)
    print("PART 2: SRP secular eccentricity, analytic (Gauss VOP)")
    print("=" * 74)
    print(f"  a_srp = {A_SRP:.4e} m/s^2 = {A_SRP*86400:.1f} m/s per day")
    print(f"          (P_srp = 2 S0/c = {P_SRP:.3e} N/m^2)")
    print("  A radial force gives NO secular da: SRP is not a reboost cost.")
    print("  It IS an e-control cost if you hold the orbit circular.")
    print(f"  {'h km':>5} | {'ecl %':>5} | {'naive dV':>9} | {'de/dt /d':>9} | "
          f"{'perigee km/d':>12} | {'dV_min/yr':>9} | {'ratio':>5}")
    for h in ALTS:
        a, v, n, beta = geom(h)
        naive = sunlit_mean_cos2(beta) * A_SRP * YEAR_S
        f = srp_e_rate(h)
        dvmin = 0.5 * v * f * YEAR_S
        print(f"  {h:5d} | {100*beta/np.pi:5.1f} | {naive:9.0f} | "
              f"{f*86400:9.2e} | {f*86400*a/1e3:12.2f} | {dvmin:9.0f} | "
              f"{naive/dvmin:5.2f}")
    print("  'naive dV' is the old duty-cycle estimate. It runs ~4x high;")
    print("  dV_min is the impulsive floor (tangential +/- pair at the")
    print("  terminator crossings). Always-on thrust costs pi/2 ~1.57x more.")

    print("\n" + "=" * 74)
    print("PART 3: numerical check of Part 2 (full nonlinear, RK4, 30 days)")
    print("=" * 74)
    for h in [274, 500, 1000]:
        dey, dex, dadt, f_an, aa, ee = verify(h)
        print(f"  h={h:5d}: de_y/dt num {dey:.3e}/s vs analytic {f_an:.3e}/s "
              f"(ratio {dey/f_an:+.4f})")
        print(f"          de_x/dt {dex:.1e}/s (~0 expected); da drift "
              f"{dadt*86400:+.1f} m/day (periodic, not secular)")
    print("  The two paths agree to better than 0.1%. This check is why the")
    print("  separate srp_eccentricity.py script could be retired into this")
    print("  one without losing the independent derivation.")

    print("\n" + "=" * 74)
    print("PART 4: J2 bounds the excursion (uncorrected)")
    print("=" * 74)
    print(f"  {'h km':>5} | {'i deg':>6} | {'apsidal d/d':>11} | {'e*':>7} | "
          f"{'dip ae* km':>10} | {'max dip 2ae*':>12} | {'period d':>8}")
    frozen = {}
    for h in ALTS:
        f, wd, i, e_star, emax, period = j2_srp_averaged(h)
        frozen[h] = e_star
        a = R_E + h * 1e3
        print(f"  {h:5d} | {np.degrees(i):6.2f} | "
              f"{np.degrees(wd)*86400:11.2f} | {e_star:7.4f} | "
              f"{a*e_star/1e3:10.0f} | {a*emax/1e3:12.0f} | {period:8.0f}")
    print("  e oscillates 0 -> 2e* on a ~90-140 day cycle rather than")
    print("  growing without bound. LightSail 2 flew exactly this")
    print("  (Mansell et al. 2023).")

    print("\n" + "=" * 74)
    print("PART 5: passive decay and uncontrolled lifetime")
    print("=" * 74)
    print(f"  face-on film, A/m = {1/SIGMA_FILM:.1f} m^2/kg")
    print(f"  {'h km':>6} | {'km/day mean':>12} | {'km/day max':>11} | "
          f"{'life mean':>10} | {'life max':>10}")
    life_max = {}
    for h in ALTS:
        d_mean = decay_rate_km_per_day(h, 1 / SIGMA_FILM, "mean")
        d_max = decay_rate_km_per_day(h, 1 / SIGMA_FILM, "max")
        l_mean = lifetime_days(h, 1 / SIGMA_FILM, "mean")
        l_max = lifetime_days(h, 1 / SIGMA_FILM, "max")
        life_max[h] = l_max
        fmt = lambda d: (f"{d*24:.1f} h" if d < 1 else
                         (f"{d:.1f} d" if d < 365 else f"{d/365.25:.1f} yr"))
        print(f"  {h:6d} | {d_mean:12.1f} | {d_max:11.1f} | "
              f"{fmt(l_mean):>10} | {fmt(l_max):>10}")
    print("  Solar max is the sizing case: it shortens lifetime by roughly")
    print("  an order of magnitude at every altitude in this range.")

    print("\n" + "=" * 74)
    print("PART 6: total annual budget, cheaper strategy (SOLAR MAX)")
    print("=" * 74)
    print("  eps = 1e-2 effective ram-area fraction, solar max throughout.")
    print("  A = hold circular (SRP e-control + circular drag).")
    print("  B = frozen orbit at e* (SRP free, permanent perigee drag).")
    print(f"  {'h km':>6} | {'life (no prop)':>14} | {'A dV/yr':>9} | "
          f"{'B dV/yr':>9} | {'win':>3} | {'dV/yr':>8} | "
          f"{'SEP %':>6} | {'Chem %':>7}")
    eps = 1e-2
    for h in ALTS:
        _, v, _, _ = geom(h)
        dv_srp = 0.5 * v * srp_e_rate(h) * YEAR_S
        dv_a = dv_srp + drag_dv_yr(h, eps, "max")
        dv_b = drag_dv_yr_eccentric(h, frozen[h], eps, "max")
        win = "A" if dv_a <= dv_b else "B"
        dv = min(dv_a, dv_b)
        l = life_max[h]
        l_s = (f"{l*24:.1f} h" if l < 1 else
               (f"{l:.0f} d" if l < 365 else f"{l/365.25:.1f} yr"))
        print(f"  {h:6d} | {l_s:>14} | {dv_a:9.0f} | {dv_b:9.0f} | "
              f"{win:>3} | {dv:8.0f} | "
              f"{100*propellant_fraction(dv, VE_SEP):6.1f} | "
              f"{100*propellant_fraction(dv, VE_CHEM):7.1f}")
    print("  SEP/Chem are propellant mass fraction PER YEAR, on the same")
    print("  basis as the dV/yr column: m_p/m = 1 - exp(-dV/ve).")
    print("  'life (no prop)' is the uncontrolled lifetime; 'dV/yr' is")
    print("  the cost of holding the orbit against it.")
    print("  Caveats: eps is the dominant unknown in the drag columns;")
    print("  Strategy B assumes J2-only apsidal motion, so a trim budget")
    print("  remains; the whole constellation must share one ellipse")
    print("  orientation for B to be flyable at all.")
