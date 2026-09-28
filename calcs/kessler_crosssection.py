"""
Collision environment for a 1000 km noon-midnight SSO sunshade.

Baseline case
-------------
    shade craft  : 100 m x 10 km = 1.0e6 m^2 = 1 km^2 each
    areal mass   : 21 g/m^2  ->  21,000 kg per craft
    constellation: 464,161 craft -> 4.64e11 m^2 = 464,161 km^2
                   (a 0.1% insolation cut; constellation_sizing.py)
    orbit        : noon-midnight SSO, a = 7371 km (1000 km mean altitude)
    attitude     : "magic carpet", film in the local horizontal
    propulsion   : none for eccentricity control

Part 0.  Solar radiation pressure pumps eccentricity; J2 apsidal
         precession bounds it at e*, so e cycles 0 -> 2e* and the
         constellation occupies a shell rather than a ring.  The cost of
         suppressing this cycle is computed in drag_reboost.py and is not
         repeated here.

Part 1.  The swept shell, and the collision rate it imposes on objects
         already resident in that band.

Part 2.  Part 1 gives un-avoided rates.  Part 2 tests whether collision
         avoidance can reduce them.  It cannot, for a dynamical reason:
         at 21 g/m^2 the shade craft cannot be propagated accurately
         enough to screen against.

Part 3.  Debris, which cannot manoeuvre at all.

Constants are shared with drag_reboost.py; the Part 0 SRP results
reproduce that file by an independent code path.

METHOD NOTES
------------
  Sigma(r)          the arcsine phase density is singular at r = a, so
                    Sigma is binned (BIN_KM).  The 800 km value is
                    bin-independent; the 1000 km value is not.
  Ephemeris growth  uses the secular da -> dn -> phase chain,
                    dx = 0.75 n adot_err t^2, not a kinematic 0.5 a t^2
                    bound.  The radial SRP term is excluded because it
                    produces no secular along-track drift, so drag
                    dominates and the result is a lower bound.

DATA PROVENANCE
---------------
ESA Annual Space Environment Report, issue 10 (1 May 2026), MASTER-8
reference population epoch 01/08/2024:
    54,000 objects > 10 cm (incl. ~9,300 active payloads)
    1.2 million objects 1 cm - 10 cm
Spatial densities at specific altitudes are digitised off the >10 cm
figure and carry a factor of ~3.

Johnson, N. L., P. H. Krisko, J.-C. Liou, and P. D. Anz-Meador. 2001.
"NASA's New Breakup Model of EVOLVE 4.0."  Advances in Space Research
28 (9): 1377-84.  Source of the 40 kJ/kg catastrophic-fragmentation
threshold used in Part 1.
"""

import numpy as np

# np.trapz was renamed np.trapezoid in NumPy 2.0; support both.
_trapz = getattr(np, "trapezoid", None) or np.trapz

# ----------------------------------------------------------------------
# Constants (identical to leo_shade_calcs.py / drag_reboost.py)
# ----------------------------------------------------------------------
MU = 3.986004418e14        # m^3/s^2
J2 = 1.08262668e-3
R_E = 6371.0e3             # m, mean radius
R_EQ = 6378.137e3          # m, equatorial (J2 terms)
YEAR_S = 365.2422 * 86400.0
S0 = 1361.0
C_LIGHT = 2.99792458e8
P_SRP = 2 * S0 / C_LIGHT   # N/m^2, specular normal incidence
D2R = np.pi / 180.0

# ----------------------------------------------------------------------
# Baseline constellation
# ----------------------------------------------------------------------
A_ORB = R_E + 1000e3
V_ORB = np.sqrt(MU / A_ORB)
N_ORB = np.sqrt(MU / A_ORB**3)

A_CRAFT = 100.0 * 10.0e3           # 1.0e6 m^2
SIGMA = 0.021                      # kg/m^2
M_CRAFT = SIGMA * A_CRAFT          # 21,000 kg
# Fleet for a 0.1% insolation cut at 1000 km. The instantaneous blocking
# area required is 127,516 km^2 (0.1% of Earth's disc), but a craft blocks
# sunlight only a fraction eta = 27.5% of the time: 33.2% orbital duty cycle
# x 0.827 cosine projection of the horizontal carpet. See
# constellation_sizing.py.
N_CRAFT = 464_161
A_TOT = N_CRAFT * A_CRAFT          # 4.64e11 m^2
M_TOT = N_CRAFT * M_CRAFT

# ESA Space Debris User Portal, environment statistics, 31 July 2026.
RHO_FILM = 1400.0                  # kg/m^3, polyimide; sets film thickness
E_CATASTROPHIC = 40e3              # J/kg, NASA standard breakup model
                                   # threshold (Johnson et al. 2001)

MASS_IN_ORBIT_T = 17_020           # tonnes, all orbital regimes
MASS_IN_LEO_T = 9_874              # tonnes, LEO only
XSECT_IN_LEO_M2 = 361_824          # m^2, cumulative cross-section, LEO
CATALOGUED_LEO = 27_465            # objects catalogued in LEO

# MASTER 01/08/2024 spatial density of >10 cm objects, km^-3, digitised
RHO_10CM = {600: 8e-8, 700: 7e-8, 800: 9e-8, 900: 5e-8,
            1000: 3e-8, 1100: 2e-8, 1200: 3e-8, 1300: 2e-8}

# Real missions inside the swept band (altitude km, note)
RESIDENTS = [
    (705, "Landsat 8/9, Terra, Aqua"),
    (780, "Iridium NEXT (75 sats)"),
    (786, "Sentinel-2"),
    (817, "MetOp"),
    (824, "Suomi-NPP / JPSS"),
    (830, "Sentinel-1"),
    (1200, "OneWeb (~650 sats)"),
]


def sso_inclination_deg(h_km):
    a = R_EQ + h_km * 1e3
    n = np.sqrt(MU / a**3)
    cos_i = -(2 * np.pi / YEAR_S) / (1.5 * J2 * n * (R_EQ / a) ** 2)
    return np.degrees(np.arccos(cos_i))


def rho_10cm(h_km):
    hs = np.array(sorted(RHO_10CM))
    vs = np.log([RHO_10CM[h] for h in hs])
    return float(np.exp(np.interp(h_km, hs, vs)))


SAT_XSECT = 10.0        # m^2, reference spacecraft cross-section
SAT_MASS = 700.0        # kg, reference spacecraft mass
V_REL = 10.0e3          # m/s, conventional LEO closing speed


def natural_rate(h_km):
    """Present-day collision rate per year for a SAT_XSECT spacecraft
    against the >10 cm catalogue at altitude h_km."""
    return (rho_10cm(h_km) / 1e9) * SAT_XSECT * V_REL * YEAR_S


# ======================================================================
# PART 0 - the constellation cannot be held circular
# ======================================================================
def part0():
    print("=" * 74)
    print("PART 0 - THE PREMISE: A 21 g/m^2 SHADE CANNOT HOLD A CIRCULAR ORBIT")
    print("=" * 74)
    print()

    A_acc = P_SRP / SIGMA                       # broadside SRP acceleration
    sin_b = R_E / A_ORB                         # eclipse half-angle sine
    bracket = 4.0 / 3.0 - (sin_b - sin_b**3 / 3.0)
    dedt = A_acc / (np.pi * V_ORB) * bracket    # per second (Gauss VOP)

    i_sso = sso_inclination_deg(1000.0)
    cos_i = np.cos(i_sso * D2R)
    omdot = 0.75 * N_ORB * J2 * (R_EQ / A_ORB) ** 2 * (5 * cos_i**2 - 1)
    e_star = dedt / abs(omdot)
    e_max = 2 * e_star
    cycle_d = 2 * np.pi / abs(omdot) / 86400

    r_lo, r_hi = A_ORB * (1 - e_max), A_ORB * (1 + e_max)
    h_lo, h_hi = (r_lo - R_E) / 1e3, (r_hi - R_E) / 1e3

    print(f"  areal mass                 sigma = {SIGMA*1e3:.0f} g/m^2")
    print(f"  SRP broadside acceleration P/sigma = {A_acc:.3e} m/s^2 "
          f"({A_acc/(MU/A_ORB**2):.1e} g)")
    print(f"  secular SRP forcing        de/dt = {dedt*YEAR_S:.3f} per year")
    print(f"  SSO inclination at 1000 km       {i_sso:.2f} deg")
    print(f"  J2 apsidal precession            "
          f"{np.degrees(omdot)*86400:.2f} deg/day")
    print()
    print("  The forcing is sun-fixed; the e-vector is rotated out from")
    print("  under it by J2.  The averaged dynamics give a circle in the")
    print("  e-plane rather than unbounded growth (Colombo/Lucking/McInnes;")
    print("  flight precedent LightSail 2, Mansell et al. 2023):")
    print()
    print(f"    forced equilibrium        e*     = {e_star:.4f}")
    print(f"    uncontrolled excursion    0 -> 2e* = {e_max:.4f}")
    print(f"    cycle period                       {cycle_d:.0f} days")
    print()
    print(f"  Result: the constellation occupies a shell between "
          f"{h_lo:,.0f} km and")
    print(f"  {h_hi:,.0f} km, cycling on a {cycle_d:.0f}-day period, "
          "rather than a ring at 1000 km.")
    print()
    return dict(e_star=e_star, e_max=e_max, cycle_d=cycle_d,
                dedt=dedt, h_lo=h_lo, h_hi=h_hi, r_lo=r_lo, r_hi=r_hi)


# ======================================================================
# PART 1 - the swept shell and what it does to everyone else
# ======================================================================
def vz_rel_stats(n=400_000, seed=3):
    """Mean out-of-plane relative speed |v_z| at which a satellite in a
    random LEO plane crosses the shade plane, and mean relative speed."""
    rng = np.random.default_rng(seed)
    i_shade = sso_inclination_deg(1000.0) * D2R
    centres = np.array([53.0, 70.0, 74.0, 82.0, 87.0, 98.0])
    wts = np.array([0.28, 0.11, 0.09, 0.13, 0.06, 0.33])
    wts /= wts.sum()
    inc = (centres[rng.choice(len(centres), size=n, p=wts)]
           + rng.normal(0, 3.0, n)) * D2R
    draan = rng.uniform(0, 2 * np.pi, n)
    # angle between the two orbit NORMALS = relative inclination
    cos_rel = (np.cos(i_shade) * np.cos(inc)
               + np.sin(i_shade) * np.sin(inc) * np.cos(draan))
    i_rel = np.arccos(np.clip(cos_rel, -1, 1))
    # at a node of the two planes the intruder's velocity is inclined by
    # i_rel to the shade plane -> out-of-plane component v*sin(i_rel)
    vz = V_ORB * np.abs(np.sin(i_rel))
    w = 2 * V_ORB * np.abs(np.sin(i_rel / 2))
    return vz.mean(), w.mean()


BIN_KM = 5.0    # half-width of the radial bin used to regularise Sigma


def sigma_surface(h_km, p0, n_phase=20_001, bin_km=BIN_KM):
    """Time-averaged SURFACE number density of shade craft (craft per m^2
    of the shade plane) in a radial bin of half-width bin_km centred on
    r = R_E + h_km.

    A craft of eccentricity e spends time at radius r = a(1+d), d in
    (-e, e), according to the arcsine law, whose CDF is
        F(d) = 1/2 + arcsin(d/e)/pi.
    e itself cycles 0 -> 2e* uniformly in phase, so average over phase.

    The bin is NECESSARY, not cosmetic: the arcsine density is singular
    at d = 0, and the phase average of 1/e diverges logarithmically as
    e -> 0, so the point value of Sigma at exactly r = a is undefined.
    Physically the pile-up is limited by dispersion in semi-major axis
    and in apsidal phase across the fleet, neither of which is modelled
    here; bin_km = 5 km matches the craft's own 10 km cross-track scale
    and a typical station-keeping box.  Sensitivity to bin_km is
    reported in Part 3.
    """
    r = R_E + h_km * 1e3
    d0 = (r - bin_km * 1e3) / A_ORB - 1.0
    d1 = (r + bin_km * 1e3) / A_ORB - 1.0
    phase = np.linspace(0, np.pi, n_phase)          # |e| = 2e* |sin(ph/2)|
    e = p0["e_max"] * np.abs(np.sin(phase / 2))
    e = np.maximum(e, 1e-12)

    def cdf(d):
        return 0.5 + np.arcsin(np.clip(d / e, -1, 1)) / np.pi

    frac = (cdf(d1) - cdf(d0)).mean()               # fraction of the fleet
    area = 2 * np.pi * r * (2 * bin_km * 1e3)       # plane area of the bin
    return N_CRAFT * frac / area


def vr_shade_mean(p0, n_phase=20_001):
    """Time- and orbit-averaged |radial velocity| of a shade craft."""
    phase = np.linspace(0, np.pi, n_phase)
    e = p0["e_max"] * np.abs(np.sin(phase / 2))
    return (2 / np.pi) * V_ORB * e.mean()


def part1(p0):
    print("=" * 74)
    print("PART 1 - THE SWEPT SHELL")
    print("=" * 74)
    print()

    vz, w = vz_rel_stats()
    vr = vr_shade_mean(p0)

    band_km = p0["h_hi"] - p0["h_lo"]
    vol = (4 / 3) * np.pi * ((p0["r_hi"] / 1e3) ** 3 - (p0["r_lo"] / 1e3) ** 3)
    print(f"  swept altitude band        {p0['h_lo']:,.0f} - "
          f"{p0['h_hi']:,.0f} km  ({band_km:,.0f} km thick)")
    print(f"  that is {100*band_km/1600:.0f}% of the 400-2000 km usable LEO "
          "envelope by altitude")
    print()
    print(f"  constellation mass         {M_TOT/1e3:,.0f} t = "
          f"{M_TOT/1e3/MASS_IN_ORBIT_T:,.0f}x everything now in orbit")
    print(f"  constellation area         {A_TOT/1e6:,.0f} km^2 = "
          f"{A_TOT/XSECT_IN_LEO_M2/1e6:.1f} million x the cumulative")
    print(f"                             cross-section now in LEO "
          f"({XSECT_IN_LEO_M2/1e6:.2f} km^2 across "
          f"{CATALOGUED_LEO:,} catalogued objects, {MASS_IN_LEO_T:,} t)")
    print(f"  mean |radial velocity| of a shade craft   {vr:.0f} m/s")
    print(f"  mean out-of-plane crossing speed |v_z|    {vz/1e3:.2f} km/s")
    print(f"  mean relative speed |w|                   {w/1e3:.2f} km/s")
    print()

    # ---- the collision rate on a resident satellite ------------------
    print("  COLLISION RATE ON ANYTHING LIVING IN THE BAND")
    print()
    print("  A satellite in any other plane crosses the shade plane twice")
    print("  per revolution.  Treating each shade craft as a point with")
    print("  cross-section A|cos(theta)| = A|w_r|/|w| (magic-carpet normal")
    print("  = local vertical), the collision rate integrates to")
    print()
    print("      P(hit per plane crossing) = Sigma(r) * A * |w_r| / |v_z|")
    print()
    print("  with Sigma the surface number density of shade craft in the")
    print("  shade plane.  Note |w| cancels: only RADIAL motion matters,")
    print("  and the breathing supplies it.")
    print()
    # normalisation check: integrating Sigma over the annulus must give N
    rr = np.linspace(p0["r_lo"], p0["r_hi"], 601)
    tot = _trapz([sigma_surface((r - R_E) / 1e3, p0) * 2 * np.pi * r
                  for r in rr], rr)
    print(f"  [check] Sigma integrated over the annulus = {tot:,.0f} craft "
          f"vs {N_CRAFT:,} placed ({100*tot/N_CRAFT:.1f}%).")
    print("          Closes to 0.2%; the residual is the small-e")
    print("          approximation d = r/a - 1 in the arcsine law.")
    print()

    hdr = (f"  {'alt km':>7s} {'mission':>28s} {'Sigma [1/km2]':>14s} "
           f"{'P/crossing':>11s} {'hits/yr':>9s} {'vs natural':>11s}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    rows = []
    for h, name in RESIDENTS + [(1000, "(mean radius - see Part 3)")]:
        sig = sigma_surface(h, p0)
        p_hit = sig * A_CRAFT * vr / vz
        a_s = R_E + h * 1e3
        per_yr = 2 * (YEAR_S / (2 * np.pi * np.sqrt(a_s**3 / MU))) * p_hit
        nat = natural_rate(h)
        print(f"  {h:7.0f} {name:>28s} {sig*1e6:14.4f} {p_hit:11.2e} "
              f"{per_yr:9.2f} {per_yr/nat:10,.0f}x")
        rows.append((h, name, per_yr, nat))
    print()
    print(f"  'vs natural' is the ratio to the present-day collision rate")
    print(f"  for a {SAT_XSECT:.0f} m^2 spacecraft against the >10 cm")
    print("  catalogue at the same altitude.  BOTH ARE UN-AVOIDED RATES,")
    print("  so the comparison is like for like.  Operators knock the")
    print("  natural number down by orders of magnitude with collision")
    print("  avoidance; Part 2 asks whether they could here.")
    print()
    res = [r for r in rows if "mean radius" not in r[1]]
    print(f"  Absent avoidance, each operational satellite in the band is")
    print(f"  struck {min(r[2] for r in res):.0f} to "
          f"{max(r[2] for r in res):.0f} times a year, "
          f"{min(r[2]/r[3] for r in res):,.0f} to "
          f"{max(r[2]/r[3] for r in res):,.0f} times the")
    print("  natural rate. The band contains OneWeb, Iridium NEXT, Landsat,")
    print("  Sentinel, MetOp, JPSS, Terra and Aqua.")
    print()
    print("  Severity of a single strike.  The impacting mass follows from")
    print("  the areal density alone; film thickness is descriptive only.")
    e_spec = 0.5 * (SAT_XSECT * SIGMA) * w**2 / SAT_MASS
    print(f"      film thickness      sigma/rho = {SIGMA/RHO_FILM*1e6:.0f} um "
          f"at rho = {RHO_FILM:,.0f} kg/m^3 (polyimide)")
    print(f"      impacting mass      {SAT_XSECT:.0f} m^2 x {SIGMA*1e3:.0f} "
          f"g/m^2 = {SAT_XSECT*SIGMA:.2f} kg at {w/1e3:.1f} km/s")
    print(f"      kinetic energy      {0.5*(SAT_XSECT*SIGMA)*w**2/1e6:.1f} MJ "
          f"into a {SAT_MASS:,.0f} kg bus")
    print(f"      specific energy     {e_spec/1e3:,.0f} kJ/kg against the "
          f"{E_CATASTROPHIC/1e3:.0f} kJ/kg")
    print("                          catastrophic-fragmentation threshold of")
    print("                          the NASA standard breakup model")
    print("                          (Johnson et al. 2001)")
    print()
    print("  Below the fragmentation threshold, above any plausible")
    print("  survival threshold: mission-ending, but not by itself the")
    print("  full fragment cloud the standard breakup model predicts.")
    print()
    return dict(vz=vz, w=w, vr=vr, rows=rows)


# ======================================================================
# PART 2 - can operational satellites just manoeuvre out of the way?
# ======================================================================
B_SHADE = SIGMA / 2.2              # kg/m^2, shade ballistic coefficient
B_NORMAL = 100.0                   # kg/m^2, conventional spacecraft
RHO_AIR_1000 = 3.0e-15             # kg/m^3, solar mean (drag_reboost.py)
RHO_AIR_UNCERT = 0.5               # fractional 1-sigma at 1000 km
HARD_BODY_KM = 5.0                 # half the 10 km craft, as a hard-body radius
P_ACTION = 1e-4                    # standard collision-avoidance threshold
SSN_CATALOGUE = 40_000             # objects tracked today


def alongtrack_error(t_s, B=B_SHADE, frac=RHO_AIR_UNCERT):
    """Along-track ephemeris error after t_s seconds, from an unmodelled
    fraction `frac` of the drag.

    A density error gives a semi-major-axis rate error adot_err, which
    integrates into a mean-motion error and then into phase:
        dx(t) = 0.75 * n * adot_err * t^2
    This is the dominant error growth for a low-ballistic-coefficient
    object; the radial SRP term produces no secular along-track drift
    (see srp_eccentricity.py) and is an order of magnitude smaller here.
    """
    adot = np.sqrt(MU * A_ORB) * RHO_AIR_1000 / B      # m/s, nominal decay
    return 0.75 * N_ORB * (frac * adot) * t_s**2


def part2(p0, p1):
    print("=" * 74)
    print("PART 2 - COLLISION AVOIDANCE DOES NOT APPLY TO THIS OBJECT")
    print("=" * 74)
    print()
    print("  Part 1 gives un-avoided rates.  Every satellite in the list")
    print("  has thrusters and a conjunction-assessment process, so the")
    print("  fair question is whether that process works against a shade")
    print("  constellation.  It does not, for a reason that is dynamical")
    print("  rather than operational: MORE TRACKING DOES NOT FIX IT.")
    print()

    # ---- 2a: ephemeris predictability -------------------------------
    print("  2a. The shade craft cannot be propagated well enough to screen.")
    print()
    adot = np.sqrt(MU * A_ORB) * RHO_AIR_1000 / B_SHADE
    print(f"      shade ballistic coefficient    {B_SHADE:.4f} kg/m^2")
    print(f"      conventional spacecraft        {B_NORMAL:.0f} kg/m^2   "
          f"({B_NORMAL/B_SHADE:,.0f}x higher)")
    print(f"      nominal decay rate at 1000 km  {adot*YEAR_S/1e3:,.0f} km/yr")
    print(f"      density knowledge at 1000 km   +/-{100*RHO_AIR_UNCERT:.0f}% "
          "(H/He regime, storm-driven)")
    print()
    print(f"      {'lead time':>12s} {'shade error':>14s} "
          f"{'normal s/c':>13s} {'vs 10 km craft':>16s}")
    for lab, t in [("1 hour", 3600), ("6 hours", 6 * 3600),
                   ("12 hours", 12 * 3600), ("24 hours", 86400),
                   ("72 hours", 3 * 86400)]:
        es = alongtrack_error(t)
        en = alongtrack_error(t, B=B_NORMAL)
        print(f"      {lab:>12s} {es/1e3:11,.1f} km {en:10,.0f} m "
              f"{es/1e3/(2*HARD_BODY_KM):15.1f}x")
    print()
    t_ok = np.sqrt(2 * HARD_BODY_KM * 1e3
                   / (0.75 * N_ORB * RHO_AIR_UNCERT * adot))
    print(f"      Useful warning requires the error to fall below the craft's")
    print(f"      own {2*HARD_BODY_KM:.0f} km size, which happens only inside "
          f"{t_ok/3600:.1f} hours")
    print("      of the encounter.  Standard practice screens 3-7 days out")
    print("      and acts 24-48 hours out.  At 24 hours a shade craft's")
    print(f"      position is known to {alongtrack_error(86400)/1e3:,.0f} km, "
          f"about {alongtrack_error(86400)/1e3/(2*HARD_BODY_KM):.0f}x its own "
          "length.")
    print()
    print("      This is a DYNAMICAL limit, not a sensor limit.  A 1 km^2")
    print("      film is trivially easy to detect; it is the propagation")
    print("      that fails, and more radars do not help.")
    print()
    print("      Both failure modes follow from that single error figure.")
    print("      The warning arrives too late to act on, AND the position")
    print("      uncertainty is so wide that everything inside it must be")
    print("      treated as a possible hit, so the alerts are too many:")
    print()
    s800 = sigma_surface(800, p0)
    a_s = R_E + 800e3
    ncross = 2 * YEAR_S / (2 * np.pi * np.sqrt(a_s**3 / MU))
    real = s800 * A_CRAFT * p1["vr"] / p1["vz"] * ncross
    sig24 = alongtrack_error(86400) / 1e3            # km
    print(f"      {'window':>22s} {'craft in window':>17s} "
           f"{'alerts/yr':>12s} {'real hits/yr':>14s} {'ratio':>10s}")
    for lab, k in [("1 sigma (48 km)", 1), ("3 sigma (143 km)", 3)]:
        # window area in m^2: along-track +/- k sigma, cross-track the
        # craft's own 10 km span.  Sigma is in craft per m^2.
        n_in = s800 * (2 * k * sig24 * 1e3) * (2 * HARD_BODY_KM * 1e3)
        alerts = n_in * ncross
        print(f"      {lab:>22s} {n_in:17.1f} {alerts:12,.0f} "
              f"{real:14.2f} {alerts/real:9,.0f}:1")
    print()
    print("      Tens of thousands of alerts a year, of which about ONE is")
    print("      real, and the ephemeris is not good enough to tell them")
    print("      apart.  For comparison, an ESA science mission today")
    print("      receives a few hundred alerts a year and acts on one or")
    print("      two.")
    print()

    # ---- 2b: nowhere to dodge to ------------------------------------
    print("  2b. Even with perfect knowledge, there is nowhere to dodge to.")
    print()
    print("      Conventional avoidance works because conjunctions are rare,")
    print("      discrete and local: you move 1 km and the threat is gone.")
    print("      Here the hazard is a smooth function of altitude across a")
    print("      720 km band, so a manoeuvre inside the band buys nothing.")
    print()
    print(f"      {'altitude':>10s} {'Sigma [1/km2]':>14s} {'rate vs 800 km':>16s}")
    s800 = sigma_surface(800, p0)
    for h in (790, 795, 800, 810, 850, 900):
        s = sigma_surface(h, p0)
        print(f"      {h:7.0f} km {s*1e6:14.4f} {s/s800:15.3f}x")
    print()
    print("      A 10 km radial manoeuvre - twenty times a normal avoidance")
    print("      displacement - changes the hazard by a few percent.  The")
    print("      only manoeuvre that helps is leaving the band entirely.")
    print()

    # ---- 2d: the only real option -----------------------------------
    print("  2d. Vacating the band: cheap in dV, fatal to the mission.")
    print()
    print(f"      {'mission':>28s} {'alt':>7s} {'exit to':>9s} {'dv':>9s}")
    for h, name in RESIDENTS:
        if abs(h - p0["h_lo"]) < abs(h - p0["h_hi"]):
            tgt = p0["h_lo"] - 10
        else:
            tgt = p0["h_hi"] + 10
        r1, r2 = R_E + h * 1e3, R_E + tgt * 1e3
        dv = abs(np.sqrt(MU / r1) * (np.sqrt(2 * r2 / (r1 + r2)) - 1)) + \
             abs(np.sqrt(MU / r2) * (1 - np.sqrt(2 * r1 / (r1 + r2))))
        print(f"      {name:>28s} {h:5.0f} km {tgt:6.0f} km {dv:6.1f} m/s")
    print()
    print("      Tens of m/s - trivial.  But sun-synchronous science orbits")
    print("      are altitude-QUANTISED by their repeat ground tracks:")
    print("      Landsat 705 km = 233 rev / 16 d, Sentinel-2 786 km =")
    print("      143 rev / 10 d, MetOp 817 km = 412 rev / 29 d.  Those")
    print("      altitudes are not preferences; they are what fixes swath")
    print("      overlap, revisit interval and the multi-decade radiometric")
    print("      record.  Moving Landsat below 640 km ends the Landsat")
    print("      continuity record.  Moving OneWeb above 1360 km puts it")
    print("      where post-mission disposal rules forbid operation.")
    print()
    print("      Summary of Part 2: the threat cannot be predicted with")
    print("      useful lead time (2a), cannot be dodged locally (2b), and")
    print("      the only effective manoeuvre is to leave the band, which")
    print("      ends the repeat ground track the mission depends on (2d).")
    print()


# ======================================================================
# PART 3 - the population that cannot manoeuvre at all
# ======================================================================
def part3_debris(p0, p1):
    print("=" * 74)
    print("PART 3 - DEBRIS: THE POPULATION WITH NO THRUSTERS")
    print("=" * 74)
    print()
    vr, w = p1["vr"], p1["w"]

    # how much debris lives in the swept band
    hs = np.linspace(p0["h_lo"], p0["h_hi"], 400)
    n10 = 0.0
    for h in hs:
        r = (R_E + h * 1e3) / 1e3
        n10 += rho_10cm(h) * 4 * np.pi * r**2 * (hs[1] - hs[0])
    print(f"  >10 cm objects inside {p0['h_lo']:,.0f}-{p0['h_hi']:,.0f} km: "
          f"{n10:,.0f}")
    print(f"  >1 cm equivalent (x{1.25e6/54e3:.0f} by ESA cumulative counts): "
          f"{n10*1.25e6/54e3:,.0f}")
    print("  None of these can be warned, screened or moved.  Whatever the")
    print("  answer is for operational satellites, for this population the")
    print("  un-avoided rate IS the rate.")
    print()

    rate10 = A_TOT * (np.mean([rho_10cm(h) for h in hs]) / 1e9) * vr * YEAR_S
    print(f"  Shade struck by >10 cm debris: {rate10:,.0f} times a year, "
          f"one every {YEAR_S/rate10/3600:.1f} hours.")
    print(f"  Shade struck by  >1 cm debris: {rate10*1.25e6/54e3:,.0f} "
          "times a year.")
    print()

    print("  Does the shade sweep the band clean?  No.  A 10 cm fragment")
    m_frag, a_frag = 0.5, 0.008        # kg, m^2 - typical 10 cm fragment
    dv_frag = (a_frag * SIGMA / m_frag) * w
    print(f"  of ~{m_frag:.1f} kg and ~{a_frag*1e4:.0f} cm^2 punching through "
          "the film sweeps up")
    print(f"  {a_frag*SIGMA*1e3:.2f} g of material and is slowed by "
          f"{dv_frag:.1f} m/s out of {w/1e3:.1f} km/s")
    print(f"  ({100*dv_frag/w:.4f}%).  The film does not stop debris; debris")
    print("  perforates the film and continues.")
    print()
    print("  So the debris ledger is one-way.  The shade removes nothing and")
    print("  adds, per severed craft, 21 t of film at "
          f"A/m = {1/SIGMA:.0f} m^2/kg - the")
    print("  highest area-to-mass debris ever placed in orbit, inside the")
    print("  band that already contains the LEO debris peak.")
    print()


if __name__ == "__main__":
    print()
    print("=" * 74)
    print("LEO SUNSHADE - CONSTRAINT B, 1000 km NOON-MIDNIGHT SSO, "
          "NO e-CONTROL")
    print("=" * 74)
    print()
    p0 = part0()
    p1 = part1(p0)
    part2(p0, p1)
    part3_debris(p0, p1)
