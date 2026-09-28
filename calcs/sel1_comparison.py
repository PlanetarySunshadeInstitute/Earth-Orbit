"""
Transport cost of a shade: 1000 km LEO vs SEL1.

Question
--------
Per Starship launch, how much shade film ends up deployed at a 1000 km
sun-synchronous orbit, versus at SEL1?

The naive comparison - drop both payloads in the same parking orbit and
compare orbit-raising delta-v - is wrong, because the two destinations do
not want the same drop-off, or even the same launch site:

  * The 1000 km shade wants a retrograde sun-synchronous plane (i = 99.5
    deg), reachable only from a high-latitude range flying south. Starship
    can launch straight into that inclination, so there is no plane change
    to pay for on orbit - but the launch itself forfeits Earth's rotation
    instead of using it, and that shows up as reduced payload.

  * An SEL1 departure wants a LOW-inclination prograde orbit. The
    departure asymptote must point along the Earth-Sun line, which lies
    within +/-23.4 deg of the equator (the Sun's declination range). A
    28.5 deg parking orbit contains every asymptote you will ever need, so
    the injection costs no plane change at all - you just pick the launch
    time. This is why every real L1 mission flies this way: ACE launched
    to 185 km x 28.7 deg, DSCOVR to 185 km x 37 deg. Departing for SEL1
    from a 99.5 deg polar orbit would demand a ~70 deg plane change and is
    never done.

The accounting therefore has three terms, not one:

    shade delivered  =  launch payload factor        (inclination cost,
                                                      paid at the pad)
                      x surviving mass fraction      (in-space delta-v,
                                                      paid by the shade)
                      x 1                            (both cases deploy
                                                      the same film)

The first term penalizes LEO. The second penalizes SEL1. This script asks
which is bigger.

Assumptions
-----------
1. Both cases drop off at 400 km circular. Results are nearly insensitive
    to this: a 200 km parking orbit changes the SEL1 injection burn by
    under 5 m/s, because at these energies the Oberth gain and the extra
    climb almost exactly cancel.
2. Leg A: launch direct to i = 99.48 deg (the SSO inclination AT 1000 km,
    not at 400 km), then a pure coplanar raise. The 400 km parking orbit
    is therefore not itself sun-synchronous and its node drifts during the
    transfer; see LTAN drift below. It is correctable by launch timing, so
    it is not charged as delta-v.
3. Leg B: launch to i = 28.5 deg from an eastern range, then a single
    trans-L1 injection. No plane change (see above), plus a small
    midcourse and halo-insertion allowance.
4. Launch payload factor derived from the Earth-rotation assist forfeited
    at the pad, propagated through the upper stage's rocket equation.
    Prograde due-east from 28.5 deg is the reference, factor = 1.00.
5. Two propulsion modes, since Isp is the biggest single lever:
    chemical (Isp 450 s, impulsive) and solar electric (Isp 3000 s,
    low-thrust). The SEP case is charged the full spiral-to-escape cost,
    ~v_circ, because a slow spiral forfeits the Oberth effect - this is
    the one term that materially hurts SEL1, and it is charged in full.
6. Tankage, tug dry mass, and structure are NOT charged. That omission
    favors the higher-delta-v case, i.e. it favors SEL1. Noted, not
    exploited.

Shading effectiveness (Part 2)
------------------------------
Freight alone does not settle the comparison, because a square metre of
film does different amounts of work at the two destinations. Part 2 closes
that gap, then combines the two into a per-launch figure.

The governing quantity is the shadowing efficiency eps: the fraction of a
shade's own area that is actually removed from the flux Earth intercepts.
Angel (2006) names it and defines it by A = f pi R_E^2 / eps, the area
needed to cut insolation by a fraction f. At SEL1 it is well below 1 for a
purely geometric reason. The Sun is 0.53 deg wide, so a shade 1.5-2.4
million km sunward casts a penumbra 14,000-23,000 km across, wider than
Earth's 12,742 km, and the spill past the limb is lost. The loss grows as
the square of the standoff distance.

We take eps from McInnes, by two independent routes that agree:

  McInnes (2010), and the earlier minimum-mass paper, locate the optimum
  station at 2.36e6 km - "independent of the shield properties,
  representing the true optimum location for an occulting disc" - and give
  6.57e6 km^2 as the area required there for a 1.7% insolation cut.
  That is eps = 0.33.

  Sanchez and McInnes (2015) redo the sizing against a globally resolved
  energy balance model rather than a solid-angle estimate, and get a
  1,434 km radius disc at 2.44e6 km. That is eps = 0.34.

Part 2 does not take either on faith: it reproduces both published areas
from the solid-angle relation and reports the residual.

Basis of the comparison
-----------------------
Delivered MASS, the report's metric throughout (Section 1.2). Both
destinations are charged the same 21 g/m^2 reference film, so mass and
area are interchangeable and the areal density cancels from the ratio.
This is conservative toward SEL1: the published SEL1 designs use lighter
material than 21 g/m^2, because SEL1 imposes no requirement to be
reflective while the LEO carpet must be reflective or it re-radiates
longwave back down (Section 1.5). Part 2 prints that sensitivity as a
sensitivity and does not claim it.

Not in scope
------------
Lifetime, station-keeping, replacement cadence, debris externalities.
Those live in their own sections and are not folded in here.
"""

import math

# ---------------------------------------------------------------- constants
MU = 398600.4418         # km^3/s^2
R_E = 6378.137           # km
J2 = 1.08263e-3
G0 = 9.80665e-3          # km/s^2
YEAR = 365.2422 * 86400.0
OMEGA_DOT_SS = 2.0 * math.pi / YEAR      # rad/s, sun-synchronous precession
V_EQ_SURFACE = 0.4651                    # km/s, Earth surface speed at equator
SUN_DECL_MAX = 23.44                     # deg

# ------------------------------------------------------------- ASSUMPTIONS
ALT_PARK = 400.0         # km, Starship drop-off, both legs
ALT_TARGET = 1000.0      # km, noon-midnight SSO
D_SEL1 = 1.5e6           # km
DV_L1_ARRIVAL = 0.050    # km/s, midcourse + halo insertion

LAT_POLAR = 34.7         # deg, Vandenberg - the SSO leg
LAT_EAST = 28.5          # deg, Cape Canaveral - the SEL1 leg
INC_SEL1_PARK = 28.5     # deg, matches the launch latitude (due east)

ISP_UPPER = 380.0        # s, upper stage, for the payload-factor derivation
M_SHIP_DRY = 100.0       # t, Starship ship dry mass
M_PAYLOAD_REF = 100.0    # t, reference due-east payload

TRANSFER_DAYS = 30.0     # days, assumed orbit-raising duration (LTAN note)

# ------------------------------------------- Part 2: shading effectiveness
R_E_MEAN = 6371.0        # km, volumetric mean radius (shadow geometry)
R_SUN = 6.957e5          # km
AU = 1.495979e8          # km

# LEO steelman, from constellation_sizing.py / noon_midnight_calcs.py:
# year-averaged blocked-area fraction at a 1,000 km noon-midnight SSO.
LEO_DUTY = 0.332         # shade cast this fraction of each orbit
LEO_COSPROJ = 0.827      # cosine projection of the horizontal carpet
ETA_LEO = LEO_DUTY * LEO_COSPROJ

# Published SEL1 sizings. (label, standoff km, insolation cut, area km^2)
SEL1_SIZINGS = [
    ("McInnes 2010, min-mass optimum", 2.36e6, 0.017, 6.57e6),
    ("Sanchez & McInnes 2015 (GREB)", 2.44e6, 0.017, math.pi * 1434.0 ** 2),
]

SIGMA_FILM = 21.0        # g/m^2, the report's reference film, both legs
# Areal densities the published SEL1 designs actually assume, for the
# sensitivity only (McInnes 2010, Table 2).
SEL1_ALT_SIGMA = [("McInnes reflecting disc", 40.2),
                  ("McInnes absorbing disc", 7.9),
                  ("Angel refracting screen", 4.2)]

PROPULSION = {
    "chemical (Isp 450 s)": dict(isp=450.0, lowthrust=False),
    "solar electric (Isp 3000 s)": dict(isp=3000.0, lowthrust=True),
}


# ---------------------------------------------------------------- geometry
def sso_inclination(a_km):
    """Inclination (deg) of a circular sun-synchronous orbit of radius a."""
    cos_i = -(2.0 * a_km ** 3.5 * OMEGA_DOT_SS) / (
        3.0 * R_E ** 2 * J2 * math.sqrt(MU))
    return math.degrees(math.acos(cos_i))


def node_drift_deg_per_day(a_km, inc_deg):
    """J2 nodal precession of a circular orbit, deg/day."""
    n = math.sqrt(MU / a_km ** 3)
    rate = -1.5 * n * J2 * (R_E / a_km) ** 2 * math.cos(math.radians(inc_deg))
    return math.degrees(rate) * 86400.0


def v_circ(a_km):
    return math.sqrt(MU / a_km)


# ------------------------------------------ Part 2: shadowing efficiency
def disk_area_solid_angle(d_standoff_km, f):
    """Area a shade must have at standoff d to cut insolation by f.

    Solid-angle match: the shade must cover the fraction f of the solar
    disc as seen from Earth, so (R_s/d)^2 / (R_sun/AU)^2 = f. Both McInnes
    (2010) Eq 5 and Sanchez & McInnes (2015) Eq 1 are this relation."""
    r_shade = R_SUN * (d_standoff_km / AU) * math.sqrt(f)
    return math.pi * r_shade ** 2


def eps_from_area(f, area_km2):
    """Angel (2006): A = f pi R_E^2 / eps, so eps = f pi R_E^2 / A.

    eps is the shade area a perfect occulter would need, over the area
    actually needed. It is independent of f, since both scale linearly."""
    return (math.pi * R_E_MEAN ** 2 * f) / area_km2


# ------------------------------------------------- launch: what the pad costs
def rotation_assist(lat_deg, inc_deg):
    """Eastward component of the launch site's inertial velocity that the
    vehicle actually gets to keep, km/s. Negative for retrograde orbits:
    the vehicle must cancel the site motion and then go the other way."""
    v_site = V_EQ_SURFACE * math.cos(math.radians(lat_deg))
    sin_az = math.cos(math.radians(inc_deg)) / math.cos(math.radians(lat_deg))
    sin_az = max(-1.0, min(1.0, sin_az))
    return v_site * sin_az


def launch_payload_factor(lat_deg, inc_deg, ref_assist):
    """Payload delivered to the drop-off, relative to the due-east
    reference. The extra ascent delta-v is charged against the upper
    stage, whose final mass is (ship dry + payload)."""
    dv_extra = ref_assist - rotation_assist(lat_deg, inc_deg)
    if dv_extra <= 0.0:
        return 1.0, dv_extra
    ve = ISP_UPPER * G0
    m_final_ref = M_SHIP_DRY + M_PAYLOAD_REF
    m_final = m_final_ref * math.exp(-dv_extra / ve)
    return max(0.0, (m_final - M_SHIP_DRY) / M_PAYLOAD_REF), dv_extra


# --------------------------------------------------- leg A: raise to 1000 km
def dv_raise_impulsive(r1, r2):
    """Coplanar Hohmann transfer."""
    a_t = 0.5 * (r1 + r2)
    dv1 = math.sqrt(MU * (2.0 / r1 - 1.0 / a_t)) - v_circ(r1)
    dv2 = v_circ(r2) - math.sqrt(MU * (2.0 / r2 - 1.0 / a_t))
    return dv1 + dv2


def dv_raise_lowthrust(r1, r2):
    """Edelbaum, coplanar: reduces to |v1 - v2|."""
    return abs(v_circ(r1) - v_circ(r2))


# ------------------------------------------------------------ leg B: to SEL1
def dv_sel1_impulsive(r1, d_l1):
    """Single burn onto a transfer reaching L1. Two-body about Earth;
    solar gravity makes the true cost slightly lower, so this mildly
    overcharges SEL1."""
    a_t = 0.5 * (r1 + d_l1)
    vp = math.sqrt(MU * (2.0 / r1 - 1.0 / a_t))
    return vp - v_circ(r1) + DV_L1_ARRIVAL


def dv_sel1_lowthrust(r1):
    """Slow spiral to escape costs ~v_circ: spread over many revolutions,
    it forfeits the Oberth effect."""
    return v_circ(r1) + DV_L1_ARRIVAL


# -------------------------------------------------------------------- main
def main():
    r_park = R_E + ALT_PARK
    r_target = R_E + ALT_TARGET
    inc_sso = sso_inclination(r_target)

    ref_assist = rotation_assist(LAT_EAST, LAT_EAST)   # due east = reference
    f_launch_a, dv_pad_a = launch_payload_factor(LAT_POLAR, inc_sso,
                                                 ref_assist)
    f_launch_b, dv_pad_b = launch_payload_factor(LAT_EAST, INC_SEL1_PARK,
                                                 ref_assist)

    print("Shade delivered per Starship launch\n")
    print("Leg A - 1000 km sun-synchronous shade")
    print(f"  launch      {LAT_POLAR:.1f} deg N range, direct to "
          f"i = {inc_sso:.2f} deg at {ALT_PARK:.0f} km")
    print(f"  rotation    {rotation_assist(LAT_POLAR, inc_sso)*1000:+.0f} m/s "
          f"vs {ref_assist*1000:+.0f} m/s due east  "
          f"(pad penalty {dv_pad_a*1000:.0f} m/s)")
    print(f"  on orbit    coplanar raise to {ALT_TARGET:.0f} km, "
          f"no plane change")
    print("\nLeg B - SEL1 shade")
    print(f"  launch      {LAT_EAST:.1f} deg N range, due east to "
          f"i = {INC_SEL1_PARK:.1f} deg at {ALT_PARK:.0f} km")
    print(f"  rotation    {rotation_assist(LAT_EAST, INC_SEL1_PARK)*1000:+.0f}"
          f" m/s (full assist, reference case)")
    print(f"  on orbit    trans-L1 injection; no plane change, since "
          f"{INC_SEL1_PARK:.1f} deg > {SUN_DECL_MAX:.2f} deg max solar "
          f"declination")

    drift = node_drift_deg_per_day(r_park, inc_sso) - 360.0 / 365.2422
    print(f"\n  [Leg A note] the {ALT_PARK:.0f} km parking orbit at "
          f"i = {inc_sso:.2f} deg is not itself sun-synchronous:")
    print(f"   its node drifts {drift:+.2f} deg/day against the Sun, i.e. "
          f"{drift*TRANSFER_DAYS:+.0f} deg ({drift*TRANSFER_DAYS*4:+.0f} min "
          f"of LTAN)")
    print(f"   over a {TRANSFER_DAYS:.0f}-day raise. Absorbed by launch "
          f"timing, so not charged as delta-v.")

    legs = {
        "impulsive": {
            "A: 1000 km SSO": dv_raise_impulsive(r_park, r_target),
            "B: SEL1": dv_sel1_impulsive(r_park, D_SEL1),
        },
        "lowthrust": {
            "A: 1000 km SSO": dv_raise_lowthrust(r_park, r_target),
            "B: SEL1": dv_sel1_lowthrust(r_park),
        },
    }
    f_launch = {"A: 1000 km SSO": f_launch_a, "B: SEL1": f_launch_b}

    print("\n" + "=" * 78)
    print(f"{'propulsion':28s} {'leg':16s} {'dv':>8s} {'launch':>8s} "
          f"{'in-space':>9s} {'net':>7s}")
    print("=" * 78)

    net = {}
    for pname, p in PROPULSION.items():
        ve = p["isp"] * G0
        mode = "lowthrust" if p["lowthrust"] else "impulsive"
        net[pname] = {}
        for leg, dv in legs[mode].items():
            f_space = math.exp(-dv / ve)
            n = f_launch[leg] * f_space
            net[pname][leg] = n
            print(f"{pname:28s} {leg:16s} {dv:5.2f} k/s "
                  f"{f_launch[leg]:7.1%} {f_space:8.1%} {n:6.1%}")
        print("-" * 78)

    print("\nShade film delivered per launch, per 100 t of due-east capacity:")
    for pname in PROPULSION:
        a = net[pname]["A: 1000 km SSO"] * 100.0
        b = net[pname]["B: SEL1"] * 100.0
        verdict = "SEL1 wins" if b > a else f"LEO wins by {a/b:.2f}x"
        print(f"  {pname:28s} {a:5.1f} t at 1000 km  vs {b:5.1f} t at SEL1"
              f"   ({verdict})")

    print("\nIn-space delta-v ratio, against the launch penalty:")
    for mode, label in [("impulsive", "chemical"), ("lowthrust", "SEP")]:
        ratio = legs[mode]["B: SEL1"] / legs[mode]["A: 1000 km SSO"]
        print(f"  {label:9s} in-space delta-v ratio {ratio:5.1f}x against "
              f"SEL1 - but the pad penalty for the")
        print(f"  {'':9s} retrograde SSO launch claws back most or all of it.")

    print("\nSensitivity - net advantage vs specific impulse:")
    print(f"  {'Isp (s)':>8s} {'mode':>10s} {'A net':>8s} {'B net':>8s} "
          f"{'A / B':>8s}")
    for isp, lowthrust in [(320, False), (380, False), (450, False),
                           (1000, True), (2000, True), (3000, True),
                           (5000, True)]:
        ve = isp * G0
        mode = "lowthrust" if lowthrust else "impulsive"
        a = f_launch_a * math.exp(-legs[mode]["A: 1000 km SSO"] / ve)
        b = f_launch_b * math.exp(-legs[mode]["B: SEL1"] / ve)
        print(f"  {isp:8.0f} {mode:>10s} {a:7.1%} {b:7.1%} {a/b:7.2f}x")

    print("\nSensitivity - result vs the assumed launch payload factor for")
    print("the retrograde SSO leg (the one soft number in this script):")
    for f in [0.65, 0.70, 0.756, 0.80, 0.85, 1.00]:
        ve = 3000.0 * G0
        a = f * math.exp(-legs["lowthrust"]["A: 1000 km SSO"] / ve)
        b = f_launch_b * math.exp(-legs["lowthrust"]["B: SEL1"] / ve)
        print(f"  factor {f:4.2f}   A {a:6.1%}   B {b:6.1%}   "
              f"A/B {a/b:5.2f}x")

    effectiveness(net)


# ======================================================================
# PART 2 - what the delivered film is worth once it arrives
# ======================================================================
def effectiveness(net):
    print("\n" + "=" * 78)
    print("PART 2 - SHADING EFFECTIVENESS, AND THE COMBINED RESULT")
    print("=" * 78)

    print("\nReproducing the published SEL1 sizings from the solid-angle")
    print("relation, as a check that we are reading them the same way:")
    print(f"  {'source':<32} {'standoff':>10} {'published':>11} "
          f"{'solid ang':>11} {'resid':>7} {'eps':>6}")
    eps_vals = []
    for label, d, f, area_pub in SEL1_SIZINGS:
        area_sa = disk_area_solid_angle(d, f)
        eps = eps_from_area(f, area_pub)
        eps_vals.append(eps)
        print(f"  {label:<32} {d/1e6:>7.2f} Gm {area_pub/1e6:>9.2f}e6 "
              f"{area_sa/1e6:>9.2f}e6 {100*(area_sa/area_pub - 1):>+6.1f}% "
              f"{eps:>6.3f}")
    print("  Both agree with the solid-angle relation to under 7%. The GREB")
    print("  disc is the smaller of the two relative to that estimate,")
    print("  because it models solar limb darkening: the centre of the solar")
    print("  disc is brighter, so occulting it is worth more than a uniform")
    print("  source would suggest. The McInnes row runs the other way by 2%,")
    print("  which is rounding on its quoted 1,450 km effective radius.")

    eps_sel1 = min(eps_vals)          # the conservative of the two
    print(f"\n  eps(SEL1) = {eps_vals[0]:.2f} / {eps_vals[1]:.2f} from the two "
          f"papers; we carry {eps_sel1:.2f}.")
    print(f"  eta(LEO)  = {LEO_DUTY:.3f} duty x {LEO_COSPROJ:.3f} cosine "
          f"projection = {ETA_LEO:.3f}")
    print(f"  Per square metre of film, SEL1 does "
          f"{eps_sel1/ETA_LEO:.2f}x the work of the 1,000 km carpet.")

    print("\nWhy eps < 1 at all - penumbra width against Earth's diameter:")
    theta_sun = 2.0 * R_SUN / AU
    for label, d, _, _ in SEL1_SIZINGS:
        print(f"  {d/1e6:.2f} Gm standoff -> penumbra {d*theta_sun:>7,.0f} km "
              f"vs Earth {2*R_E_MEAN:,.0f} km")
    print(f"  1,000 km LEO shade      -> penumbra "
          f"{1000.0*theta_sun:>7.1f} km: negligible, all of it lands on Earth.")

    print("\nCOMBINED: delivered mass x effectiveness, per Starship launch.")
    print(f"Same {SIGMA_FILM:.0f} g/m^2 film charged to both legs, so areal")
    print("density cancels and this is a like-for-like mass comparison.")
    print(f"  {'propulsion':<28} {'A: LEO':>8} {'B: SEL1':>8} {'B/A':>7}")
    for pname in PROPULSION:
        a = net[pname]["A: 1000 km SSO"] * ETA_LEO
        b = net[pname]["B: SEL1"] * eps_sel1
        print(f"  {pname:<28} {a:>8.4f} {b:>8.4f} {b/a:>6.2f}x")
    print("  Columns are effective blocked area per unit launched mass,")
    print("  normalised to a due-east Starship payload of 1.")

    sep = "solar electric (Isp 3000 s)"
    a = net[sep]["A: 1000 km SSO"] * ETA_LEO
    b = net[sep]["B: SEL1"] * eps_sel1
    print(f"\n  Headline (SEP, the report's baseline): SEL1 returns "
          f"{b/a:.2f}x the")
    print("  cooling per kilogram launched. Transport is very nearly a wash")
    print(f"  ({net[sep]['A: 1000 km SSO']:.1%} vs "
          f"{net[sep]['B: SEL1']:.1%} of payload delivered); the difference "
          f"is effectiveness.")

    print("\nSensitivity - SEL1 is charged 21 g/m^2 above, but the published")
    print("SEL1 designs are lighter, because nothing at SEL1 requires the")
    print("film to be reflective. Per kg, using each design's own density:")
    for label, sigma in SEL1_ALT_SIGMA:
        scale = SIGMA_FILM / sigma
        print(f"  {label:<26} {sigma:>5.1f} g/m^2  -> "
              f"{b*scale/a:>5.2f}x per kg launched")
    print("  Not claimed in the report; recorded so the 21 g/m^2 choice is")
    print("  visible as the conservative one.")


if __name__ == "__main__":
    main()
