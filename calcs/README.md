# Calculations for *Earth Orbit Sunshades: Constraints Assessment*

Every quantitative claim in the report is produced by one of the nine
scripts here. 

## Running them

```
pip install -r requirements.txt
python3 leo_shade_calcs.py
```

Python 3.9 or later, NumPy 1.22 or later (2.x is fine — the one API that
moved, `np.trapz` → `np.trapezoid`, is handled). Nothing else is needed.
Total runtime for all nine is about two minutes, dominated by
`drag_reboost.py` Part 3 (the RK4 check) and `kessler_crosssection.py`.

Run them from inside this directory; five of them import
`leo_shade_calcs.py` as the shared geometry module.

## What each script produces

| Script                      | Report sections | What it gives the report                                                                                                                                                                                                                                  |
| --------------------------- | --------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `leo_shade_calcs.py`        | 3.1, 3.2        | SSO inclination vs altitude (96.6° at 274 km, 97.7° at 575 km, 100.9° at 1,300 km); dawn-dusk shading season (150 d/yr at 274 km falling to 74 d/yr at 800 km). Also the shared geometry module for the four scripts below it.                            |
| `noon_midnight_calcs.py`    | 3.2, 4.5        | Shade-time fraction per orbit, 33.2% at 1,000 km to 40.8% at 274 km, every day of the year; the 0.827 cosine projection of the horizontal carpet; the matching eclipse fraction.                                                                          |
| `constellation_sizing.py`   | 3.3, 4.5        | Earth's intercept disk (1.275 × 10⁸ km²); η = 27.5% = 33.2% duty × 0.827 projection; the fleet for a 0.1% cut — 464,161 craft, 464,161 km², 9.75 Mt of film.                                                                                              |
| `penumbra_intensity.py`     | 3.3             | Penumbral spreading: a 100 m ribbon at 1,000 km casts a 9.4 km penumbra blocking about 1% at peak, no umbra.                                                                                                                                              |
| `elliptical_heliotropic.py` | 2.4             | Shade fraction for sun-apogee ellipses against the circular reference (33.2% circular, 27.6% at 1,000 × 8,000 km, 23.3% at 1,000 × 20,000 km); SRP acceleration on the film, 4.3 × 10⁻⁴ m/s² = 37 m/s per day; why the apsidal sun-lock cannot be closed. |
| `shade_fraction_table.py`   | 4.1             | Table 2: noon-midnight SSO shade fraction at 274, 350, 500, 575, 800, 1,000 and 2,000 km (40.8% falling to 27.5%), averaged over the year, with SSO inclination and the matching eclipse fraction. Reuses the geometry in `noon_midnight_calcs.py`, so shared altitudes match exactly. |
| `drag_reboost.py`           | 4.2–4.5         | Constraint A apart from the 4.1 shade-fraction table. Drag Δv vs altitude and ε; SRP eccentricity pumping; the J2-bounded ±360 km breathing on a 139-day cycle; uncontrolled lifetime; and the total annual budget that decides hold-circular against frozen-orbit at each altitude. |
| `sel1_comparison.py` | 6 | Two parts. **Part 1**, the freight bill: fraction of a Starship payload arriving as usable film at a 1,000 km SSO (75%) against SEL1 (77%), with the launch-inclination penalty and the low-thrust spiral charged in full; also the chemical-propulsion case (Isp 450 s), where the SSO delivers 1.45× more film. **Part 2**, what the film is worth on arrival: shadowing efficiency at SEL1 (0.33 from McInnes 2010, 0.34 from Sánchez and McInnes 2015, both reproduced from their published areas) against the carpet's 0.275, giving 1.24× more cooling per kilogram delivered to SEL1. |
| `kessler_crosssection.py`   | 5               | Constraint B. The 640–1,360 km swept shell; 3–4 strikes per year on each resident satellite, 12,000–32,000× the natural rate; 93,766 >10 cm debris strikes per year on the shade; why conjunction screening cannot work at this areal density.            |

## Assumptions shared across all of them

These are set identically wherever they appear, and were reconciled in
August 2026 after an audit found `drag_reboost.py` and §4 disagreeing:

- Film 21 g/m², drag coefficient 2.2, specular reflection.
- Shade craft 100 m × 10 km = 1 km², so 21 t of film per craft.
- Earth radius 6,371 km for shadow geometry, 6,378.137 km in J2 terms.
- Solar constant 1,361 W/m²; P_srp = 2 S₀/c (specular, normal incidence).
- Solar electric propulsion at Isp 3,000 s; chemical at Isp 300 s
  (450 s in `sel1_comparison.py`, where the stage is a real upper stage
  rather than a station-keeping thruster).
- Propellant fractions are **per year** unless a script says otherwise.
- Reference orbit for the steelman: 1,000 km noon-midnight SSO.

## Notes

**Atmospheric density.** The `RHO` table in `drag_reboost.py` is the only
place density enters, and it holds *representative* exponential-atmosphere
values, not an extraction from a model run. They agree with standard
reference tables (Vallado, CIRA-class) to about 25% and reproduce
NRLMSISE-00 behaviour qualitatively. This is not the binding uncertainty:
ε, the fraction of sail area presented to the airstream, is unknown to two
orders of magnitude and swamps it. Every table that depends on density
says so.

**ε.** Nothing in the literature pins the flatness and pointing accuracy of
a 100 m × 10 km membrane, because nothing like it has been flown. The
report sweeps ε from 10⁻³ to 10⁻¹ and uses 10⁻² as its working value.
Results scale linearly in ε.

**Shadowing efficiency at SEL1.** `sel1_comparison.py` Part 2 charges SEL1
a shadowing efficiency of 0.33, the more conservative of the two published
values. Note this is a different quantity from the ε above, which the
report reserves for the ram-area fraction. Shadowing efficiency is not a
free parameter: it falls as the square of the standoff distance, because
the Sun's 0.53° angular width spreads the shade's penumbra wider than the
Earth and the light that spills past the limb is lost. It therefore depends
on where the shade can hold station, which depends in turn on how hard
solar radiation pressure pushes it. A reflective shade is driven out to
McInnes's 2.36 million km optimum, where it is 0.33. A black or refracting
shade barely feels SRP, sits near the classical L1 point at 1.5 million km,
and reaches about 0.68 (Angel 2006). We charge SEL1 the reflective case and
the report's own 21 g/m² film — the assumption least favourable to SEL1.
The script prints the lighter alternatives as a sensitivity; the report
does not claim them.
