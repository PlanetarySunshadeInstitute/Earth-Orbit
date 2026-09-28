# Earth Orbit Sunshades: Constraints Assessment — calculations

Python calculations behind the Planetary Sunshade Institute report
**Earth Orbit Sunshades: Constraints Assessment** (Morgan Goodwin, August 2026),
part of the ARIA-funded Space Reflector Baseline Survey.

**Read the report:** [planetarysunshade.org/publications](https://planetarysunshade.org/publications/)

## What the report finds

The report narrows Earth-orbit sunshade concepts to their strongest candidate,
a noon-midnight sun-synchronous "magic carpet" at 1,000 km, and tests it
against drag, solar radiation pressure, constellation scale, collision risk
and launch economics. A 0.1% cut in sunlight would need about 464,000 craft
and 9.75 million tonnes of film in a 720 km-thick shell, and every satellite
in that shell would be struck three to four times a year. A Starship delivers
slightly more usable film to the Sun-Earth L1 point than to that orbit, and
each kilogram there buys about 1.24× the cooling. Earth-orbit sunshades are
not physically impossible; they are the more expensive way to buy less shade.

## The code

Every quantitative claim in the report comes from one of the nine scripts in
[`calcs/`](calcs/). [`calcs/README.md`](calcs/README.md) maps each script to
the report sections and numbers it produces, and lists the shared assumptions.

```
cd calcs
pip install -r requirements.txt
python3 leo_shade_calcs.py
```

Requires Python 3.9+ and NumPy 1.22+ (NumPy 2.x works). Run scripts from
inside `calcs/`, since several import `leo_shade_calcs.py` as a shared module.
Each script prints its results to the terminal.

## Citing

See [`CITATION.cff`](CITATION.cff), or cite the report directly:

> Goodwin, Morgan. 2026. *Earth Orbit Sunshades: Constraints Assessment.*
> Planetary Sunshade Institute.

## License

Code released under the [MIT License](LICENSE). The report itself is
published separately at
[planetarysunshade.org/publications](https://planetarysunshade.org/publications/).
