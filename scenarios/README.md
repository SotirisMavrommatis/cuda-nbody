# Scenario files

A scenario is a plain text file that says **what is in the simulation and how it
moves**. To run one on Windows, drag the `.txt` file onto `simulate.bat`. The GIF ends up
in the `output` folder.

## Format

```text
# Lines starting with # are comments.

steps = 1500        # how many simulation steps
dt = 0.01           # time per step (smaller = more accurate, slower)
save_every = 6      # save a GIF frame every this many steps
view = 5            # the picture shows x and y from -view to +view

galaxy   count=4096  x=-3 y=-1  vx=0.3   spin=1
cluster  count=2048  x=3  y=1   vx=-0.4  mass=0.6 radius=0.25
body     x=0 y=4  vy=-0.5  mass=2
```

Each object line starts with its type, followed by `key=value` settings. Any setting you
leave out takes its default value.

### Settings for every object

| Setting | Meaning | Default |
|---|---|---|
| `x`, `y`, `z` | Position of the centre | 0 |
| `vx`, `vy`, `vz` | Velocity of the centre | 0 |
| `mass` | See each type below | 1 |

### `galaxy`: spiral disk galaxy

A heavy central body plus a thin disk of stars on circular orbits.

| Setting | Meaning | Default |
|---|---|---|
| `count` | Bodies in the galaxy (centre + stars) | 4096 |
| `mass` | Mass of the central body | 1 |
| `disk_mass` | Total mass of all the stars together | 0.1 |
| `radius` | Outer radius of the disk | 1.2 |
| `inner` | Inner radius of the disk (the empty area near the centre) | 0.2 |
| `spin` | `1` = counter-clockwise, `-1` = clockwise | 1 |
| `tilt` | Tilt of the disk around the x axis, in degrees | 0 |

### `cluster`: ball of stars (Plummer sphere)

A round cluster of stars in equilibrium: left alone, it neither collapses nor flies apart.

| Setting | Meaning | Default |
|---|---|---|
| `count` | Number of stars | 2048 |
| `mass` | Total mass of the cluster | 1 |
| `radius` | Scale radius. Half the mass lies within about 1.3 × radius. | 0.3 |

### `body`: a single heavy object

A black hole, a star or a planet: one point mass. `count` is always 1.

## Tips

- **Total bodies decide the speed.** Up to about 16K bodies runs in seconds, 65K takes
  around 15–20 s for 1500 steps, and beyond that it gets slow (the cost grows with N²).
- **Bound or flying apart?** To escape a mass `M` at distance `r`, an object needs a speed
  of about `sqrt(2·M/r)`. Slower than that and the two objects will eventually merge.
- **Things fly too fast or the stars scatter strangely:** lower `dt` (for example 0.005)
  and raise `steps` to match.
- **Objects leave the picture:** raise `view`.
- If a setting is misspelled, the program tells you which line is wrong.
