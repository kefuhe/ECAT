# Sbarbot Vertical Strain Volume — Parameter & Sign Convention Reference

## Model Reference

Barbot S., Moore J. D., and Lambert V. (2017),
*Displacement and Stress Associated with Distributed Anelastic Deformation
in a Half-Space*, Bulletin of the Seismological Society of America,
**107**(2), 821–855. doi: 10.1785/0120160237

---

## Coordinate System

Sbarbot uses a **local Cartesian frame** with base at the free surface:

| Axis | Direction | CSI mapping |
|------|-----------|-------------|
| x1'  | Along-strike (North when θ=0) |  `y` (Northing) |
| x2'  | Perpendicular to strike, horizontal (East when θ=0) | `x` (Easting) |
| x3   | **Down** (positive into the Earth) | `depth` |

> **Note:** x3 is always downward, regardless of θ. The primed axes x1', x2'
> rotate with the strike angle θ, but in a standard θ=0 configuration,
> x1'=North and x2'=East.

---

## Source Geometry

A **vertical strain volume** is a cuboid (box) buried in a half-space.
It is defined by a reference point (q1, q2, q3) and three dimensions (L, T, W):

```
                        Strike direction (x1')  θ=0 → North
                              ↑
                              |
          ┌───────────────────┬───────────────────┐ ← depth = q3  (top)
          │                   │                   │
          │        q2-T/2     │(q1,q2)  q2+T/2   │   T = thickness
          │         ←─────────┼─────────→         │   (horiz, ⊥ strike)
          │                   │                   │
          │                   │                   │
          │                   │ L (along strike)  │
          │                   │                   │
          │                   │                   │
          └───────────────────┴───────────────────┘ ← depth = q3+W (bottom)
                         x1' = q1+L

          ← x2' →

     Plan view (looking DOWN):         Cross-section (looking along strike):

      q2-T/2        q2+T/2                  free surface
         ├─── T ────┤                    ════════════════════
         │          │                    ┊                  ┊
   q1 →  │  Source  │              q3 →  ┌──────────────────┐
         │  Volume  │                    │   Strain Volume   │
   q1+L→ │          │            q3+W →  └──────────────────┘
         └──────────┘                    ┊                  ┊
```

### Parameter table

| Parameter | Unit | Description |
|-----------|------|-------------|
| `q1`      | km   | x1' coordinate of reference corner (start of L; center of T) |
| `q2`      | km   | x2' coordinate of reference corner (center of T; start of L measured from here in x1') |
| `q3`      | km   | **Top** depth of the volume (positive down) |
| `L`       | km   | Length along strike, x1' direction: volume spans [q1, q1+L] |
| `T`       | km   | Thickness perpendicular to strike, x2' direction: volume spans [q2−T/2, q2+T/2] |
| `W`       | km   | Width (depth extent), x3 direction: volume spans [q3, q3+W] |
| `theta`   | rad  | Strike angle measured clockwise from North (in CSI: from +y axis) |

### CSI class mapping (`Sbarbot.addVolume`)

```python
sbar.addVolume(x, y, depth, L, T, W, strike, eps=...)
```

| CSI arg   | Sbarbot param | Notes |
|-----------|---------------|-------|
| `x`       | q2            | Easting of volume center (converted to x2') |
| `y`       | q1            | Northing of volume start (converted to x1') |
| `depth`   | q3            | Top depth in km |
| `L`       | L             | Along-strike length |
| `T`       | T             | Cross-strike thickness |
| `W`       | W             | Depth extent |
| `strike`  | θ             | Degrees, converted to radians internally |

---

## Strain Components & Sign Conventions

The source volume undergoes **six independent inelastic strain components**.
Below, directions are described for the default case **θ = 0** (strike = North).

### Quick reference table

| Component | Positive (+ε) | Negative (−ε) |
|-----------|---------------|---------------|
| **eps11** | N-S extension (outward push ± N/S), uplift | N-S compression, subsidence |
| **eps22** | E-W extension (outward push ± E/W), uplift | E-W compression, subsidence |
| **eps33** | Vertical extension → **surface UPLIFT** | Vertical compression → **surface SUBSIDENCE** |
| **eps12** | **Left-lateral** shear (E→N, W→S) | **Right-lateral** shear (E→S, W→N) |
| **eps13** | Along-strike vertical shear (deep→+strike; surface→opposite) | Reverse sense |
| **eps23** | Cross-strike vertical shear (deep→+x2'; surface→opposite) | Reverse sense |

### Detailed descriptions

#### Normal strains (eps11, eps22, eps33) — volume changes

**eps11** — Along-strike normal strain (x1'x1'):
- **+eps11**: The volume extends in the N-S (along-strike) direction.
  Surface is pushed outward to the north and south. Moderate uplift above the source.
- **−eps11**: Compression in x1'. Surface pulled inward, moderate subsidence.
- **Fault analogy**: ±eps11 ≈ opening/closing of a vertical fracture perpendicular to strike (E-W fault plane).

**eps22** — Cross-strike normal strain (x2'x2'):
- **+eps22**: The volume extends in the E-W (cross-strike) direction.
  Surface pushed eastward and westward. Moderate uplift.
- **−eps22**: Compression in x2'. Surface pulled inward, moderate subsidence.
- **Fault analogy**: ±eps22 ≈ opening/closing of a vertical fracture parallel to strike (N-S fault plane, like a dike).

**eps33** — Vertical normal strain (x3 x3):
- **+eps33**: The volume extends vertically (grows taller). **Strong surface UPLIFT** directly above the source. This is the dominant uplift component.
- **−eps33**: The volume compresses vertically. **Surface SUBSIDENCE** above the source.
- **Fault analogy**: ±eps33 ≈ opening/closing of a horizontal fracture (sill). +eps33 acts like sill inflation.

> **Important**: +eps33 causes **uplift**, not subsidence. The volume grows into
> the surrounding medium and pushes the free surface upward — similar to a
> pressure source or Mogi inflation.

#### Shear strains (eps12, eps13, eps23) — shape changes

**eps12** — Horizontal shear (x1'x2'):
- **+eps12 = Left-lateral**: East side moves North, West side moves South.
  Also: North side moves East, South side moves West.
  Purely horizontal, negligible vertical displacement.
- **−eps12 = Right-lateral**: East side moves South, West side moves North.
- **Fault analogy**: Direct analogy with strike-slip faulting on a vertical fault.

**eps13** — Along-strike vertical shear (x1'x3):
- **+eps13**: Within the volume, deeper layers are displaced along +x1' (North for θ=0) relative to shallower layers. At the **surface**, this produces:
  - Horizontal displacement **opposite to strike** (southward for θ=0)
  - **Uplift** on the opposite-to-strike side (south), **subsidence** on the along-strike side (north)
- **−eps13**: Reverse sense — surface moves along strike (northward), north-side uplift.
- **Note**: eps13 does NOT have a simple "thrust" or "normal" fault analogy.
  The surface displacement pattern reflects the elastic response of a buried
  shear zone, not a discrete fault dislocation.

**eps23** — Cross-strike vertical shear (x2'x3):
- **+eps23**: Within the volume, deeper layers are displaced along +x2' (East for θ=0) relative to shallower layers. At the **surface**, this produces:
  - Horizontal displacement **opposite to +x2'** (westward for θ=0)
  - **Uplift** on the −x2' side (west), **subsidence** on the +x2' side (east)
- **−eps23**: Reverse sense — surface moves eastward, east-side uplift.
- **Note**: Same caveat as eps13 — no simple thrust/normal fault analogy.

---

## Numerical Verification of Sign Conventions

The following values were computed at diagnostic surface points for a
reference source at (q1=0, q2=0, q3=10, L=60, T=60, W=20, θ=0), with
ε=1×10⁻³ and G=30 GPa, ν=0.25.

### +eps12 (Left-lateral)
```
East  side (0, +30): u1(N) = +4.62e-3   → East side moves North  ✓
West  side (0, -30): u1(N) = -4.62e-3   → West side moves South  ✓
Vertical component ≈ 0                  → Pure horizontal shear   ✓
```

### +eps13 (Along-strike vertical shear)
```
Center (30, 0):      u1(N) = -1.13e-2   → Surface moves SOUTH
South side (-10, 0): u3(Up) = +6.18e-3  → Uplift on south side
North side (70, 0):  u3(Up) = -6.18e-3  → Subsidence on north side
```

### +eps23 (Cross-strike vertical shear)
```
Center (30, 0):      u2(E) = -7.88e-3   → Surface moves WEST
West  side (30,-30): u3(Up) = +6.87e-3  → Uplift on west side
East  side (30,+30): u3(Up) = -6.87e-3  → Subsidence on east side
```

### +eps33 (Vertical extension)
```
Center (30, 0):      u3(Up) = +1.54e-2  → Strong UPLIFT above source  ✓
```

### +eps11 (N-S extension)
```
North side (70, 0):  u1(N) = +5.94e-3   → Pushed northward (outward)
South side (-10,0):  u1(N) = -5.94e-3   → Pushed southward (outward)
Center (30, 0):      u3(Up) = +3.28e-3  → Moderate uplift            ✓
```

### +eps22 (E-W extension)
```
East side (30,+30):  u2(E) = +5.65e-3   → Pushed eastward (outward)
West side (30,-30):  u2(E) = -5.65e-3   → Pushed westward (outward)
Center (30, 0):      u3(Up) = +2.14e-3  → Moderate uplift            ✓
```

---

## Common Geophysical Applications

| Geophysical process | Primary strain component(s) |
|--------------------|-----------------------------|
| Interseismic creep on a strike-slip fault | eps12 (left- or right-lateral) |
| Dike inflation (vertical, parallel to strike) | +eps22 (opening ⊥ strike) |
| Sill inflation | +eps33 (vertical opening) |
| Post-seismic relaxation (viscoelastic) | Combination depending on co-seismic mechanism |
| Magma chamber pressurization | +eps11 + eps22 + eps33 (isotropic) |
| Aseismic afterslip in shear zone | eps12, eps13, eps23 depending on geometry |

---

## Material Parameters

| Parameter | Symbol | Default | Unit |
|-----------|--------|---------|------|
| Shear modulus | G | 30×10⁹ | Pa |
| Poisson's ratio | ν | 0.25 | — |

The displacement solution is **linear** in G⁻¹ and ε. Doubling the strain
doubles the displacement; doubling G halves it.

---

## Diagnostic Plots

Sign convention diagnostic plots are available in the test directory:

| File | Content |
|------|---------|
| `sbarbot_sign_eps11.png` | eps11 (N-S extension) surface displacements |
| `sbarbot_sign_eps12.png` | eps12 (left-lateral) surface displacements |
| `sbarbot_sign_eps13.png` | eps13 (along-strike vertical shear) |
| `sbarbot_sign_eps22.png` | eps22 (E-W extension) surface displacements |
| `sbarbot_sign_eps23.png` | eps23 (cross-strike vertical shear) |
| `sbarbot_sign_eps33.png` | eps33 (vertical extension / UPLIFT) |
| `sbarbot_sign_cross_sections.png` | Cross-section profiles for all components |
| `sbarbot_fault_analogies.png` | Fault analogy comparison |
| `sbarbot_opening_closing.png` | Opening vs closing modes |

These are generated by `test_sbarbot_sign_conventions.py`.
