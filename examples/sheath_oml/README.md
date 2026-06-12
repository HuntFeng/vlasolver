# 1D Orbital-MotionLimited (OML) Sheath (Full Two-Species)

A **full two-species** (electron + ion) Vlasov-Poisson simulation of a OML sheath forming at a dielectric wall. Both species are evolved with the Vlasov equation; electrons are reflected by the sheath electric field while ions are accelerated toward the wall.

## Physics

- **Domain**: $[0, 20] \times [0, 20]$ in $(x, y)$
- **Wall**: at $y = 0$, biased to the floating potential $\phi_w = -\ln\sqrt{m_i / (2\pi m_e)}$
- **Normalization**: $T_e = 1$, $T_i = 0.1$, $m_i/m_e = 2 \times 1836$
- **Boundary conditions**: Periodic in $x$, Bohm sheath inflow/outflow in $y$
- **2st-order immersed interface Poisson solver**

## Build

```bash
cmake -B build && cmake --build build
```

## Run

```bash
build/sheath_oml examples/sheath_oml/input.ini
```

## Analysis

```bash
python examples/sheath_oml/plottings.py
```

## Results

![Potential](figures/potential.png)


Near the wall ($y \to 0$), the potential drops sharply, electrons are repelled, and ions are accelerated to the Bohm velocity. The charge will be deposited on the wall, and the sheath will reach a steady state where the ion flux to the wall balances the electron flux.
