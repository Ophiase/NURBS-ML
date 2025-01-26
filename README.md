# NURBS-ML: Machine Learning for NURBS Modeling

A research project exploring machine learning techniques for NURBS (*Non-Uniform Rational B-Splines*) modeling and generation.

## Project Goals

### Phase 1: Core Functionality (Current)

- ✅ Basic NURBS curve implementation
- ✅ Visualization tools for 2D/3D NURBS
- ✅ Synthetic curve generation
- ✅ Basic interpolation methods

### Phase 2: ML Integration (In Development)

- ❌ NURBS Curve Autoencoder
  - NURBS Surface → Point Cloud → NURBS Surface
  - For trajectories.

- ❌ NURBS Surface Autoencoder
  - NURBS Surface → Point Cloud → NURBS Surface
  - For 3D mesh reconstruction

### Phase 3: Advanced Applications (Future)

- ❌ Constrained NURBS optimization
- ❌ Real-time NURBS prediction
- ❌ CAD (*Computer-Aided Design*) system integration

## Current Capabilities

Makefile commands

```bash
# Some demonstrations
make demo
# Unit tests
make tests
```

Specific Commands

```bash
# run a specific demo
python3 -m main --demo <which_demo> 
# eg: <which_demo> = basic, interpolation, surface, synthetic
```