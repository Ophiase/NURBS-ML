# NURBS-ML: Machine Learning for NURBS Modeling

A research project exploring machine learning techniques for NURBS (*Non-Uniform Rational B-Splines*) modeling and generation.

## Project Goals

### Phase 1: Core Functionality (Current)

- ✅ Basic NURBS curve implementation
- ✅ Visualization tools for 2D/3D NURBS
- ✅ Synthetic curve generation
- ✅ Basic interpolation methods

### Phase 2: ML Integration (In Development)

- ❌ Closest NURBS Curve
  - (NURBS Curve, Point Cloud) $\to$ NURBS Surface
  - For trajectories.

- ❌ NURBS Curve Autoencoder
  - NURBS Curve $\to$ Point Cloud $\to$ NURBS curve
  - For trajectories.

- ❌ NURBS Surface Autoencoder
  - NURBS Surface → Point Cloud $\to$ NURBS Surface
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

## Closest NURBS Curve

Given a point cloud $P = \set{p_i \in \mathbb{R}^3}_{i=1}^N$, we want to find a NURBS curve $C(t)$ that minimizes the distance to the point cloud. The curve is defined by:
- **Control points** $\{c_j \in \mathbb{R}^3\}_{j=1}^M$
- **Weights** $\{w_j \in \mathbb{R}\}_{j=1}^M$
- **Knot vector** $\{u_k\}_{k=1}^{M+d+1}$ (where $d$ is the degree)

To simplify, we can set all the weights to $1.0$. The number of control point $M$ is also a fixed hyper parameter. To simplify even more, we can also take uniform knot.

### Loss Function


The loss function measures the average distance between the point cloud and the NURBS curve:

$$
\mathcal{L} = \frac{1}{N} \sum_{i=1}^N \min_{t_i} \|C(t_i) - p_i\|^2
$$

- $t_i$ is the parameter value that minimizes the distance for point $p_i$.

### Optimization Strategy

1. **Initialization**:
  - Initialize control points, weights, and knots.
  - Use uniform knots and equal weights for simplicity.

2. **Stochastic Gradient Descent (SGD)**:
  - Shuffle the point cloud and divide it into batches of size $B$.
  - For each batch:
    - Compute the closest point on the curve for each $ p_i $ in the batch.
    - Compute the gradient of the loss with respect to the parameters.
    - Update the parameters using the gradient.

3. **Closest Point Calculation**:
  - For each point $p_i$, find $t_i$ that minimizes $\|C(t_i) - p_i\|$.
  - Use a numerical optimizer (e.g., Brent's method) for this 1D minimization.

4. **Gradient Computation**:
  - Use automatic differentiation (e.g., PyTorch) to compute gradients.