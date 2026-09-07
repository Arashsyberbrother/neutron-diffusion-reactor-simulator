# Numerical Discretization and Algorithmic Formulation

## 1. Second-Order Central Finite Difference Scheme

Consider a uniform spatial mesh partitioning the interval $[0, L]$ into $N$ equal cells with node spacing $\Delta x = L / N$. The nodes are located at coordinates:

$$x_i = i \Delta x, \quad i = 0, 1, 2, \dots, N$$

The boundary nodes are $x_0 = 0$ and $x_N = L$. With Dirichlet zero-flux conditions $\phi_0 = 0$ and $\phi_N = 0$, the remaining unknowns are the interior nodal fluxes:

$$\boldsymbol{\phi} = [\phi_1, \phi_2, \dots, \phi_{N-1}]^T \in \mathbb{R}^{N-1}$$

Using Taylor series expansion around node $x_i$:

$$\phi(x_i + \Delta x) = \phi_i + \Delta x \phi'_i + \frac{\Delta x^2}{2} \phi''_i + \frac{\Delta x^3}{6} \phi'''_i + \frac{\Delta x^4}{24} \phi^{(4)}_i + \mathcal{O}(\Delta x^5)$$

$$\phi(x_i - \Delta x) = \phi_i - \Delta x \phi'_i + \frac{\Delta x^2}{2} \phi''_i - \frac{\Delta x^3}{6} \phi'''_i + \frac{\Delta x^4}{24} \phi^{(4)}_i + \mathcal{O}(\Delta x^5)$$

Adding both expansions and solving for $\phi''_i$ yields the standard central difference approximation:

$$\left.\frac{d^2\phi}{dx^2}\right|_{x_i} = \frac{\phi_{i-1} - 2\phi_i + \phi_{i+1}}{\Delta x^2} - \frac{\Delta x^2}{12} \phi^{(4)}(\xi_i)$$

The truncation error is strictly second order, $\mathcal{O}(\Delta x^2)$.

---

## 2. Discrete Loss and Production Operators

Substituting the discrete second derivative into the continuous diffusion equation at interior node $i$:

$$-D \left[\frac{\phi_{i-1} - 2\phi_i + \phi_{i+1}}{\Delta x^2}\right] + \Sigma_a \phi_i = \frac{1}{k_{\text{eff}}} \nu\Sigma_f \phi_i$$

Rearranging into operator form:

$$\left(-\frac{D}{\Delta x^2}\right) \phi_{i-1} + \left(\frac{2D}{\Delta x^2} + \Sigma_a\right) \phi_i + \left(-\frac{D}{\Delta x^2}\right) \phi_{i+1} = \frac{1}{k_{\text{eff}}} \nu\Sigma_f \phi_i$$

This defines the linear generalized matrix eigenvalue problem:

$$\mathbf{A} \boldsymbol{\phi} = \frac{1}{k_{\text{eff}}} \mathbf{F} \boldsymbol{\phi}$$

### Mathematical Properties of Loss Matrix $\mathbf{A}$

1. **Sparsity**: $\mathbf{A}$ is strictly tridiagonal with $N-1$ interior unknowns.
2. **Symmetry**: $A_{i, i+1} = A_{i+1, i} = -D / \Delta x^2$.
3. **Strict Diagonal Dominance**: For $\Sigma_a > 0$:
   $$|A_{i,i}| = \frac{2D}{\Delta x^2} + \Sigma_a > \sum_{j \ne i} |A_{i,j}| = \frac{2D}{\Delta x^2}$$
4. **Positive Definiteness**: All eigenvalues of $\mathbf{A}$ are strictly positive ($\lambda_{\min}(\mathbf{A}) > 0$).
5. **M-Matrix Character and Inverse Positivity**: The off-diagonal entries are non-positive ($A_{i,j} \le 0$ for $i \ne j$), diagonal entries are positive, and $\mathbf{A}$ is strictly diagonally dominant for $\Sigma_a > 0$. Under these conditions, $\mathbf{A}$ is a non-singular M-matrix satisfying the inverse-positivity property $\mathbf{A}^{-1} \ge 0$, which ensures that any non-negative source $\mathbf{s} \ge 0$ yields a physically meaningful non-negative flux solution $\boldsymbol{\phi} = \mathbf{A}^{-1}\mathbf{s} \ge 0$.
   Separately, in the context of the generalized eigenvalue problem $\mathbf{A}\boldsymbol{\phi} = \frac{1}{k_{\text{eff}}} \mathbf{F}\boldsymbol{\phi}$, Perron-Frobenius theory applies to the non-negative iteration operator $\mathbf{T} = \mathbf{A}^{-1}\mathbf{F}$, ensuring that the dominant eigenvalue $k_{\text{eff}} = \rho(\mathbf{T})$ is unique and positive, with a corresponding strictly non-negative fundamental eigenmode.

---

## 3. The Thomas Algorithm (TDMA)

At each step of the iterative eigenvalue solver, we solve a fixed-source diffusion problem:

$$\mathbf{A} \boldsymbol{\phi} = \mathbf{s}$$

Because $\mathbf{A}$ is tridiagonal and strictly diagonally dominant, Gaussian elimination without pivoting is unconditionally stable and requires only $\mathcal{O}(N)$ operations.

Let the tridiagonal system be expressed as:

$$a_i \phi_{i-1} + b_i \phi_i + c_i \phi_{i+1} = s_i, \quad i = 0, \dots, n-1$$

### Forward Elimination:
$$c'_0 = \frac{c_0}{b_0}, \quad s'_0 = \frac{s_0}{b_0}$$

$$c'_i = \frac{c_i}{b_i - a_i c'_{i-1}}, \quad s'_i = \frac{s_i - a_i s'_{i-1}}{b_i - a_i c'_{i-1}}, \quad i = 1, \dots, n-1$$

### Back Substitution:
$$\phi_{n-1} = s'_{n-1}$$

$$\phi_i = s'_i - c'_i \phi_{i+1}, \quad i = n-2, \dots, 0$$

The total operation count is $5(N-1)$ floating point operations, dramatically outperforming general $\mathcal{O}(N^3)$ matrix inversion algorithms.

---

## 4. The Power Iteration Algorithm

Power iteration is the standard workhorse algorithm for neutron diffusion and transport eigenvalue problems in reactor physics. The iteration operates as follows:

```
Initialize: phi^(0) > 0, k^(0) = 1.0, m = 0
Repeat:
  1. Compute fission source vector:
     s^(m) = (1 / k^(m)) * F * phi^(m)

  2. Solve fixed-source diffusion equation:
     A * phi_tilde^(m+1) = s^(m)

  3. Compute updated fission production and eigenvalue:
     P^(m+1) = \int_0^L \nu\Sigma_f \phi_tilde^(m+1) dx
     P^(m)   = \int_0^L \nu\Sigma_f \phi^(m) dx
     k^(m+1) = k^(m) * (P^(m+1) / P^(m))

  4. Normalize flux for numerical stability:
     phi^(m+1) = phi_tilde^(m+1) / ||phi_tilde^(m+1)||_{L2}

  5. Evaluate convergence criteria:
     delta_k  = |k^(m+1) - k^(m)| < tol_k
     res_flux = ||phi^(m+1) - phi^(m)||_2 / ||phi^(m+1)||_2 < tol_flux

  6. If converged, exit; else m = m + 1
```

### Convergence Rate and Dominance Ratio

The error in mode $n$ decreases per iteration proportional to the dominance ratio:

$$\sigma = \frac{k_2}{k_1} = \frac{\Sigma_a + D B_{g,1}^2}{\Sigma_a + D B_{g,2}^2} < 1$$

For a $100\text{ cm}$ slab, $B_{g,1}^2 = (\pi / 100)^2 \approx 9.87 \times 10^{-4}\text{ cm}^{-2}$ while $B_{g,2}^2 = (2\pi / 100)^2 \approx 3.95 \times 10^{-3}\text{ cm}^{-2}$. The dominance ratio is:

$$\sigma = \frac{0.02 + 1.0 \times 9.87 \times 10^{-4}}{0.02 + 1.0 \times 3.95 \times 10^{-3}} \approx \frac{0.020987}{0.023948} \approx 0.876$$

With $\sigma \approx 0.876$, the spectral radius ensures rapid and reliable geometric convergence in approximately 35 to 45 iterations.

The characteristic spatial scale of the discrete diffusion process is governed by the thermal diffusion length:

$$L_d = \sqrt{\frac{D}{\Sigma_a}} = \sqrt{\frac{1.0}{0.020}} \approx 7.07106781\text{ cm}$$

which confirms that the baseline grid spacing $\Delta x = 1.0\text{ cm}$ adequately resolves the continuous spatial flux gradients ($L_d / \Delta x \approx 7.07$).

---


## 5. Conservative Discretization for Heterogeneous Media

In multi-region heterogeneous reactors (e.g., fuel core surrounded by a non-multiplying reflector), cross sections and diffusion coefficients vary discontinuously across material interfaces:

$$-\frac{d}{dx}\left(D(x) \frac{d\phi}{dx}\right) + \Sigma_a(x)\phi(x) = \frac{1}{k_{\text{eff}}} \nu\Sigma_f(x)\phi(x)$$

Standard point-wise finite differencing fails across material discontinuities because $D(x)$ is not differentiable at interfaces. Instead, a conservative cell-centered / control-volume finite-difference discretization is required.

### 5.1 Control Volume Balance and Interface Currents

Integrating the diffusion equation over a control volume cell $[x_{i-1/2}, x_{i+1/2}]$ centered at node $x_i$:

$$-\int_{x_{i-1/2}}^{x_{i+1/2}} \frac{d}{dx}\left(D(x) \frac{d\phi}{dx}\right) dx + \int_{x_{i-1/2}}^{x_{i+1/2}} \Sigma_a(x)\phi(x) dx = \frac{1}{k_{\text{eff}}} \int_{x_{i-1/2}}^{x_{i+1/2}} \nu\Sigma_f(x)\phi(x) dx$$

Applying the fundamental theorem of calculus:

$$J_{i+1/2} - J_{i-1/2} + \bar{\Sigma}_{a,i} \phi_i \Delta x = \frac{1}{k_{\text{eff}}} \overline{\nu\Sigma}_{f,i} \phi_i \Delta x$$

where $J(x) = -D(x) \frac{d\phi}{dx}$ denotes the neutron current in the positive $x$-direction.

### 5.2 Derivation of the Harmonic Mean Diffusion Coefficient

Consider the cell interface $x_{i+1/2}$ located midway between node $x_i$ (material properties $D_i$) and node $x_{i+1}$ (material properties $D_{i+1}$). Assuming uniform properties within each half-cell of length $\Delta x / 2$:

The current from node $i$ to the interface $x_{i+1/2}$ is:

$$J_{i+1/2} = -D_i \frac{\phi_{i+1/2} - \phi_i}{\Delta x / 2}$$

Similarly, the current from the interface to node $i+1$ is:

$$J_{i+1/2} = -D_{i+1} \frac{\phi_{i+1} - \phi_{i+1/2}}{\Delta x / 2}$$

Enforcing flux continuity $\phi(x_{i+1/2}^-) = \phi(x_{i+1/2}^+)$ and solving for the interface flux:

$$\phi_{i+1/2} = \frac{D_i \phi_i + D_{i+1} \phi_{i+1}}{D_i + D_{i+1}}$$

Substituting $\phi_{i+1/2}$ back into the current expression yields:

$$J_{i+1/2} = - \left(\frac{2 D_i D_{i+1}}{D_i + D_{i+1}}\right) \frac{\phi_{i+1} - \phi_i}{\Delta x} = -D_{i+1/2} \frac{\phi_{i+1} - \phi_i}{\Delta x}$$

where the interface diffusion coefficient is the **harmonic mean**:

$$D_{i+1/2} = \frac{2 D_i D_{i+1}}{D_i + D_{i+1}}$$

The harmonic mean is physically rigorous because neutron diffusion resistance acts in series across the interface:

$$\frac{\Delta x}{D_{i+1/2}} = \frac{\Delta x / 2}{D_i} + \frac{\Delta x / 2}{D_{i+1}}$$

### 5.3 Heterogeneous Tridiagonal Matrix Formulation

Dividing the discrete balance equation by $\Delta x$, the matrix elements for interior node $i$ ($i = 1, \dots, N-1$) are:

$$a_i \phi_{i-1} + b_i \phi_i + c_i \phi_{i+1} = \frac{1}{k_{\text{eff}}} f_i \phi_i$$

where:

$$a_i = -\frac{D_{i-1/2}}{\Delta x^2}$$

$$c_i = -\frac{D_{i+1/2}}{\Delta x^2}$$

$$b_i = \frac{D_{i-1/2} + D_{i+1/2}}{\Delta x^2} + \bar{\Sigma}_{a,i}$$

$$f_i = \overline{\nu\Sigma}_{f,i}$$

For interface nodes where a material boundary coincides exactly with a grid node $x_i$, the nodal macroscopic cross sections represent the average of the two adjacent half-cells:

$$\bar{\Sigma}_{a,i} = \frac{1}{2} (\Sigma_{a,\text{left}} + \Sigma_{a,\text{right}}), \quad \overline{\nu\Sigma}_{f,i} = \frac{1}{2} (\nu\Sigma_{f,\text{left}} + \nu\Sigma_{f,\text{right}})$$

### 5.4 Discrete Interface Continuity Verification

At a material interface between core (fuel) and reflector, physical solutions must satisfy:
1. **Flux continuity**: $\phi(x_{\text{int}}^-) = \phi(x_{\text{int}}^+)$
2. **Current continuity**: $J(x_{\text{int}}^-) = J(x_{\text{int}}^+)$, where $J = -D \frac{d\phi}{dx}$

In the discrete formulation:
- Scalar flux continuity is satisfied within discretization error:
  $$\Delta \phi_{\text{int}} = |\phi_{i} - \phi_{i}| = 0 \quad \text{(at node)}, \quad |\phi(x_{\text{int}}^+) - \phi(x_{\text{int}}^-)| \le \mathcal{O}(\Delta x)$$
- Discrete face current continuity is satisfied directly by the flux gradient across the interface:
  $$J_{i+1/2} = -D_{i+1/2} \frac{\phi_{i+1} - \phi_i}{\Delta x}$$
  Because the same numerical current flux leaving cell $i$ enters cell $i+1$, net current balance is conserved to machine precision across all cell faces.

