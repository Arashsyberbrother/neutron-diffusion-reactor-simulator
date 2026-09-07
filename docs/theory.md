# Reactor Physics and Neutron Diffusion Theory

## 1. Physical Motivation and Conservation Principle

Neutron transport in a nuclear reactor core is governed fundamentally by the linear Boltzmann neutron transport equation, which accounts for neutron streaming, collisions, absorption, and fission across phase space $(\mathbf{r}, E, \mathbf{\hat{\Omega}}, t)$. In large, weakly absorbing, highly scattering media where neutron angular distributions are quasi-isotropic, transport theory reduces through Fick's Law to the neutron diffusion approximation.

For a steady-state multiplying system in one spatial dimension $x \in [0, L]$ with one energy group, the neutron balance equation states:

$$\text{Leakage Rate} + \text{Absorption Rate} = \text{Fission Production Rate}$$

Mathematically:

$$\frac{d J(x)}{dx} + \Sigma_a \phi(x) = \frac{1}{k_{\text{eff}}} \nu\Sigma_f \phi(x)$$

where:
- $\phi(x)$ is the scalar neutron flux [$\text{cm}^{-2}\,\text{s}^{-1}$]
- $J(x)$ is the neutron current density [$\text{cm}^{-1}\,\text{s}^{-1}$]
- $\Sigma_a$ is the macroscopic absorption cross section [$\text{cm}^{-1}$]
- $\nu\Sigma_f$ is the macroscopic fission production cross section (average neutrons emitted per fission $\nu$ times fission cross section $\Sigma_f$) [$\text{cm}^{-1}$]
- $k_{\text{eff}}$ is the effective multiplication factor (eigenvalue)

Applying Fick's Law, $J(x) = -D \frac{d\phi(x)}{dx}$, where $D$ is the diffusion coefficient [$\text{cm}$], yields the standard steady-state 1D diffusion eigenvalue equation:

$$-D \frac{d^2\phi(x)}{dx^2} + \Sigma_a \phi(x) = \frac{1}{k_{\text{eff}}} \nu\Sigma_f \phi(x)$$

---

## 2. Boundary Conditions and Extrapolated Endpoints

For a bare homogeneous core slab of physical thickness $L$, neutrons escaping the outer surfaces do not return. In exact transport theory, the angular flux into the core from the outside is zero (vacuum boundary condition). In diffusion theory, this vacuum condition is approximated by:

$$\phi(x_{\text{extrapolated}}) = 0$$

where $x_{\text{extrapolated}} = -d$ and $L + d$, with linear extrapolation distance $d \approx 0.7104 \lambda_{\text{tr}}$ (where $\lambda_{\text{tr}} = 1 / \Sigma_{\text{tr}}$ is the transport mean free path).

For the baseline benchmark problem, we adopt the standard canonical textbook convention setting the zero-flux boundary condition directly at the slab boundaries:

$$\phi(0) = 0, \quad \phi(L) = 0$$

---

## 3. Analytical Reference Solution

For a homogeneous slab with constant material properties $D, \Sigma_a, \nu\Sigma_f$ and zero Dirichlet boundary conditions, the differential operator $-D \frac{d^2}{dx^2}$ has spatial eigenfunctions:

$$\phi_n(x) = C_n \sin\left(\frac{n \pi x}{L}\right), \quad n = 1, 2, 3, \dots$$

The corresponding second derivative is:

$$\frac{d^2\phi_n}{dx^2} = -\left(\frac{n \pi}{L}\right)^2 \phi_n(x) = -B_{g,n}^2 \phi_n(x)$$

where the geometric buckling for mode $n$ is:

$$B_{g,n}^2 = \left(\frac{n \pi}{L}\right)^2$$

Substituting this into the diffusion eigenvalue equation:

$$D B_{g,n}^2 \phi_n(x) + \Sigma_a \phi_n(x) = \frac{1}{k_n} \nu\Sigma_f \phi_n(x)$$

Factoring out $\phi_n(x) \ne 0$:

$$D B_{g,n}^2 + \Sigma_a = \frac{\nu\Sigma_f}{k_n}$$

Solving for the fundamental mode ($n = 1$) eigenvalue yields the exact analytical formula:

$$k_{\text{eff,analytical}} = \frac{\nu\Sigma_f}{\Sigma_a + D B_g^2}$$

where:

$$B_g^2 = \left(\frac{\pi}{L}\right)^2$$

### Benchmark Parameters and Evaluation

For the baseline research case:
- Slab length: $L = 100.0\text{ cm}$
- Diffusion coefficient: $D = 1.0\text{ cm}$
- Absorption cross section: $\Sigma_a = 0.02\text{ cm}^{-1}$
- Neutron production: $\nu\Sigma_f = 0.025\text{ cm}^{-1}$
- Thermal diffusion length: $L_d = \sqrt{D / \Sigma_a} = \sqrt{1.0 / 0.020} \approx 7.0711\text{ cm}$
- Infinite multiplication factor: $k_\infty = \nu\Sigma_f / \Sigma_a = 0.025 / 0.020 = 1.2500$

The geometric buckling is:

$$B_g^2 = \left(\frac{\pi}{100}\right)^2 \approx 9.8696044 \times 10^{-4}\text{ cm}^{-2}$$

The analytical fundamental eigenvalue is:

$$k_{\text{eff,analytical}} = \frac{0.025}{0.02 + 1.0 \times 9.8696044 \times 10^{-4}} = \frac{0.025}{0.02098696044} \approx 1.19121585$$

### Exact Criticality Condition

A nuclear reactor core is strictly critical when $k_{\text{eff}} = 1.0$. Setting the analytical eigenvalue to unity:

$$\frac{\nu\Sigma_f}{\Sigma_a + D B_g^2} = 1.0 \implies \nu\Sigma_f = \Sigma_a + D B_g^2$$

For the baseline core geometry ($L = 100\text{ cm}, D = 1.0\text{ cm}, \Sigma_a = 0.02\text{ cm}^{-1}$), the critical production cross section is:

$$\nu\Sigma_{f,\text{crit}} = 0.02 + 1.0 \times 9.8696044 \times 10^{-4} \approx 0.0209869604\text{ cm}^{-1}$$

---

## 4. Normalization and Core Power Density

Because the diffusion equation is a linear homogeneous eigenvalue problem, any scalar multiple of an eigenmode $\phi(x)$ satisfies the equation. In a real reactor, the absolute magnitude of the flux is determined by the total operational power level $P_{\text{core}}$ [$\text{MW}$]:

$$P_{\text{core}} = E_f \int_{\text{core}} \Sigma_f(x) \phi(x)\,dx$$

where $E_f \approx 200\text{ MeV} \approx 3.204 \times 10^{-11}\text{ J}$ is the recoverable energy released per fission event.

In this simulator, two standard scientific normalizations are supported:
1. Continuous $L_2$ integral normalization:
   $$\int_0^L \phi(x)^2\,dx = 1.0$$
   For the fundamental mode $\phi_1(x) = C \sin(\pi x / L)$, $\int_0^L \sin^2(\pi x / L)\,dx = L/2$, leading to exact normalization constant $C = \sqrt{2/L}$.
2. Relative Core Power Density $P(x) / \bar{P}$:
   $$P_{\text{rel}}(x) = \frac{\phi(x)}{\frac{1}{L} \int_0^L \phi(x)\,dx}$$
   ensuring the core-averaged relative power density is identically $1.0$.

---

## 5. Heterogeneous Media, Interface Physics, and Reflector Savings

### 5.1 Physical Interface Conditions

In multi-region heterogeneous reactors (such as a core surrounded by an external reflector), macroscopic material cross sections $\Sigma_a(x)$, $\nu\Sigma_f(x)$, and the diffusion coefficient $D(x)$ undergo step discontinuities across material interfaces $x_{\text{int}}$.

At any material interface without an infinitesimal singular surface source, conservation of neutrons requires two boundary continuity conditions:

1. **Continuity of Scalar Neutron Flux**:
   $$\phi(x_{\text{int}}^-) = \phi(x_{\text{int}}^+)$$

2. **Continuity of Net Neutron Current**:
   With the standard positive $x$-direction coordinate convention, Fick's Law states:
   $$J(x) = -D(x) \frac{d\phi(x)}{dx}$$

   Equating the net current across the interface:
   $$J(x_{\text{int}}^-) = J(x_{\text{int}}^+)$$
   $$-D_{\text{left}} \left.\frac{d\phi}{dx}\right|^- = -D_{\text{right}} \left.\frac{d\phi}{dx}\right|^+$$

   or equivalently:
   $$D_{\text{left}} \left.\frac{d\phi}{dx}\right|^- = D_{\text{right}} \left.\frac{d\phi}{dx}\right|^+$$

Because $D_{\text{left}} \ne D_{\text{right}}$, the spatial derivative of the flux is discontinuous across the interface:
$$\frac{\left.\frac{d\phi}{dx}\right|^+}{\left.\frac{d\phi}{dx}\right|^-} = \frac{D_{\text{left}}}{D_{\text{right}}}$$

### 5.2 Reflector Savings Physics

Surrounding an active core with a non-fissionable, low-absorption, high-scattering reflector (such as heavy water, light water, beryllium, or graphite) alters the neutron balance in several critical ways:
1. **Neutron Return (Albedo Effect)**: Neutrons that escape the core into the reflector undergo multiple scattering collisions. Rather than being lost irrevocably to vacuum, a large fraction are scattered back into the core, drastically reducing net core leakage.
2. **Reactivity Increase**: The reduced leakage increases the effective multiplication factor:
   $$\Delta k_{\text{refl}} = k_{\text{eff,reflected}} - k_{\text{eff,bare}} > 0$$
3. **Flux and Power Flattening**: The return of neutrons elevates the flux near the core perimeter. In a bare slab, the flux drops to zero at the boundary; in a reflected core, the flux at the fuel boundary remains substantial. This flattens the fission power density profile, significantly lowering the peak-to-average power ratio ($P_{\text{peak}} / \bar{P}_{\text{core}}$ drops from $1.571$ to $1.225$).
4. **Critical Size Reduction**: Because fewer neutrons are lost to leakage, a reflected reactor requires significantly less fissile fuel to attain criticality than a bare reactor. The reduction in critical core half-thickness is termed the **reflector saving** ($\delta$):
   $$T_{\text{crit,bare}} = T_{\text{crit,refl}} + 2\delta$$
