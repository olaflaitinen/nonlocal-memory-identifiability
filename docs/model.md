# Model, kernels and identifiability formulas

This file records the equations implemented by the `nmi` package, with the same
numbering as the manuscript.

## Model (1)

On the torus of length L in n spatial dimensions, the population density u and
the cognitive map k satisfy

```
u_t = d Laplacian(u) - alpha div( u grad( G_R * k ) )
k_t = beta u - mu k
```

with d, alpha, beta, mu > 0 and perceptual range R > 0. The total mass
M = integral of u is conserved, and the mean density is u_bar = M / L^n.

Implemented in `nmi.spectral_1d.simulate_1d`, `nmi.spectral_2d.simulate_2d` and
`nmi.finite_volume.simulate_fv_1d`.

## Kernels, equations (2) and (3), and Remark 2

One dimension:

```
top-hat:   G(y) = (1/2) 1_[-1,1](y),    G_hat(s) = sin(s) / s
Gaussian:  G(y) = (2 pi)^(-1/2) exp(-y^2 / 2),   G_hat(s) = exp(-s^2 / 2)
```

with G_R(y) = R^(-1) G(y / R), so that the Fourier multiplier at the torus
wavenumber xi_m = 2 pi m / L is G_hat(R xi_m).

Two dimensions: the top-hat kernel is the normalised indicator of the unit disc
with G_hat(s) = 2 J_1(|s|) / |s|, the Gaussian kernel has
G_hat(s) = exp(-|s|^2 / 2), and G_R(y) = R^(-2) G(y / R).

For the one-dimensional top-hat kernel the admissible range is 0 < R < L / 2,
and it is convenient to set theta = 2 pi R / L, so that R xi_m = m theta.

Implemented in `nmi.kernels`.

## Proposition 1, equation (4)

The transformation (alpha, beta) to (alpha / c, c beta) leaves the density
unchanged, so alpha and beta enter the observation only through the product

```
gamma = alpha beta
```

Verified numerically in `tests/test_scaling_symmetry.py`.

## Equation (5)

With k_0 = 0 the memory equation integrates to k = beta M_mu[u], where
M_mu[u](x, t) = integral from 0 to t of exp(-mu (t - s)) u(x, s) ds, and the
density solves the single nonlocal-in-time equation

```
u_t = d u_xx - gamma ( u ( G_R * M_mu[u] )_x )_x
```

## Equation (6) and Proposition 3

Linearising around the uniform state, each Fourier mode m is not zero evolves
with the mode matrix

```
A_m = [[ -d xi_m^2,  alpha u_bar xi_m^2 G_hat(R xi_m) ],
       [ beta,       -mu                              ]]
```

The uniform state is unstable if and only if kappa u_bar G_hat(R xi_m) > 1 for
some m at least one, where kappa = gamma / (d mu) is the aggregation ratio. The
critical ratio is

```
kappa_c(R) = 1 / ( u_bar max over m at least one of G_hat(R xi_m) )
```

Implemented in `nmi.linear_theory.mode_matrix`, `nmi.linear_theory.is_unstable`
and `nmi.steady_state.critical_kappa`.

## Equation (7) and Theorem 1

Every positive C1 steady state satisfies

```
log u* - kappa ( G_R * u* ) = constant on the torus
```

so that u* is a fixed point of

```
u -> M exp( kappa G_R * u ) / integral of exp( kappa G_R * u )
```

Stationary data determine at most the pair (kappa, R). For the top-hat kernel
the pair is recovered from the modes m = 1 and m = 2 through Lemma 1, and for
the Gaussian kernel from any two modes with distinct wavenumbers.

Implemented in `nmi.steady_state.fixed_point_steady_state`,
`nmi.steady_state.identity_residual` and
`nmi.steady_state.continuation_in_kappa`.

## Theorem 2

For the linearised system with w_m(0) = 0 and v_m(0) not zero, the parameters
(d, mu, gamma, R) are recovered from the modes m = 1 and m = 2 in four steps.

1. The diffusion rate follows from the initial slope,
   d = -v'(0) / (xi^2 v(0)).
2. A mode is a pure exponential if and only if G_hat(R xi_m) = 0, which is read
   off the data.
3. Otherwise v satisfies v'' + p v' + q v = 0 with p = d xi^2 + mu and
   q = xi^2 ( d mu - gamma u_bar G_hat(R xi_m) ), and the pair (p, q) is
   determined by v.
4. The first mode gives mu = p - d xi_1^2, and each mode gives the product
   gamma G_hat(R xi_m). For the top-hat kernel, Lemma 1 applied to
   (gamma / R) sin(m theta) determines theta and hence R and gamma. For the
   Gaussian kernel, the ratio of the two products determines R and then gamma.

Implemented in `nmi.linear_theory.reconstruct_parameters_theorem2`.

## Proposition 4

On a bounded connected Lipschitz domain with the no-flux condition, a weak
steady state with u* at least c > 0 satisfies the same identity, and the
stationary location density is

```
p(x) = exp( kappa G_R * u* ) / integral over Omega of exp( kappa G_R * u* )
```

Implemented in `nmi.wolf.masked_solver.masked_steady_state` and verified on a
bounded interval in `tests/test_bounded_domain_identity.py`. In the wolf
application the aggregation ratio is reported in the dimensionless form
kappa u_bar of Proposition 3, where u_bar is the mean density of the uniform
state on the study area, so that the value is comparable across individuals of
different home range size.

## Corollary 1

Let m_star = ceil( L / (2 R) ). The top-hat transform satisfies
sin(m_star theta) at most zero, so a positive steady state with a nonzero
Fourier coefficient at m_star cannot be a steady state of a Gaussian model.

Implemented in `nmi.linear_theory.m_star`.

## Equation (8)

Each synthetic condition is placed twenty per cent above the onset,

```
kappa = 1.2 kappa_c(R),   gamma = kappa d mu
```

Implemented in `nmi.design.table3_values`.

## Equation (9)

The synthetic observation model is

```
y_ij = u(x_i, t_j; theta) + sigma eps_ij,   eps_ij standard normal,
sigma = eta max over x and t of u
```

Implemented in `nmi.observation` and `nmi.likelihood.gaussian_loglik`.

## Equation (10)

For the location fixes x_1 to x_n of one individual, the weighted
log-likelihood is

```
l_w(kappa, R) = (n_eff / n) sum over i of log p(x_i; kappa, R)
```

Implemented in `nmi.wolf.point_likelihood.weighted_point_loglik`.
