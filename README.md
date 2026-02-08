# Monte Carlo Library

A Monte Carlo derivatives pricing engine supporting stochastic models, variance reduction, Greeks estimation, implied volatility, and performance optimization.

## Features

- Monte Carlo option pricing
- Geometric Brownian Motion (GBM)
- Heston stochastic volatility model
- European, Asian, and American options
- Variance reduction:
  - Antithetic variates
  - Control variates
- Option Greeks:
  - Black-Scholes analytical Greeks
  - Monte Carlo finite difference Greeks
  - Pathwise Delta
- Implied volatility solver
- Monte Carlo convergence analysis
- Numba accelerated simulation kernels

## Validation

Greeks are validated against Black-Scholes analytical values.

Example:

| Greek | Monte Carlo | Black-Scholes | Error |
|---|---:|---:|---:|
| Delta | 0.635997 | 0.636831 | 0.13% |
| Gamma | 0.019004 | 0.018762 | 1.29% |
| Vega | 37.374 | 37.524 | 0.40% |
| Theta | -6.407 | -6.414 | 0.11% |

## Installation

```bash
pip install .
```

## Testing

```bash
pytest
```