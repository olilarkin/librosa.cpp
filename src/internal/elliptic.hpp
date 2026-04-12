// Elliptic integral and Jacobi elliptic function implementations
// for elliptic (Cauer) IIR filter design.
//
// Algorithms ported from:
// - scipy.signal._filter_design: ellipap, _ellipdeg, _arc_jac_sn, _arc_jac_sc1
// - scipy.special: ellipk, ellipj
// - Orfanidis, "Lecture Notes on Elliptic Filter Design"
//
// The AGM-based elliptic integral K(m) follows the same algorithm as
// Boost.Math (BSL-1.0 license) and scipy.special.ellipk.

#pragma once

#include <cmath>
#include <complex>
#include <limits>
#include <vector>

namespace librosa {
namespace internal {
namespace elliptic {

using Real = double;
using Complex = std::complex<Real>;

/// Complete elliptic integral of the first kind K(m) via AGM.
/// m is the parameter (not the modulus k; m = k^2).
inline Real ellipk(Real m) {
    if (m < 0.0 || m >= 1.0) {
        if (m == 1.0) return std::numeric_limits<Real>::infinity();
        return std::numeric_limits<Real>::quiet_NaN();
    }
    Real a = 1.0;
    Real b = std::sqrt(1.0 - m);
    for (int i = 0; i < 100; ++i) {
        Real a_new = 0.5 * (a + b);
        Real b_new = std::sqrt(a * b);
        if (std::abs(a_new - b_new) < 1e-15 * a_new) {
            a = a_new;
            break;
        }
        a = a_new;
        b = b_new;
    }
    return M_PI / (2.0 * a);
}

/// K(1-m): complete elliptic integral with complementary parameter.
/// scipy's ellipkm1(p) = K(1-p).
inline Real ellipkm1(Real p) {
    return ellipk(1.0 - p);
}

/// Jacobi elliptic functions sn(u, m), cn(u, m), dn(u, m).
struct JacobiResult {
    Real sn, cn, dn;
};

inline JacobiResult ellipj(Real u, Real m) {
    if (m < 1e-15) {
        return {std::sin(u), std::cos(u), 1.0};
    }
    if (m > 1.0 - 1e-15) {
        Real s = std::tanh(u);
        Real c = 1.0 / std::cosh(u);
        return {s, c, c};
    }

    // AGM algorithm from Abramowitz & Stegun 16.4
    // Compute AGM sequence: a, b, c
    constexpr int MAX_ITER = 50;
    Real a[MAX_ITER + 1], b_arr[MAX_ITER + 1], c[MAX_ITER + 1];
    int n = 0;
    a[0] = 1.0;
    b_arr[0] = std::sqrt(1.0 - m);
    c[0] = std::sqrt(m);

    while (std::abs(c[n]) > 1e-15 && n < MAX_ITER) {
        a[n + 1] = 0.5 * (a[n] + b_arr[n]);
        b_arr[n + 1] = std::sqrt(a[n] * b_arr[n]);
        c[n + 1] = 0.5 * (a[n] - b_arr[n]);
        n++;
    }

    // Compute phi_n = 2^n * a_n * u
    Real phi = std::ldexp(a[n] * u, n);  // 2^n * a_n * u

    // Back-substitute to find phi_0
    for (int i = n; i >= 1; --i) {
        // phi_{i-1} = 0.5*(phi_i + arcsin(c_i/a_i * sin(phi_i)))
        phi = 0.5 * (phi + std::asin(c[i] / a[i] * std::sin(phi)));
    }

    Real sn_val = std::sin(phi);
    Real cn_val = std::cos(phi);
    Real dn_val = cn_val == 0.0 ? 1.0 : std::sqrt(1.0 - m * sn_val * sn_val);

    return {sn_val, cn_val, dn_val};
}

/// Inverse Jacobi sn for complex argument: find z such that sn(z, m) = w.
/// Port of scipy's _arc_jac_sn using descending Landen transformation.
inline Complex arc_jac_sn(Complex w, Real m) {
    Real k = std::sqrt(m);
    if (k >= 1.0) return std::atanh(w);
    if (k < 1e-15) return std::asin(w);

    // Build the Landen sequence of moduli
    std::vector<Real> ks;
    ks.push_back(k);
    for (int i = 0; i < 50; ++i) {
        Real k_ = ks.back();
        Real kp = std::sqrt((1.0 - k_) * (1.0 + k_));
        Real k_next = (1.0 - kp) / (1.0 + kp);
        ks.push_back(k_next);
        if (k_next == 0.0) break;
    }

    // Compute K from the Landen sequence: K = pi/2 * prod(1 + k_i) for i >= 1
    Real K_val = M_PI / 2.0;
    for (size_t i = 1; i < ks.size(); ++i) {
        K_val *= (1.0 + ks[i]);
    }

    // Forward-transform w through the Landen sequence
    Complex wn = w;
    for (size_t i = 0; i + 1 < ks.size(); ++i) {
        Real kn = ks[i];
        Real knext = ks[i + 1];
        // complement: sqrt((1 - kn*wn)*(1 + kn*wn))
        Complex kw = Complex(kn, 0.0) * wn;
        Complex comp = std::sqrt((Complex(1.0, 0.0) - kw) * (Complex(1.0, 0.0) + kw));
        wn = 2.0 * wn / ((1.0 + knext) * (1.0 + comp));
    }

    // u = 2/pi * arcsin(w_final), z = K * u
    Complex u = (2.0 / M_PI) * std::asin(wn);
    return K_val * u;
}

/// Real inverse Jacobian sc with complementary modulus.
/// Solve for z in w = sc(z, 1-m), where sc(z, m) = -i * sn(i*z, 1-m).
/// Port of scipy's _arc_jac_sc1.
inline Real arc_jac_sc1(Real w, Real m) {
    Complex z = arc_jac_sn(Complex(0.0, w), m);
    // z should be purely imaginary
    return z.imag();
}

/// Solve the degree equation for elliptic filter design.
/// Given N and m1, find m such that N * K(m)/K'(m) = K(m1)/K'(m1).
/// Uses nome-based approach from scipy's _ellipdeg (Orfanidis Eq. 49).
inline Real ellipdeg(int N, Real m1) {
    if (m1 < 1e-20) return 0.0;
    if (m1 > 1.0 - 1e-20) return 1.0;

    Real K1 = ellipk(m1);
    Real K1p = ellipk(1.0 - m1);
    Real q1 = std::exp(-M_PI * K1p / K1);
    Real q = std::pow(q1, 1.0 / N);

    // scipy uses MMAX=7: num = sum_{n=0}^{7} q^{n(n+1)}, den = 1 + 2*sum_{n=1}^{8} q^{n^2}
    constexpr int MMAX = 7;
    Real num = 0.0;
    for (int n = 0; n <= MMAX; ++n) {
        num += std::pow(q, static_cast<Real>(n) * (n + 1));
    }
    Real den = 1.0;
    for (int n = 1; n <= MMAX + 1; ++n) {
        den += 2.0 * std::pow(q, static_cast<Real>(n) * n);
    }

    Real result = 16.0 * q * std::pow(num / den, 4);
    if (result < 0.0) result = 0.0;
    if (result > 1.0) result = 1.0;
    return result;
}

} // namespace elliptic
} // namespace internal
} // namespace librosa
