#pragma once

#include <librosa/types.hpp>

namespace librosa {
namespace internal {

/// 1-D interpolation (linear or cubic spline with not-a-knot BCs)
class Interp1d {
public:
    enum class Kind { Linear, Cubic };

    Interp1d(const ArrayXr& x, const ArrayXr& y, Kind kind = Kind::Cubic)
        : x_(x), y_(y), kind_(kind) {
        Eigen::Index n = x.size();
        if (kind_ == Kind::Cubic && n >= 3) {
            computeCubicCoefficients();
        }
    }

    ArrayXr operator()(const ArrayXr& x_new) const {
        Eigen::Index m = x_new.size();
        ArrayXr result(m);

        for (Eigen::Index i = 0; i < m; ++i) {
            result(i) = evaluate(x_new(i));
        }

        return result;
    }

private:
    ArrayXr x_, y_;
    ArrayXr b_, c_, d_;  // Cubic spline coefficients
    Kind kind_;

    Real evaluate(Real xv) const {
        Eigen::Index n = x_.size();

        // Clamp to range
        if (xv <= x_(0)) return y_(0);
        if (xv >= x_(n - 1)) return y_(n - 1);

        // Binary search for interval
        Eigen::Index lo = 0, hi = n - 1;
        while (hi - lo > 1) {
            Eigen::Index mid = (lo + hi) / 2;
            if (x_(mid) > xv) {
                hi = mid;
            } else {
                lo = mid;
            }
        }

        Real dx = xv - x_(lo);

        if (kind_ == Kind::Linear || n < 3) {
            Real slope = (y_(hi) - y_(lo)) / (x_(hi) - x_(lo));
            return y_(lo) + slope * dx;
        }

        // Cubic: S(x) = y[i] + b[i]*(x-x[i]) + c[i]*(x-x[i])^2 + d[i]*(x-x[i])^3
        return y_(lo) + b_(lo) * dx + c_(lo) * dx * dx + d_(lo) * dx * dx * dx;
    }

    void computeCubicCoefficients() {
        // Natural/not-a-knot cubic spline (matching scipy's default)
        Eigen::Index n = x_.size();
        Eigen::Index nm1 = n - 1;

        // Intervals and divided differences
        ArrayXr h(nm1), delta(nm1);
        for (Eigen::Index i = 0; i < nm1; ++i) {
            h(i) = x_(i + 1) - x_(i);
            delta(i) = (y_(i + 1) - y_(i)) / h(i);
        }

        // Set up tridiagonal system for second derivatives m[i]
        // Interior equations: h[i-1]*m[i-1] + 2*(h[i-1]+h[i])*m[i] + h[i]*m[i+1]
        //                   = 6*(delta[i] - delta[i-1])
        // With not-a-knot BCs: third derivative continuous at x[1] and x[n-2]

        // System size = n
        ArrayXr diag(n), upper(n), lower(n), rhs(n);
        diag.setZero();
        upper.setZero();
        lower.setZero();
        rhs.setZero();

        // Interior rows (i = 1..n-2)
        for (Eigen::Index i = 1; i < nm1; ++i) {
            lower(i) = h(i - 1);
            diag(i) = 2.0 * (h(i - 1) + h(i));
            upper(i) = h(i);
            rhs(i) = 6.0 * (delta(i) - delta(i - 1));
        }

        if (n == 3) {
            // Special case: only one interior point
            // Not-a-knot with 3 points means m[0]=m[1]=m[2] (linear second derivative)
            // Use natural spline (m[0]=0, m[n-1]=0) for 3 points
            diag(0) = 1.0;
            rhs(0) = 0.0;
            diag(n - 1) = 1.0;
            rhs(n - 1) = 0.0;
        } else {
            // Not-a-knot BC at left: third derivative continuous at x[1]
            // d[0] = d[1] => (m[1]-m[0])/h[0] = (m[2]-m[1])/h[1]
            // => h[1]*m[0] - (h[0]+h[1])*m[1] + h[0]*m[2] = 0
            diag(0) = h(1);
            upper(0) = -(h(0) + h(1));
            // upper(0) goes to m[1], but we need to also reference m[2]
            // Rewrite: row 0 is h[1]*m[0] - (h[0]+h[1])*m[1] + h[0]*m[2] = 0
            // This is a 3-term equation, but our tridiagonal only has lower/diag/upper
            // We need to handle this differently.
            // Actually for not-a-knot, we can eliminate using the interior equations.
            // Simpler approach: use the standard not-a-knot formulation

            // Left not-a-knot: h[1]*(m[1]-m[0])/h[0] = h[0]*(m[2]-m[1])/h[1]
            // => h[1]^2*m[0] - (h[0]^2 + h[1]^2 + h[0]*h[1] + ... ) ...
            // Let's use the direct formulation:
            // Row 0: h[1]*m[0] - (h[0]+h[1])*m[1] + h[0]*m[2] = 0
            // This involves m[2], so it's not tridiagonal. We handle it by
            // eliminating m[0] using row 1.

            // Alternative: use the compact formula that modifies row 0 and row n-1
            // Left not-a-knot:
            //   m[0] = m[1] - h[0]*(m[2]-m[1])/h[1]
            //        = m[1]*(1+h[0]/h[1]) - m[2]*h[0]/h[1]
            // Substitute into row 1: h[0]*m[0] + 2*(h[0]+h[1])*m[1] + h[1]*m[2] = rhs[1]
            // h[0]*(m[1]*(1+h[0]/h[1]) - m[2]*h[0]/h[1]) + 2*(h[0]+h[1])*m[1] + h[1]*m[2] = rhs[1]
            // m[1]*(h[0]*(1+h[0]/h[1]) + 2*(h[0]+h[1])) + m[2]*(h[1] - h[0]^2/h[1]) = rhs[1]
            Real r0 = h(0) / h(1);
            diag(0) = 1.0;  // We'll solve for m[0] at the end
            upper(0) = 0.0;
            rhs(0) = 0.0;

            // Modify row 1 to eliminate m[0]
            // m[0] = m[1]*(1+r0) - m[2]*r0
            // Substituting into row 1:
            lower(1) = 0.0;  // m[0] eliminated
            diag(1) = 2.0 * (h(0) + h(1)) + h(0) * (1.0 + r0);
            upper(1) = h(1) - h(0) * r0;
            // rhs(1) stays the same

            // Right not-a-knot:
            // m[n-1] = m[n-2]*(1+h[n-2]/h[n-3]) - m[n-3]*h[n-2]/h[n-3]
            Real rn = h(nm1 - 1) / h(nm1 - 2);
            diag(n - 1) = 1.0;
            lower(n - 1) = 0.0;
            rhs(n - 1) = 0.0;

            // Modify row n-2 to eliminate m[n-1]
            upper(nm1 - 1) = 0.0;  // m[n-1] eliminated
            diag(nm1 - 1) = 2.0 * (h(nm1 - 2) + h(nm1 - 1)) + h(nm1 - 1) * (1.0 + rn);
            lower(nm1 - 1) = h(nm1 - 2) - h(nm1 - 1) * rn;
            // rhs(nm1-1) stays the same
        }

        // Solve tridiagonal system using Thomas algorithm
        ArrayXr m(n);
        // Forward sweep
        ArrayXr cp(n), dp(n);
        cp(0) = upper(0) / diag(0);
        dp(0) = rhs(0) / diag(0);
        for (Eigen::Index i = 1; i < n; ++i) {
            Real w = lower(i) / (diag(i) - lower(i) * cp(i - 1));
            cp(i) = upper(i) / (diag(i) - lower(i) * cp(i - 1));
            dp(i) = (rhs(i) - lower(i) * dp(i - 1)) / (diag(i) - lower(i) * cp(i - 1));
        }

        // Back substitution
        m(n - 1) = dp(n - 1);
        for (Eigen::Index i = n - 2; i >= 0; --i) {
            m(i) = dp(i) - cp(i) * m(i + 1);
        }

        // For not-a-knot with n>3, recover m[0] and m[n-1]
        if (n > 3) {
            Real r0 = h(0) / h(1);
            m(0) = m(1) * (1.0 + r0) - m(2) * r0;

            Real rn = h(nm1 - 1) / h(nm1 - 2);
            m(n - 1) = m(n - 2) * (1.0 + rn) - m(n - 3) * rn;
        }

        // Compute coefficients
        b_.resize(nm1);
        c_.resize(nm1);
        d_.resize(nm1);

        for (Eigen::Index i = 0; i < nm1; ++i) {
            c_(i) = m(i) / 2.0;
            d_(i) = (m(i + 1) - m(i)) / (6.0 * h(i));
            b_(i) = delta(i) - h(i) * (2.0 * m(i) + m(i + 1)) / 6.0;
        }
    }
};

} // namespace internal
} // namespace librosa
