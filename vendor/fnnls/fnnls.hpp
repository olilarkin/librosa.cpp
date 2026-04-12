#pragma once

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <limits>

namespace fnnls {

using ArrayXb = Eigen::Array<bool, Eigen::Dynamic, 1>;
template <typename T>
using MatrixX_ = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
template <typename T>
using VectorX_ = Eigen::Matrix<T, Eigen::Dynamic, 1>;
template <typename T>
using ArrayX_ = Eigen::Array<T, Eigen::Dynamic, 1>;

inline const Eigen::ArrayXi to_indices(const Eigen::Ref<const ArrayXb> &b_array) {
    Eigen::ArrayXi indices = Eigen::ArrayXi::LinSpaced(b_array.size(), 0, b_array.size() - 1);
    const auto shifted_iterator
        = std::stable_partition(indices.data(), indices.data() + indices.size(),
                                [&b_array](Eigen::Index i) { return b_array(i); });
    const size_t data_size = shifted_iterator - indices.data();
    indices.conservativeResize(data_size);
    return indices;
}

inline const ArrayXb operator!(const Eigen::Ref<const ArrayXb> &array_b) {
    return ArrayXb::NullaryExpr(array_b.size(),
                                [&array_b](Eigen::Index i) { return !array_b(i); });
}

template <typename T>
inline Eigen::ArrayXi::Index argmax_of_masked_array(const Eigen::Ref<const VectorX_<T>> &w,
                                                    const Eigen::Ref<const ArrayXb> &P) {
    // The function calculates the argmax of w masked by inverted boolean array P,
    // i.e. argmax(W * ~P).
    Eigen::ArrayXi::Index argmax;
    (ArrayX_<T>::NullaryExpr(P.size(), [&w, &P](Eigen::Index i) {
        if (!P(i)) {
            return w(i);
        }
        constexpr T dzero = T{};
        return dzero;
    })).maxCoeff(&argmax);
    return argmax;
}

template <typename T>
inline VectorX_<T> least_squares_solver(const Eigen::Ref<const MatrixX_<T>> &ZTZ,
                                        const Eigen::Ref<const VectorX_<T>> &ZTx,
                                        const Eigen::Ref<const Eigen::ArrayXi> &indices) {
    return ZTZ(indices, indices)
        .template selfadjointView<Eigen::Upper>()
        .ldlt()
        .solve(ZTx(indices));
}

template <typename T>
inline T get_alpha(const Eigen::Ref<const ArrayX_<T>> &d, const Eigen::Ref<const ArrayX_<T>> &s,
                   const Eigen::Ref<const Eigen::ArrayXi> &indices) {
    // Please note that there is an erroneous extra minus sign in the article
    // on line C2.
    // See instead for example the pseudocode at:
    // https://en.wikipedia.org/wiki/Non-negative_least_squares
    return (d(indices) / (d(indices) - s(indices))).minCoeff();
}

template <typename T>
inline const MatrixX_<T> construct_ZTZ(const Eigen::Ref<const Eigen::SparseMatrix<T>> &ZT_sp,
                                       const int n) {
    MatrixX_<T> ZTZ = MatrixX_<T>::Zero(n, n);
    ZTZ.template triangularView<Eigen::Upper>() = MatrixX_<T>(ZT_sp * ZT_sp.transpose());
    ZTZ.template triangularView<Eigen::Lower>() = ZTZ.transpose();
    return ZTZ;
}

template <typename T>
inline VectorX_<T> fnnls_solver(const Eigen::Map<MatrixX_<T>> &ZT,
                                const Eigen::Map<VectorX_<T>> &x, int max_iterations = 0,
                                T tolerance = -1.0, const T *const precompute_ptr = nullptr) {
    // This implementation is optimised for speed when ZT is intermediately
    // sparse. Please note that Python uses C-style Row-Major storage and Eigen
    // uses Fortran style Col-Major storage. The incoming Z matrix will appear
    // transposed in Eigen. Hence the name ZT to mimic that a Row-Major matrix
    // from Python was sent in.

    const auto n = ZT.rows();

    if (ZT.cols() != x.rows()) {
        throw std::runtime_error(
            "Mismatched sizes of data matrix ZT and target vector x: ZT size = ("
            + std::to_string(n) + ", " + std::to_string(ZT.cols()) + "), x size = "
            + std::to_string(x.rows()) + " (number of rows of ZT should match the size of x)");
    }

    constexpr auto dzero = T{};
    constexpr auto epsilon = std::numeric_limits<T>::epsilon();
    constexpr auto denorm_min = std::numeric_limits<T>::denorm_min();

    max_iterations = max_iterations ? max_iterations : 3 * n;
    tolerance = tolerance > dzero ? tolerance : epsilon * n;

    const Eigen::SparseMatrix<T> ZT_sp = ZT.sparseView();

    const MatrixX_<T> ZTZ = precompute_ptr
                                ? Eigen::Map<MatrixX_<T>>(const_cast<T *>(precompute_ptr), n, n)
                                : construct_ZTZ<T>(ZT_sp, n);

    const VectorX_<T> ZTx = ZT_sp * x;

    constexpr int max_repetitions{5};
    int no_update{};

    ArrayXb P = ArrayXb::Constant(n, false);
    VectorX_<T> d = VectorX_<T>::Zero(n);
    VectorX_<T> s = VectorX_<T>::Zero(n);
    VectorX_<T> w = ZTx;

    for (int iter{};; ++iter) {
        if (P.all() || (w(to_indices(!P)).maxCoeff() < tolerance)) {
            break;
        } else if (iter == max_iterations) {
            throw std::runtime_error(
                "Solution not converged after maximum number of allowed iterations!");
        }
        const ArrayXb P_previous = P;
        P(argmax_of_masked_array<T>(w, P)) = true;
        s(to_indices(P)) = least_squares_solver<T>(ZTZ, ZTx, to_indices(P));

        for (;;) {
            if (!(P.any()) || (s(to_indices(P)).minCoeff() >= dzero)) {
                break;
            }
            // Coefficients that are in the passive set P, but have become
            // non-positive needs adjustment. alpha has to be larger than 0 and
            // smaller than 1. Precision issues where alpha becomes 0 or 1 for a
            // coefficient in the passive set has to be turned off.
            const T alpha = get_alpha<T>(d, s, to_indices(P && (s.array() < dzero)));
            d.noalias() += alpha * (s - d);
            P(to_indices(d.array() <= denorm_min)) = false;
            s(to_indices(P)) = least_squares_solver<T>(ZTZ, ZTx, to_indices(P));
            s(to_indices(!P)).setZero();
        }

        if ((P_previous == P).all()) {
            ++no_update;
        } else {
            no_update = 0;
        }

        if (no_update >= max_repetitions) {
            break;
        }

        d = s;
        w.noalias() = -(ZTZ * d.sparseView());
        w.noalias() += ZTx;
    }

    // Residual is given by
    // const T residual = (x - ZT.transpose() * d).norm();
    return d;
}
} // namespace fnnls
