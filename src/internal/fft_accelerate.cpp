#include "fft.hpp"

#include <librosa/util/exceptions.hpp>

#include <Accelerate/Accelerate.h>

#include <cmath>
#include <complex>
#include <cstdint>
#include <vector>

// vDSP_DFT only supports lengths of the form f * 2^n with f in {1, 3, 5, 15}.
// For the common STFT power-of-two sizes we use the native zrop/zop path.
// For arbitrary lengths (e.g. fft-based resample at sr=22050) we fall back to
// Bluestein's algorithm, which expresses an N-point DFT as a length-M complex
// convolution where M >= 2N-1 is chosen as a supported length (next power of
// two).
//
// Normalization:
//   * Native real forward (zrop) produces 2x FFTW. We halve on output.
//   * Native real inverse (zrop) expects 2x input and returns N * IDFT;
//     because the caller passes FFTW-format 1x input, we double on output.
//   * Complex DFT (zop) matches FFTW unscaled directly.
//   * The Bluestein path computes unscaled DFT (FFTW convention) directly.

namespace librosa {
namespace internal {

namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;

vDSP_DFT_SetupD try_zop_setup(vDSP_Length n, vDSP_DFT_Direction dir,
                              vDSP_DFT_SetupD previous = nullptr) {
    return vDSP_DFT_zop_CreateSetupD(previous, n, dir);
}

int next_bluestein_length(int n) {
    // Need M >= 2N-1, M supported by zop. Power of 2 is always supported.
    int target = 2 * n - 1;
    int m = 1;
    while (m < target) m <<= 1;
    return m;
}

// Per-length Bluestein state. Computes X[k] = sum_{j=0}^{N-1} x[j] e^{-i 2pi jk/N}
// (forward) or the inverse (unscaled), for arbitrary N.
class Bluestein {
public:
    Bluestein(int n, bool forward) : n_(n), forward_(forward) {
        m_ = next_bluestein_length(n);

        chirp_re_.resize(n);
        chirp_im_.resize(n);
        const double sign = forward_ ? -1.0 : 1.0;
        for (int k = 0; k < n; ++k) {
            // Using (k*k) mod (2N) avoids catastrophic cancellation when k^2 is
            // large — the modulus leaves the complex exponential unchanged.
            const double phase = sign * kPi * static_cast<double>(
                (static_cast<int64_t>(k) * k) % (2 * n)) / n;
            chirp_re_[k] = std::cos(phase);
            chirp_im_[k] = std::sin(phase);
        }

        // Build kernel b'[n] of length M: b'[0] = 1, b'[j] = e^{-sign i pi j^2/N}
        // for 1 <= j < N, and mirrored at the tail (b'[M-j] = b'[j]).
        std::vector<double> kernel_re(m_, 0.0);
        std::vector<double> kernel_im(m_, 0.0);
        for (int j = 0; j < n; ++j) {
            const double phase = -sign * kPi * static_cast<double>(
                (static_cast<int64_t>(j) * j) % (2 * n)) / n;
            const double cr = std::cos(phase);
            const double ci = std::sin(phase);
            kernel_re[j] = cr;
            kernel_im[j] = ci;
            if (j > 0) {
                kernel_re[m_ - j] = cr;
                kernel_im[m_ - j] = ci;
            }
        }

        fft_forward_ = try_zop_setup(static_cast<vDSP_Length>(m_), vDSP_DFT_FORWARD);
        fft_inverse_ = try_zop_setup(static_cast<vDSP_Length>(m_), vDSP_DFT_INVERSE, fft_forward_);
        if (!fft_forward_ || !fft_inverse_) {
            throw ParameterError("Accelerate FFT backend failed to create Bluestein setup");
        }

        kernel_fft_re_.resize(m_);
        kernel_fft_im_.resize(m_);
        vDSP_DFT_ExecuteD(fft_forward_,
                          kernel_re.data(), kernel_im.data(),
                          kernel_fft_re_.data(), kernel_fft_im_.data());

        a_re_.resize(m_);
        a_im_.resize(m_);
        c_re_.resize(m_);
        c_im_.resize(m_);
    }

    ~Bluestein() {
        if (fft_inverse_) vDSP_DFT_DestroySetupD(fft_inverse_);
        if (fft_forward_) vDSP_DFT_DestroySetupD(fft_forward_);
    }

    // Compute y[k] for k = 0..n-1 from complex input x[j], j = 0..n-1.
    // Output length is n. For forward, matches FFTW unscaled DFT; for inverse,
    // matches FFTW unscaled IDFT (i.e. sum_k X[k] e^{+2pi i jk/N}).
    void execute(const std::complex<double>* in, std::complex<double>* out) {
        // a[j] = x[j] * chirp[j] for j < n, else 0.
        for (int j = 0; j < n_; ++j) {
            const double xr = in[j].real();
            const double xi = in[j].imag();
            a_re_[j] = xr * chirp_re_[j] - xi * chirp_im_[j];
            a_im_[j] = xr * chirp_im_[j] + xi * chirp_re_[j];
        }
        for (int j = n_; j < m_; ++j) {
            a_re_[j] = 0.0;
            a_im_[j] = 0.0;
        }

        // A = FFT(a)
        vDSP_DFT_ExecuteD(fft_forward_,
                          a_re_.data(), a_im_.data(),
                          c_re_.data(), c_im_.data());

        // A .* kernel_fft, in place in a_re_/a_im_.
        for (int k = 0; k < m_; ++k) {
            const double ar = c_re_[k], ai = c_im_[k];
            const double br = kernel_fft_re_[k], bi = kernel_fft_im_[k];
            a_re_[k] = ar * br - ai * bi;
            a_im_[k] = ar * bi + ai * br;
        }

        // conv = IFFT(a .* kernel_fft). vDSP inverse scales by M, so divide.
        vDSP_DFT_ExecuteD(fft_inverse_,
                          a_re_.data(), a_im_.data(),
                          c_re_.data(), c_im_.data());

        const double inv_m = 1.0 / m_;
        for (int k = 0; k < n_; ++k) {
            const double cr = c_re_[k] * inv_m;
            const double ci = c_im_[k] * inv_m;
            out[k] = std::complex<double>(cr * chirp_re_[k] - ci * chirp_im_[k],
                                          cr * chirp_im_[k] + ci * chirp_re_[k]);
        }
    }

private:
    int n_;
    int m_;
    bool forward_;
    vDSP_DFT_SetupD fft_forward_ = nullptr;
    vDSP_DFT_SetupD fft_inverse_ = nullptr;
    std::vector<double> chirp_re_, chirp_im_;
    std::vector<double> kernel_fft_re_, kernel_fft_im_;
    std::vector<double> a_re_, a_im_;
    std::vector<double> c_re_, c_im_;
};

} // namespace

struct RealFft::Impl {
    int n_fft;
    int n_half;

    // Native zrop path (fast). Non-null iff both setups exist.
    vDSP_DFT_SetupD forward_setup = nullptr;
    vDSP_DFT_SetupD inverse_setup = nullptr;
    std::vector<double> even_in;
    std::vector<double> odd_in;
    std::vector<double> spec_re;
    std::vector<double> spec_im;

    // Bluestein fallback (general sizes).
    std::unique_ptr<Bluestein> bluestein_forward;
    std::unique_ptr<Bluestein> bluestein_inverse;
    std::vector<std::complex<double>> bl_in;
    std::vector<std::complex<double>> bl_out;

    explicit Impl(int n) : n_fft(n), n_half(n / 2) {
        if (n <= 0) {
            throw ParameterError("RealFft requires positive n_fft");
        }
        // Native zrop requires even N and a supported length. Odd N must go
        // straight to Bluestein.
        if ((n & 1) == 0) {
            forward_setup = vDSP_DFT_zrop_CreateSetupD(nullptr, static_cast<vDSP_Length>(n),
                                                       vDSP_DFT_FORWARD);
            if (forward_setup) {
                inverse_setup = vDSP_DFT_zrop_CreateSetupD(forward_setup,
                                                           static_cast<vDSP_Length>(n),
                                                           vDSP_DFT_INVERSE);
            }
        }
        if (forward_setup && inverse_setup) {
            even_in.resize(n_half);
            odd_in.resize(n_half);
            spec_re.resize(n_half);
            spec_im.resize(n_half);
        } else {
            if (forward_setup) vDSP_DFT_DestroySetupD(forward_setup);
            if (inverse_setup) vDSP_DFT_DestroySetupD(inverse_setup);
            forward_setup = nullptr;
            inverse_setup = nullptr;
            bluestein_forward = std::make_unique<Bluestein>(n, /*forward=*/true);
            bluestein_inverse = std::make_unique<Bluestein>(n, /*forward=*/false);
            bl_in.resize(n);
            bl_out.resize(n);
        }
    }

    ~Impl() {
        if (inverse_setup) vDSP_DFT_DestroySetupD(inverse_setup);
        if (forward_setup) vDSP_DFT_DestroySetupD(forward_setup);
    }

    bool uses_native() const { return forward_setup != nullptr; }
};

RealFft::RealFft(int n_fft)
    : impl_(std::make_unique<Impl>(n_fft)), n_fft_(n_fft) {}

RealFft::~RealFft() = default;
RealFft::RealFft(RealFft&&) noexcept = default;
RealFft& RealFft::operator=(RealFft&&) noexcept = default;

void RealFft::forward(const Real* in, Complex* out) {
    const int H = impl_->n_half;

    if (impl_->uses_native()) {
        for (int i = 0; i < H; ++i) {
            impl_->even_in[i] = in[2 * i];
            impl_->odd_in[i]  = in[2 * i + 1];
        }

        vDSP_DFT_ExecuteD(impl_->forward_setup,
                          impl_->even_in.data(), impl_->odd_in.data(),
                          impl_->spec_re.data(), impl_->spec_im.data());

        out[0] = Complex(impl_->spec_re[0] * 0.5, 0.0);
        out[H] = Complex(impl_->spec_im[0] * 0.5, 0.0);
        for (int k = 1; k < H; ++k) {
            out[k] = Complex(impl_->spec_re[k] * 0.5, impl_->spec_im[k] * 0.5);
        }
        return;
    }

    // Bluestein fallback: treat real input as complex, compute N-point DFT,
    // keep first N/2+1 bins.
    const int N = impl_->n_fft;
    for (int i = 0; i < N; ++i) {
        impl_->bl_in[i] = std::complex<double>(in[i], 0.0);
    }
    impl_->bluestein_forward->execute(impl_->bl_in.data(), impl_->bl_out.data());
    for (int k = 0; k <= H; ++k) {
        out[k] = impl_->bl_out[k];
    }
}

void RealFft::inverse(const Complex* in, Real* out) {
    const int H = impl_->n_half;

    if (impl_->uses_native()) {
        impl_->spec_re[0] = in[0].real();
        impl_->spec_im[0] = in[H].real();
        for (int k = 1; k < H; ++k) {
            impl_->spec_re[k] = in[k].real();
            impl_->spec_im[k] = in[k].imag();
        }

        vDSP_DFT_ExecuteD(impl_->inverse_setup,
                          impl_->spec_re.data(), impl_->spec_im.data(),
                          impl_->even_in.data(), impl_->odd_in.data());

        // vDSP's roundtrip yields 2N * x given its 2x packed forward output;
        // feeding FFTW-format (1x) input already halves that to N * x,
        // which matches FFTW's unscaled inverse convention.
        for (int i = 0; i < H; ++i) {
            out[2 * i]     = impl_->even_in[i];
            out[2 * i + 1] = impl_->odd_in[i];
        }
        return;
    }

    // Bluestein inverse: expand input to full N via conjugate symmetry, run
    // unscaled IDFT, take real parts.
    const int N = impl_->n_fft;
    impl_->bl_in[0] = in[0];
    for (int k = 1; k <= H; ++k) {
        impl_->bl_in[k] = in[k];
        if (k < N - k) {
            impl_->bl_in[N - k] = std::conj(in[k]);
        }
    }
    impl_->bluestein_inverse->execute(impl_->bl_in.data(), impl_->bl_out.data());
    for (int i = 0; i < N; ++i) {
        out[i] = impl_->bl_out[i].real();
    }
}

struct ComplexFft::Impl {
    int n_fft;
    vDSP_DFT_SetupD forward_setup = nullptr;
    std::vector<double> in_re;
    std::vector<double> in_im;
    std::vector<double> out_re;
    std::vector<double> out_im;

    std::unique_ptr<Bluestein> bluestein;
    std::vector<std::complex<double>> bl_in;
    std::vector<std::complex<double>> bl_out;

    explicit Impl(int n) : n_fft(n) {
        forward_setup = vDSP_DFT_zop_CreateSetupD(nullptr, static_cast<vDSP_Length>(n),
                                                  vDSP_DFT_FORWARD);
        if (forward_setup) {
            in_re.resize(n); in_im.resize(n);
            out_re.resize(n); out_im.resize(n);
        } else {
            bluestein = std::make_unique<Bluestein>(n, /*forward=*/true);
            bl_in.resize(n); bl_out.resize(n);
        }
    }

    ~Impl() {
        if (forward_setup) vDSP_DFT_DestroySetupD(forward_setup);
    }

    bool uses_native() const { return forward_setup != nullptr; }
};

ComplexFft::ComplexFft(int n_fft)
    : impl_(std::make_unique<Impl>(n_fft)), n_fft_(n_fft) {}

ComplexFft::~ComplexFft() = default;
ComplexFft::ComplexFft(ComplexFft&&) noexcept = default;
ComplexFft& ComplexFft::operator=(ComplexFft&&) noexcept = default;

void ComplexFft::forward(const Complex* in, Complex* out) {
    const int N = impl_->n_fft;
    if (impl_->uses_native()) {
        for (int i = 0; i < N; ++i) {
            impl_->in_re[i] = in[i].real();
            impl_->in_im[i] = in[i].imag();
        }
        vDSP_DFT_ExecuteD(impl_->forward_setup,
                          impl_->in_re.data(), impl_->in_im.data(),
                          impl_->out_re.data(), impl_->out_im.data());
        for (int i = 0; i < N; ++i) {
            out[i] = Complex(impl_->out_re[i], impl_->out_im[i]);
        }
        return;
    }
    impl_->bluestein->execute(in, out);
}

} // namespace internal
} // namespace librosa
