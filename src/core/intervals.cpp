#include <librosa/core/intervals.hpp>
#include <librosa/util/exceptions.hpp>
#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>
#include <set>
#include <tuple>
#include <vector>

namespace librosa {

// ============================================================================
// Pythagorean Intervals
// ============================================================================

ArrayXr pythagorean_intervals(int bins_per_octave, bool sort) {
    const double log2_3 = std::log2(3.0);

    ArrayXr log_ratios(bins_per_octave);

    for (int i = 0; i < bins_per_octave; ++i) {
        // modf(i * log2(3)) gives fractional part
        double int_part;
        double frac = std::modf(i * log2_3, &int_part);
        if (frac < 0.0) {
            frac += 1.0;
        }
        log_ratios(i) = frac;
    }

    if (sort) {
        // Sort log_ratios in ascending order
        std::vector<Real> vals(log_ratios.data(), log_ratios.data() + log_ratios.size());
        std::sort(vals.begin(), vals.end());
        for (int i = 0; i < bins_per_octave; ++i) {
            log_ratios(i) = vals[i];
        }
    }

    // Convert from log2 to ratios
    ArrayXr result(bins_per_octave);
    for (int i = 0; i < bins_per_octave; ++i) {
        result(i) = std::pow(2.0, log_ratios(i));
    }

    return result;
}

// ============================================================================
// P-limit Intervals (Crystal Growth)
// ============================================================================

namespace {

using PrimePowers = std::vector<int>;

double harmonic_distance(const std::vector<double>& logs,
                         const PrimePowers& a,
                         const PrimePowers& b) {
    // HD = log2(ab / gcd(a,b)^2)
    size_t n = logs.size();
    double result = 0.0;
    for (size_t i = 0; i < n; ++i) {
        int a_num = std::max(a[i], 0);
        int a_den = a_num - a[i];
        int b_num = std::max(b[i], 0);
        int b_den = b_num - b[i];
        int gcd = std::min(a_num, b_num) - std::max(a_den, b_den);
        result += logs[i] * (a[i] + b[i] - 2 * gcd);
    }
    return std::round(result * 1e6) / 1e6;
}

double crystal_tie_break_score(const PrimePowers& a, const std::vector<double>& logs) {
    double result = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        result += logs[i] * std::abs(a[i]);
    }
    return result;
}

} // anonymous namespace

ArrayXr plimit_intervals(const std::vector<int>& primes,
                         int bins_per_octave,
                         bool sort) {
    size_t n_primes = primes.size();
    std::vector<double> logs(n_primes);
    for (size_t i = 0; i < n_primes; ++i) {
        logs[i] = std::log2(static_cast<double>(primes[i]));
    }

    // Seed set: each prime and its inverse
    std::vector<PrimePowers> seeds;
    for (size_t i = 0; i < n_primes; ++i) {
        PrimePowers pos(n_primes, 0);
        pos[i] = 1;
        seeds.push_back(pos);
        PrimePowers neg(n_primes, 0);
        neg[i] = -1;
        seeds.push_back(neg);
    }

    // Distance cache
    std::map<std::pair<PrimePowers, PrimePowers>, double> distances;

    // Initialize with root
    PrimePowers root(n_primes, 0);
    std::vector<PrimePowers> intervals;
    intervals.push_back(root);

    // Frontier
    std::vector<PrimePowers> frontier = seeds;

    // Set for quick membership checks
    auto to_set_key = [](const PrimePowers& p) { return p; };
    std::set<PrimePowers> interval_set;
    interval_set.insert(root);
    std::set<PrimePowers> frontier_set;
    for (const auto& s : frontier) frontier_set.insert(s);

    while (static_cast<int>(intervals.size()) < bins_per_octave) {
        double best_score = std::numeric_limits<double>::infinity();
        size_t best_f = 0;

        for (size_t f = 0; f < frontier.size(); ++f) {
            double HD = 0.0;
            for (const auto& s : intervals) {
                auto key = std::make_pair(s, frontier[f]);
                auto it = distances.find(key);
                if (it == distances.end()) {
                    double d = harmonic_distance(logs, frontier[f], s);
                    distances[key] = d;
                    distances[std::make_pair(frontier[f], s)] = d;
                    HD += d;
                } else {
                    HD += it->second;
                }
            }

            if (HD < best_score ||
                (std::abs(HD - best_score) < 1e-9 &&
                 crystal_tie_break_score(frontier[f], logs) <
                 crystal_tie_break_score(frontier[best_f], logs))) {
                best_score = HD;
                best_f = f;
            }
        }

        PrimePowers new_point = frontier[best_f];
        frontier.erase(frontier.begin() + best_f);
        frontier_set.erase(new_point);
        intervals.push_back(new_point);
        interval_set.insert(new_point);

        // Expand frontier
        for (const auto& seed : seeds) {
            PrimePowers new_seed(n_primes);
            for (size_t i = 0; i < n_primes; ++i) {
                new_seed[i] = new_point[i] + seed[i];
            }
            if (interval_set.find(new_seed) == interval_set.end() &&
                frontier_set.find(new_seed) == frontier_set.end()) {
                frontier.push_back(new_seed);
                frontier_set.insert(new_seed);
            }
        }
    }

    // Convert prime powers to log2 ratios in [0, 1)
    ArrayXr log_ratios(bins_per_octave);
    for (int i = 0; i < bins_per_octave; ++i) {
        double lr = 0.0;
        for (size_t p = 0; p < n_primes; ++p) {
            lr += intervals[i][p] * logs[p];
        }
        // modf to get fractional part
        double int_part;
        double frac = std::modf(lr, &int_part);
        if (frac < 0.0) {
            frac += 1.0;
        }
        log_ratios(i) = frac;
    }

    if (sort) {
        std::vector<Real> vals(log_ratios.data(), log_ratios.data() + log_ratios.size());
        std::sort(vals.begin(), vals.end());
        for (int i = 0; i < bins_per_octave; ++i) {
            log_ratios(i) = vals[i];
        }
    }

    ArrayXr result(bins_per_octave);
    for (int i = 0; i < bins_per_octave; ++i) {
        result(i) = std::pow(2.0, log_ratios(i));
    }

    return result;
}

// ============================================================================
// Interval Frequencies
// ============================================================================

ArrayXr interval_frequencies(int n_bins,
                             Real fmin,
                             const std::string& intervals,
                             int bins_per_octave,
                             Real tuning,
                             bool sort) {
    ArrayXr ratios;

    if (intervals == "equal") {
        ratios.resize(bins_per_octave);
        for (int i = 0; i < bins_per_octave; ++i) {
            ratios(i) = std::pow(2.0, (tuning + i) / bins_per_octave);
        }
    } else if (intervals == "pythagorean") {
        ratios = pythagorean_intervals(bins_per_octave, sort);
    } else if (intervals == "ji3") {
        ratios = plimit_intervals({3}, bins_per_octave, sort);
    } else if (intervals == "ji5") {
        ratios = plimit_intervals({3, 5}, bins_per_octave, sort);
    } else if (intervals == "ji7") {
        ratios = plimit_intervals({3, 5, 7}, bins_per_octave, sort);
    } else {
        throw ParameterError("Unknown interval type: " + intervals);
    }

    // Tile ratios across octaves
    int n_octaves = static_cast<int>(std::ceil(static_cast<Real>(n_bins) / bins_per_octave));

    ArrayXr all_ratios(n_octaves * bins_per_octave);
    for (int oct = 0; oct < n_octaves; ++oct) {
        Real oct_mult = std::pow(2.0, oct);
        for (int i = 0; i < bins_per_octave; ++i) {
            all_ratios(oct * bins_per_octave + i) = oct_mult * ratios(i);
        }
    }

    // Trim to n_bins
    ArrayXr result = all_ratios.head(n_bins);

    if (sort) {
        std::vector<Real> vals(result.data(), result.data() + result.size());
        std::sort(vals.begin(), vals.end());
        for (int i = 0; i < n_bins; ++i) {
            result(i) = vals[i];
        }
    }

    return result * fmin;
}

ArrayXr interval_frequencies(int n_bins,
                             Real fmin,
                             const ArrayXr& intervals,
                             bool sort) {
    int bins_per_octave = static_cast<int>(intervals.size());
    int n_octaves = static_cast<int>(std::ceil(static_cast<Real>(n_bins) / bins_per_octave));

    ArrayXr all_ratios(n_octaves * bins_per_octave);
    for (int oct = 0; oct < n_octaves; ++oct) {
        Real oct_mult = std::pow(2.0, oct);
        for (int i = 0; i < bins_per_octave; ++i) {
            all_ratios(oct * bins_per_octave + i) = oct_mult * intervals(i);
        }
    }

    ArrayXr result = all_ratios.head(n_bins);

    if (sort) {
        std::vector<Real> vals(result.data(), result.data() + result.size());
        std::sort(vals.begin(), vals.end());
        for (int i = 0; i < n_bins; ++i) {
            result(i) = vals[i];
        }
    }

    return result * fmin;
}

} // namespace librosa
