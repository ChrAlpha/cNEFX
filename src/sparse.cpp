#include "cnefx/core.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace cnefx {
Selection Selection::parse(std::span<const std::uint8_t> bytes) {
  if (bytes.size() != 28 || bytes[0] != '0' || bytes[1] != '1' ||
      bytes[2] != '0' || bytes[3] != '0')
    throw std::invalid_argument(
        "Unsupported pixel-shift selector version or length.");
  const auto u16 = [&](unsigned p) {
    return int(bytes[p]) | (int(bytes[p + 1]) << 8);
  };
  return {u16(6),    u16(8),    u16(10),   u16(12),   u16(14),  u16(16),
          bytes[18], bytes[19], bytes[20], bytes[22], bytes[24]};
}

std::vector<std::pair<int, int>> Selection::coordinates(int width,
                                                        int height) const {
  const std::array<int, 11> fields{
      reference_height, reference_width, half_grid_start, first_row,
      middle_step,      outer_step,      inner_step,      middle_count,
      outer_count,      inner_before,    inner_after};
  if (std::any_of(fields.begin(), fields.end(),
                  [](int x) { return x < 0 || x > 65535; }) ||
      width < 8 || height < 8 || width > 65535 || height > 65535)
    throw std::invalid_argument("Unsupported sparse selector geometry.");
  const std::int64_t visits = std::int64_t(outer_count) * middle_count *
                              (inner_before + inner_after + 1);
  if (visits > 4096 || visits * (width / 2 - 2) > 1000000)
    throw std::invalid_argument(
        "Sparse selector exceeds the supported work budget.");
  int row = first_row - (reference_height - height) / 2;
  const int lower = (reference_width - width) / 4;
  std::vector<std::pair<int, int>> coordinates;
  for (int outer = 0; outer < outer_count; ++outer) {
    if (outer)
      row += outer_step;
    for (int middle = 0; middle < middle_count; ++middle) {
      if (middle)
        row += middle_step;
      for (int inner = 0; inner <= inner_before + inner_after; ++inner) {
        if (inner)
          row += inner_step;
        if (row < 0 || row > height)
          continue;
        int column = 1;
        for (int half = 1; half < width / 2 - 1; ++half) {
          if (half < half_grid_start / 2 || half < lower ||
              half > lower + width / 2)
            continue;
          if (half != 1)
            column += 2;
          if (row < 2 || row >= height - 2 || column < 1 || column >= width - 1)
            throw std::invalid_argument("Unsafe vendor neighborhood.");
          coordinates.emplace_back(row, column);
        }
      }
    }
  }
  return coordinates;
}

namespace {
double divide(double a, double b) {
  return b != 0   ? a / b
         : a == 0 ? std::numeric_limits<double>::quiet_NaN()
                  : std::numeric_limits<double>::infinity();
}
Sample store(double value) {
  if (std::isnan(value))
    return 0;
  return value <= 16383 ? static_cast<Sample>(value) : 16383;
}
bool le(double a, double b) { return !(a > b); }

Sample repair_value(unsigned frame, Sample current, double mean,
                    Sample diagonal, Sample guard, double r, double s, double t,
                    double estimate) {
  Sample output = current;
  const auto blend = [&](double weight, double other) {
    return store(current * other + estimate * weight);
  };
  if (le(s, r + 0.1)) {
    if (r < s) {
      if (t + 0.1 < r)
        output = blend(0.7, 0.3);
    } else if (t + 0.1 < s)
      output = blend(0.3, 0.7);
  } else if (le(t, r + 0.1) && le(r, t + 0.1))
    output = blend(0.5, 0.5);
  const auto high = [&]() -> Sample {
    if (le(t, r + 0.3))
      return divide(current, mean) < divide(diagonal, mean) ? blend(0.5, 0.5)
                                                            : current;
    if (le(mean, 2000) ||
        le(divide(current, mean) + 0.4, divide(diagonal, mean)))
      return current;
    if (le(estimate, current + 500) || estimate + 500 < mean)
      return store(estimate + 0.0);
    return store(current + estimate * 0.0);
  };
  const auto low = [&]() {
    if (le(1.1, r))
      return blend(0.5, 0.5);
    if (estimate + 2000 < guard || estimate + 300 < diagonal)
      return store(current + estimate * 0.0);
    return store(estimate + 0.0);
  };
  const auto close = [&]() {
    return store(le(t, r) ? estimate + 0.0 : current + estimate * 0.0);
  };
  if (le(s, t + 0.1)) {
    if (le(t, s + 0.1)) {
      if (s < t && t < r + 0.1)
        output = blend(0.5, 0.5);
    } else
      output = high();
  } else if (frame == 0) {
    if (le(r, t + 0.1))
      output = le(t, r + 0.1) ? close() : current;
    else if (!le(r + 0.1, s))
      output = low();
  } else if (frame == 1 || frame == 2) {
    if (t + 0.1 < r) {
      if (s < r + 0.1)
        output = low();
    } else if (r + 0.1 < t)
      output = current;
    else
      output = close();
  } else if (le(r, t + 0.1))
    output = le(t, r + 0.1) ? close() : current;
  else if (s < r + 0.1)
    output = low();
  return output;
}
} // namespace

void detail::repair(std::array<Plane, 4> &frames,
                    const std::vector<std::pair<int, int>> &coordinates,
                    const Check &check) {
  if (coordinates.empty())
    return;
  const auto width = static_cast<std::ptrdiff_t>(frames[0].width);
  // Only changed positions need an immutable snapshot. Neighbor reads see the
  // original planes; the estimate sees all preceding repairs in vendor order.
  std::vector<std::pair<std::ptrdiff_t, std::array<Sample, 4>>> saved;
  saved.reserve(coordinates.size());
  for (auto [y, x] : coordinates) {
    const auto i = y * width + x;
    saved.push_back({i,
                     {frames[0].pixels[i], frames[1].pixels[i],
                      frames[2].pixels[i], frames[3].pixels[i]}});
  }
  std::sort(saved.begin(), saved.end(),
            [](auto &a, auto &b) { return a.first < b.first; });
  const auto original = [&](unsigned f, std::ptrdiff_t i) {
    const auto found =
        std::lower_bound(saved.begin(), saved.end(), i,
                         [](auto &a, auto b) { return a.first < b; });
    return found != saved.end() && found->first == i ? found->second[f]
                                                     : frames[f].pixels[i];
  };
  constexpr int mapping[4][5] = {
      {-1, -1, 2, 1, 3}, {-1, 1, 3, 0, 2}, {1, 1, 0, 3, 1}, {1, -1, 1, 2, 0}};
  constexpr int green[4][2] = {{1, 3}, {0, 2}, {3, 1}, {2, 0}};
  for (auto [y, x] : coordinates) {
    if (check && x == coordinates.front().second)
      check();
    const auto i = y * width + x;
    for (unsigned f = 0; f < 4; ++f) {
      const auto [dy, dx, df, rf, ff] = mapping[f];
      const auto d = i + dy * width + dx, far = i + 2 * dy * width;
      const auto current = original(f, i), diagonal = original(df, d);
      const double mean = (unsigned(original(green[f][0], i + dx)) +
                           original(green[f][1], i + dy * width)) /
                          2.0;
      const double far_mean =
          (unsigned(original(df, far + dx)) + original(f, i + dy * width)) /
          2.0;
      const double denominator = static_cast<float>(
          divide(frames[rf].pixels[d], frames[df].pixels[d]));
      const double estimate = divide(frames[ff].pixels[far], denominator);
      frames[f].pixels[i] = repair_value(
          f, current, mean, diagonal, f == 0 ? original(3, i) : current,
          divide(original(rf, d), diagonal), divide(far_mean, mean),
          divide(original(ff, far), current), estimate);
    }
  }
}
} // namespace cnefx
