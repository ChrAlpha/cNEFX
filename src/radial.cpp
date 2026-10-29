#include "cnefx/core.hpp"

#include <algorithm>
#include <cmath>

namespace cnefx {
namespace {
double weight(double distance) {
  distance = std::abs(distance);
  const double squared = distance * distance, cubed = squared * distance;
  return distance <= 1   ? 1 - 2 * squared + cubed
         : distance <= 2 ? 4 - 8 * distance + 5 * squared - cubed
                         : 0;
}
} // namespace

Rgb correct_crop(const Rgb &image, std::uint32_t raw_width,
                 std::uint32_t raw_height, unsigned scale, Crop crop,
                 const RadialCorrection &correction, const Check &check) {
  if ((scale != 1 && scale != 2) || raw_width < 8 || raw_height < 8 ||
      raw_width % 2 || raw_height % 2 ||
      image.width != scale * (raw_width - 1) ||
      image.height != scale * (raw_height - 1) ||
      image.pixels.size() != sample_count(image.width, image.height, 3))
    throw std::invalid_argument("Expected full reconstructed RGB geometry.");
  if (*std::max_element(image.pixels.begin(), image.pixels.end()) > 16383)
    throw std::invalid_argument("RGB samples exceed 14 bits.");
  validate_crop(crop, raw_width, raw_height);
  Rgb output{scale * crop.width, scale * crop.height, {}};
  output.pixels.resize(sample_count(output.width, output.height, 3));
  const auto x0 = scale * (crop.left - 1), y0 = scale * (crop.top - 1);
  const double cx =
      scale * ((raw_width - 1) / 2.0 + correction.center[0] / 256.0 - 1);
  const double cy =
      scale * ((raw_height - 1) / 2.0 + correction.center[1] / 256.0 - 1);
  const double radius = scale *
                        std::sqrt(double(raw_width) * raw_width +
                                  double(raw_height) * raw_height) /
                        2;
  for (unsigned y = 0; y < output.height; ++y) {
    if (check && y % 64 == 0)
      check();
    const double dy = double(y0 + y) - cy;
    for (unsigned x = 0; x < output.width; ++x) {
      const auto target = (static_cast<std::size_t>(y) * output.width + x) * 3;
      output.pixels[target + 1] =
          image.pixels[(static_cast<std::size_t>(y0 + y) * image.width + x0 +
                        x) *
                           3 +
                       1];
      const double dx = double(x0 + x) - cx;
      const double normalized = std::sqrt(dx * dx + dy * dy) / radius;
      const double squared = normalized * normalized;
      for (unsigned color = 0; color < 2; ++color) {
        const auto &triple = correction.coefficients[color];
        const double a = triple[0] / 1048576.0, b = triple[1] / 1048576.0,
                     c = triple[2] / 1048576.0;
        const double factor = squared * b + a + squared * normalized * c + 1;
        const double sx = factor * dx + cx, sy = factor * dy + cy;
        double sum = 0;
        if (sx > -2 && sx < image.width + 1.0 && sy > -2 &&
            sy < image.height + 1.0) {
          const auto ix = static_cast<int>(sx), iy = static_cast<int>(sy);
          for (int row = -1; row <= 2; ++row) {
            const int yy = iy + row;
            const double wy = weight(yy - sy);
            for (int col = -1; col <= 2; ++col) {
              const int xx = ix + col;
              const double wx = weight(xx - sx);
              if (yy >= 0 && yy < int(image.height) && xx >= 0 &&
                  xx < int(image.width))
                sum +=
                    image.pixels[(static_cast<std::size_t>(yy) * image.width +
                                  xx) *
                                     3 +
                                 color * 2] *
                    wx * wy;
            }
          }
        }
        output.pixels[target + color * 2] =
            static_cast<Sample>(std::clamp(sum, 0.0, 16383.0));
      }
    }
  }
  return output;
}
} // namespace cnefx
