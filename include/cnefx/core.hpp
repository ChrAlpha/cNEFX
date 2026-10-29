#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace cnefx {
using Sample = std::uint16_t;
using Check = std::function<void()>;

struct Crop {
  std::uint32_t left, top, width, height;
  bool operator==(const Crop &) const = default;
};
struct Plane {
  std::uint32_t width = 0, height = 0;
  std::vector<Sample> pixels;
};
struct Rgb {
  std::uint32_t width = 0, height = 0;
  std::vector<Sample> pixels;
};
struct RadialCorrection {
  std::array<std::int32_t, 2> center{};
  std::array<std::array<std::int32_t, 3>, 2> coefficients{};
  bool operator==(const RadialCorrection &) const = default;
};
struct Selection {
  int reference_height = 0, reference_width = 0, half_grid_start = 2, first_row = 4;
  int middle_step = 0, outer_step = 0, inner_step = 0;
  int middle_count = 1, outer_count = 1, inner_before = 0, inner_after = 0;
  static Selection parse(std::span<const std::uint8_t> bytes);
  std::vector<std::pair<int, int>> coordinates(int width, int height) const;
};

using Reader = std::function<Plane(unsigned)>;
using Groups = std::vector<std::vector<unsigned>>;
Groups output_groups(unsigned count, unsigned outputs);
std::size_t sample_count(std::uint32_t width, std::uint32_t height, unsigned channels);
void validate_crop(Crop crop, std::uint32_t width, std::uint32_t height);
Rgb reconstruct(unsigned count, const Reader &reader, Crop crop, const Selection &selection,
                unsigned iso, const std::optional<RadialCorrection> &radial = {},
                const Check &check = {});
Rgb correct_crop(const Rgb &image, std::uint32_t raw_width, std::uint32_t raw_height,
                 unsigned scale, Crop crop, const RadialCorrection &correction,
                 const Check &check = {});

namespace detail {
void repair(std::array<Plane, 4> &frames, const std::vector<std::pair<int, int>> &coordinates,
            const Check &check);
}
} // namespace cnefx
