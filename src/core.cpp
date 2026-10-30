#include "cnefx/core.hpp"

#include <algorithm>
#include <limits>
#include <numeric>

namespace cnefx {
namespace {
constexpr std::array<unsigned, 4> phase_x{0, 1, 1, 0}, phase_y{0, 0, 1, 1};
constexpr unsigned channels[4][4] = {{2, 1, 0, 1}, {1, 2, 1, 0}, {0, 1, 2, 1}, {1, 0, 1, 2}};

void sample(Rgb &output, const Plane &raw, Crop crop, unsigned scale, unsigned group,
            unsigned frame, const Check &check) {
  for (unsigned corner = 0; corner < 4; ++corner) {
    const auto channel = channels[frame][corner];
    const auto y0 = (phase_y[corner] + 1 - (crop.top % 2)) % 2;
    const auto x0 = (phase_x[corner] + 1 - (crop.left % 2)) % 2;
    for (std::uint32_t y = y0; y < crop.height; y += 2) {
      if (check && y % 128 < 2)
        check();
      const auto sy = static_cast<std::size_t>(y + crop.top - phase_y[frame]) * raw.width;
      const auto dy = static_cast<std::size_t>(scale * y + phase_y[group]) * output.width;
      for (std::uint32_t x = x0; x < crop.width; x += 2) {
        const Sample value = raw.pixels[sy + x + crop.left - phase_x[frame]];
        auto &target = output.pixels[(dy + scale * x + phase_x[group]) * 3 + channel];
        target = channel == 1 && frame >= 2 ? (unsigned(value) + target) >> 1 : value;
      }
    }
  }
}

void average(Plane &first, const Plane &second, const std::vector<std::pair<int, int>> &coordinates,
             const Check &check) {
  std::vector<Sample> originals;
  originals.reserve(coordinates.size());
  for (auto [y, x] : coordinates)
    originals.push_back(first.pixels[static_cast<std::size_t>(y + 2) * first.width + x]);
  for (unsigned y = 2; y < first.height; ++y) {
    if (check && y % 128 == 0)
      check();
    for (unsigned x = 0; x < first.width; ++x) {
      auto &target = first.pixels[static_cast<std::size_t>(y) * first.width + x];
      target =
          (unsigned(target) + second.pixels[static_cast<std::size_t>(y - 2) * first.width + x]) >>
          1;
    }
  }
  std::size_t i = 0;
  for (auto [y, x] : coordinates) {
    first.pixels[static_cast<std::size_t>(y) * first.width + x] =
        second.pixels[static_cast<std::size_t>(y - 2) * first.width + x];
    first.pixels[static_cast<std::size_t>(y + 2) * first.width + x] = originals[i++];
  }
}
} // namespace

std::size_t sample_count(std::uint32_t width, std::uint32_t height, unsigned channels) {
  if (!width || !height || !channels || width > 65535 || height > 65535 || channels > 3)
    throw std::invalid_argument("Unsupported image dimensions.");
  const std::uint64_t size = std::uint64_t(width) * height * channels;
  if (size > std::numeric_limits<std::size_t>::max() / sizeof(Sample))
    throw std::invalid_argument("Image exceeds address space.");
  return static_cast<std::size_t>(size);
}

void validate_crop(Crop crop, std::uint32_t width, std::uint32_t height) {
  if (!crop.left || !crop.top || !crop.width || !crop.height || crop.left > width ||
      crop.top > height || crop.width > width - crop.left || crop.height > height - crop.top)
    throw std::invalid_argument("Crop is outside the recovered sampling domain.");
}

Groups output_groups(unsigned count, unsigned outputs) {
  const bool allowed =
      (count == 4 && outputs == 1) || (count == 8 && (outputs == 1 || outputs == 2)) ||
      (count == 16 && (outputs == 1 || outputs == 4)) ||
      (count == 32 && (outputs == 1 || outputs == 2 || outputs == 4 || outputs == 8));
  if (!allowed)
    throw std::invalid_argument("Unsupported capture count and output count combination.");
  Groups groups(outputs);
  for (unsigned i = 0; i < outputs; ++i) {
    for (unsigned j = 0; j < count / outputs; ++j)
      groups[i].push_back(count == 32 && outputs == 4 ? i * 4 + j % 4 + (j / 4) * 16
                                                      : i * (count / outputs) + j);
  }
  return groups;
}

Rgb reconstruct(unsigned count, const Reader &reader, Crop crop, const Selection &selection,
                unsigned iso, const std::optional<RadialCorrection> &radial, const Check &check,
                unsigned jobs) {
  output_groups(count, 1);
  if (!iso)
    throw std::invalid_argument("Expected a positive ISO value.");
  const bool paired = count == 8 || count == 32;
  const unsigned reduced = paired ? count / 2 : count;
  const unsigned scale = reduced == 16 ? 2 : 1;
  std::uint32_t width = 0, height = 0;
  std::size_t size = 0;
  std::vector<std::pair<int, int>> coordinates;
  Crop sampling{};
  Rgb output;
  // The first plane fixes the geometry; every plane must match it.
  const auto prepare = [&](const Plane &plane) {
    width = plane.width;
    height = plane.height;
    if (width < 8 || height < 8 || width % 2 || height % 2)
      throw std::invalid_argument("Expected even RAW planes at least 8x8.");
    size = sample_count(width, height, 1);
    validate_crop(crop, width, height);
    coordinates =
        iso <= 3200 ? selection.coordinates(width, height) : std::vector<std::pair<int, int>>{};
    sampling = radial ? Crop{1, 1, width - 1, height - 1} : crop;
    output = {scale * sampling.width, scale * sampling.height, {}};
    output.pixels.resize(sample_count(output.width, output.height, 3));
  };
  const auto fetch = [&](std::span<const unsigned> indices) {
    if (check)
      check();
    auto planes = reader(indices);
    if (planes.size() != indices.size())
      throw std::invalid_argument("Reader returned an incomplete batch.");
    for (const auto &plane : planes) {
      if (!size)
        prepare(plane);
      if (plane.width != width || plane.height != height || plane.pixels.size() != size)
        throw std::invalid_argument("RAW frames must have identical geometry.");
      if (*std::max_element(plane.pixels.begin(), plane.pixels.end()) > 16383)
        throw std::invalid_argument("RAW samples exceed 14 bits.");
    }
    return planes;
  };
  for (unsigned start = 0; start < reduced; start += 4) {
    if (paired) {
      for (unsigned i = 0; i < 4; ++i) {
        auto pair = fetch(std::array{start + i, start + i + count / 2});
        average(pair[0], pair[1], coordinates, check);
        pair.pop_back();
        sample(output, pair[0], sampling, scale, start / 4, i, check);
      }
    } else {
      auto planes = fetch(std::array{start, start + 1, start + 2, start + 3});
      std::array<Plane, 4> frames{std::move(planes[0]), std::move(planes[1]), std::move(planes[2]),
                                  std::move(planes[3])};
      planes.clear();
      detail::repair(frames, coordinates, check);
      for (unsigned i = 0; i < 4; ++i)
        sample(output, frames[i], sampling, scale, start / 4, i, check);
    }
  }
  return radial ? correct_crop(output, width, height, scale, crop, *radial, check, jobs)
                : std::move(output);
}
} // namespace cnefx
