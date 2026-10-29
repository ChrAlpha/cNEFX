#include "cnefx/core.hpp"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>

using namespace cnefx;
namespace {
void require(bool ok) {
  if (!ok)
    throw std::runtime_error("Core test failed.");
}
template <class F> void rejects(F f) {
  bool rejected = false;
  try {
    f();
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  require(rejected);
}
unsigned read32(std::istream &input) {
  unsigned result = 0;
  for (unsigned i = 0; i < 4; ++i) {
    int c = input.get();
    if (c < 0)
      throw std::runtime_error("Truncated fixture.");
    result |= unsigned(c) << (8 * i);
  }
  return result;
}
void replay(const char *source, const char *destination) {
  std::ifstream input(source, std::ios::binary);
  const auto count = read32(input), width = read32(input),
             height = read32(input);
  const Crop crop{read32(input), read32(input), read32(input), read32(input)};
  const auto iso = read32(input), corrected = read32(input);
  std::array<std::uint8_t, 28> selector{};
  input.read(reinterpret_cast<char *>(selector.data()), selector.size());
  RadialCorrection radial;
  for (auto &v : radial.center)
    v = static_cast<std::int32_t>(read32(input));
  for (auto &triple : radial.coefficients)
    for (auto &v : triple)
      v = static_cast<std::int32_t>(read32(input));
  const auto offset = input.tellg();
  std::vector<unsigned> reads(count);
  const auto result = reconstruct(
      count,
      [&](unsigned index) {
        require(reads.at(index)++ == 0);
        Plane plane{width, height,
                    std::vector<Sample>(sample_count(width, height, 1))};
        input.seekg(offset + std::streamoff(index * plane.pixels.size() * 2));
        for (auto &v : plane.pixels) {
          const int a = input.get(), b = input.get();
          if (a < 0 || b < 0)
            throw std::runtime_error("Truncated samples.");
          v = a | (b << 8);
        }
        return plane;
      },
      crop, Selection::parse(selector), iso,
      corrected ? std::optional(radial) : std::nullopt);
  require(
      std::all_of(reads.begin(), reads.end(), [](auto n) { return n == 1; }));
  std::ofstream output(destination, std::ios::binary);
  for (auto v : result.pixels) {
    output.put(char(v & 255));
    output.put(char(v >> 8));
  }
  output.close();
  require(bool(output));
}
void tests() {
  for (const auto [count, outputs] :
       std::vector<std::pair<unsigned, unsigned>>{{4, 1},
                                                  {8, 1},
                                                  {8, 2},
                                                  {16, 1},
                                                  {16, 4},
                                                  {32, 1},
                                                  {32, 2},
                                                  {32, 4},
                                                  {32, 8}}) {
    auto groups = output_groups(count, outputs);
    std::vector<unsigned> visited(count);
    for (const auto &group : groups)
      for (auto index : group)
        ++visited.at(index);
    require(std::all_of(visited.begin(), visited.end(),
                        [](auto n) { return n == 1; }));
  }
  require(output_groups(32, 4)[1] ==
          std::vector<unsigned>({4, 5, 6, 7, 20, 21, 22, 23}));
  rejects([] { output_groups(16, 2); });
  Selection empty;
  empty.outer_count = 0;
  auto result = reconstruct(
      4,
      [](unsigned i) {
        return Plane{8, 8,
                     std::vector<Sample>(
                         64, std::array<Sample, 4>{100, 201, 300, 400}[i])};
      },
      {2, 2, 2, 2}, empty, 3201);
  require(result.pixels == std::vector<Sample>({100, 300, 300, 201, 200, 400,
                                                400, 200, 201, 300, 300, 100}));
  result = reconstruct(
      16,
      [](unsigned i) {
        return Plane{8, 8, std::vector<Sample>(64, 100 * (i / 4 + 1))};
      },
      {2, 2, 1, 1}, empty, 3201);
  require(result.pixels == std::vector<Sample>({100, 100, 100, 200, 200, 200,
                                                400, 400, 400, 300, 300, 300}));
  result = reconstruct(
      8,
      [](unsigned i) {
        Plane p{8, 8, std::vector<Sample>(64, i < 4 ? 100 : 201)};
        if (i >= 4)
          std::fill(p.pixels.begin() + 16, p.pixels.end(), 401);
        return p;
      },
      {3, 3, 1, 1}, empty, 3201);
  require(result.pixels == std::vector<Sample>({150, 150, 150}));
  rejects([&] {
    reconstruct(
        4, [](unsigned) { return Plane{9, 8, std::vector<Sample>(72)}; },
        {1, 1, 4, 4}, empty, 3201);
  });
  rejects([&] {
    reconstruct(
        4, [](unsigned) { return Plane{8, 8, std::vector<Sample>(64, 16384)}; },
        {1, 1, 4, 4}, empty, 3201);
  });
  Selection unsafe{8, 8, 2, 1};
  rejects([&] { unsafe.coordinates(8, 8); });
  Selection costly;
  costly.outer_count = costly.middle_count = 255;
  rejects([&] { costly.coordinates(8, 8); });
  bool cancelled = false;
  try {
    reconstruct(
        4, [](unsigned) { return Plane{}; }, {1, 1, 4, 4}, empty, 3201, {},
        [] { throw std::runtime_error("cancelled"); });
  } catch (const std::runtime_error &e) {
    cancelled = std::string(e.what()) == "cancelled";
  }
  require(cancelled);
}
} // namespace
int main(int argc, char **argv) {
  try {
    if (argc == 4 && std::string(argv[1]) == "--replay")
      replay(argv[2], argv[3]);
    else if (argc == 1)
      tests();
    else
      return 2;
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
