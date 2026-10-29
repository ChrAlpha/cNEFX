#include "nefx.hpp"

#include <algorithm>
#include <iostream>
#include <set>

using namespace cnefx;
using namespace cnefx::detail;
namespace {
void require(bool ok) {
  if (!ok)
    throw std::runtime_error("Format test failed.");
}
void merge(int argc, char **argv, bool corrected) {
  const unsigned count = argc - 3;
  output_groups(count, 1);
  std::vector<Source> sources;
  std::set<std::pair<std::uint64_t, std::uint64_t>> identities;
  for (int i = 3; i < argc; ++i) {
    sources.emplace_back(argv[i]);
    const auto &state = sources.back().initial;
    if (!identities.emplace(state.device, state.inode).second)
      throw std::invalid_argument("Duplicate source identity.");
  }
  std::vector<Metadata> metadata(count);
  auto first = read_frame(sources[0]);
  metadata[0] = std::move(first.second);
  const auto &info = metadata[0];
  if (corrected && !info.radial)
    throw std::invalid_argument(info.radial_error);
  auto pixels = reconstruct(
      count,
      [&](unsigned i) {
        if (!i)
          return std::move(first.first);
        auto frame = read_frame(sources[i]);
        metadata[i] = std::move(frame.second);
        return std::move(frame.first);
      },
      info.sequence.crop, Selection::parse(info.sequence.selector), info.sequence.iso,
      corrected ? info.radial : std::nullopt);
  validate_sequence(metadata, corrected);
  Publication pending(argv[2], false);
  write_nefx(sources[0], pending.temporary(), pixels, count);
  verify_rgb(pending.temporary(), pixels);
  for (const auto &source : sources)
    source.verify();
  pending.publish([&] {
    for (const auto &source : sources)
      source.check();
  });
  pending.cleanup();
}
void tests() {
  require(crc16(Bytes{'1', '2', '3', '4', '5', '6', '7', '8', '9'}) == 0x29b1);
  const Rgb values{4, 1, {0, 1, 16382, 16383, 1, 2, 3, 4, 5, 6, 7, 8}};
  auto payload = packed14(values);
  Bytes bytes;
  payload.chunks([&](auto chunk) { bytes.insert(bytes.end(), chunk.begin(), chunk.end()); }, {});
  require(bytes.size() == 32);
  for (unsigned i = 0; i < 3; ++i) {
    std::uint64_t word = 0;
    for (unsigned k = 0; k < 7; ++k)
      word |= std::uint64_t(bytes[i * 7 + k]) << (k * 8);
    for (unsigned k = 0; k < 4; ++k)
      require(((word >> (14 * k)) & 16383) == values.pixels[i * 4 + k]);
  }
  require(std::all_of(bytes.begin() + 21, bytes.end(), [](auto b) { return b == 0; }));
  Rgb image{640, 424, std::vector<Sample>(640 * 424 * 3, 100)};
  for (unsigned y = 0; y < 424; ++y)
    for (unsigned c = 0; c < 3; ++c)
      image.pixels[(y * 640 + 639) * 3 + c] = 888;
  std::fill(image.pixels.end() - 640 * 3, image.pixels.end(), 999);
  const auto preview_bytes = preview(image);
  for (unsigned y = 0; y < 424; ++y)
    for (unsigned x = 0; x < 640; ++x)
      for (unsigned c = 0; c < 3; ++c)
        require(u16(preview_bytes, (y * 640 * 3 + x * 3 + c) * 2) ==
                (x == 639 || y == 423 ? 0 : 100));
  Metadata a;
  a.sequence.count = 4;
  a.shot = 10;
  a.timestamp = "2026-10-29T00:00:00.000000";
  std::vector<Metadata> sequence(4, a);
  for (unsigned i = 0; i < 4; ++i)
    sequence[i].shot += i;
  validate_sequence(sequence, false);
  sequence[2].shot += 1;
  bool rejected = false;
  try {
    validate_sequence(sequence, false);
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  require(rejected);
}
} // namespace
int main(int argc, char **argv) {
  try {
    if (argc == 1)
      tests();
    else if (argc >= 7 &&
             (std::string(argv[1]) == "--merge" || std::string(argv[1]) == "--merge-corrected"))
      merge(argc, argv, std::string(argv[1]) == "--merge-corrected");
    else
      return 2;
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
