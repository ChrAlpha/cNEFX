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
template <class F> bool rejects(F f) {
  try {
    f();
  } catch (const std::invalid_argument &) {
    return true;
  }
  return false;
}
void sequences() {
  // Two interleaved 4-frame bursts in shuffled order, as a directory may list them.
  std::vector<Metadata> input;
  for (const auto [burst, shot] : std::vector<std::pair<unsigned, unsigned>>{
           {7, 3}, {9, 2}, {7, 1}, {9, 4}, {7, 4}, {9, 1}, {7, 2}, {9, 3}}) {
    Metadata item;
    item.sequence.make = "NIKON CORPORATION";
    item.sequence.model = "NIKON Z 8";
    item.sequence.count = 4;
    item.sequence.burst = burst;
    item.sequence.identity_hash = "camera";
    item.shot = shot;
    input.push_back(item);
  }
  const auto groups = group_sequences(input);
  require(groups == std::vector<std::vector<unsigned>>({{2, 6, 0, 4}, {5, 1, 7, 3}}));
  for (const auto &group : groups) {
    std::vector<Metadata> ordered;
    for (const auto index : group)
      ordered.push_back(input[index]);
    validate_sequence(ordered, false);
    require(rejects([&] { validate_sequence(std::span(ordered).first(3), false); }));
    ordered[3].shot = ordered[2].shot;
    require(rejects([&] { validate_sequence(ordered, false); }));
  }
  // Equal burst identities from another body form their own sequence.
  auto other = input;
  for (auto &item : other)
    item.sequence.identity_hash = "other camera";
  other.insert(other.begin(), input.begin(), input.end());
  require(group_sequences(other).size() == 4);
  input[5].sequence.model = "NIKON Z f";
  require(rejects([&] { group_sequences(input); }));
  // Only source captures with active pixel-shift metadata are classified as such.
  Directory maker;
  maker.set(0x56, 7, Bytes{'0', '1', '0', '0', 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0});
  maker.set(0x5e, 7, Bytes{'0', '1', '0', '0', 1});
  const auto root = [](const Directory &note) {
    Directory exif, result;
    exif.tags[37500].children.push_back(note);
    result.tags[34665].children.push_back(exif);
    return result;
  };
  require(!skip_reason(root(maker)));
  auto ordinary = maker;
  ordinary.tags[0x56].data[12] = 0;
  const std::string_view inactive = "No active pixel-shift metadata.";
  require(skip_reason(root(ordinary)) == inactive && skip_reason(Directory{}) == inactive);
  auto merged = maker;
  merged.set(0x0e2d, 7, Bytes{});
  require(skip_reason(root(merged)) == std::string_view("Already merged NEFX."));
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
      [&](std::span<const unsigned> batch) {
        std::vector<Plane> planes;
        for (const auto i : batch) {
          if (!i) {
            planes.push_back(std::move(first.first));
            continue;
          }
          auto frame = read_frame(sources[i]);
          metadata[i] = std::move(frame.second);
          planes.push_back(std::move(frame.first));
        }
        return planes;
      },
      info.sequence.crop, info.sequence.selection, info.sequence.iso,
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
  sequences();
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
