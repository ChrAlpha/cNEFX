#include "nefx.hpp"

#include <algorithm>
#include <bit>
#include <ctime>
#include <limits>

#ifndef CNEFX_VERSION
#define CNEFX_VERSION "unknown"
#endif

namespace cnefx::detail {
namespace {
void validate_rgb(const Rgb &image) {
  if (image.pixels.size() != sample_count(image.width, image.height, 3) ||
      *std::max_element(image.pixels.begin(), image.pixels.end()) > 16383)
    throw std::invalid_argument("Expected nonempty RGB samples in the 14-bit range.");
}
Directory &child(Tag &tag) {
  if (tag.children.size() != 1)
    throw std::invalid_argument("Expected one embedded Nikon directory.");
  return tag.children.front();
}
void patch(Directory &directory, unsigned code,
           const std::vector<std::pair<unsigned, Bytes>> &updates, bool crc = false) {
  const auto &tag = directory.at(code);
  auto data = tag.data;
  if (tag.kind != 7 || data.size() < 4 || !std::equal(data.begin(), data.begin() + 4, "0100"))
    throw std::invalid_argument("Unsupported Nikon metadata update block.");
  for (const auto &[offset, value] : updates) {
    if (offset > data.size() - (crc ? 2 : 0) || value.size() > data.size() - (crc ? 2 : 0) - offset)
      throw std::invalid_argument("Truncated Nikon metadata update block.");
    std::copy(value.begin(), value.end(), data.begin() + offset);
  }
  if (crc)
    put16(data, data.size() - 2, crc16(std::span(data).first(data.size() - 2)));
  directory.set(code, 7, std::move(data));
}
Bytes text(const std::string &value) {
  Bytes result(value.begin(), value.end());
  result.push_back(0);
  return result;
}
} // namespace

std::uint16_t crc16(std::span<const std::uint8_t> bytes) {
  unsigned crc = 65535;
  for (auto b : bytes) {
    crc ^= unsigned(b) << 8;
    for (unsigned i = 0; i < 8; ++i)
      crc = ((crc << 1) ^ ((crc & 0x8000) ? 0x1021 : 0)) & 65535;
  }
  return crc;
}
Payload packed14(const Rgb &image) {
  validate_rgb(image);
  if (image.width % 4)
    throw std::invalid_argument("NEFX packing requires width divisible by four.");
  const auto row_bytes = std::size_t(image.width) * 3 * 7 / 4,
             stride = (row_bytes + 15) & ~std::size_t{15};
  return {std::uint64_t(image.height) * stride,
          [&image, stride](const Sink &sink, const Check &check) {
            Bytes bytes(stride * std::min(image.height, 64u));
            for (unsigned start = 0; start < image.height; start += 64) {
              if (check)
                check();
              const auto rows = std::min(64u, image.height - start);
              for (unsigned y = 0; y < rows; ++y) {
                const auto *source = image.pixels.data() + std::size_t(start + y) * image.width * 3;
                auto *target = bytes.data() + y * stride;
                for (unsigned i = 0; i < image.width * 3 / 4; ++i) {
                  std::uint64_t word = 0;
                  for (unsigned k = 0; k < 4; ++k)
                    word |= std::uint64_t(source[i * 4 + k]) << (14 * k);
                  for (unsigned k = 0; k < 7; ++k)
                    target[i * 7 + k] = (word >> (8 * k)) & 255;
                }
              }
              sink(std::span(bytes).first(rows * stride));
            }
          }};
}
Bytes preview(const Rgb &image, const Check &check) {
  validate_rgb(image);
  Bytes result(640 * 424 * 3 * 2);
  std::array<unsigned, 641> xs{};
  std::array<unsigned, 425> ys{};
  for (unsigned i = 0; i < xs.size(); ++i)
    xs[i] = static_cast<unsigned>(i / (640.0 / image.width));
  for (unsigned i = 0; i < ys.size(); ++i)
    ys[i] = static_cast<unsigned>(i / (424.0 / image.height));
  std::vector<std::uint64_t> integral((std::size_t(image.width) + 1) * 3);
  for (unsigned row = 0; row < 424; ++row) {
    if (check)
      check();
    const auto y0 = ys[row], y1 = std::min(ys[row + 1], image.height - 1);
    if (y1 <= y0)
      continue;
    std::fill(integral.begin(), integral.end(), 0);
    for (unsigned y = y0; y < y1; ++y)
      for (unsigned x = 0; x < image.width; ++x)
        for (unsigned c = 0; c < 3; ++c)
          integral[(x + 1) * 3 + c] += image.pixels[(std::size_t(y) * image.width + x) * 3 + c];
    for (unsigned x = 1; x <= image.width; ++x)
      for (unsigned c = 0; c < 3; ++c)
        integral[x * 3 + c] += integral[(x - 1) * 3 + c];
    for (unsigned col = 0; col < 640; ++col) {
      const auto x0 = xs[col], x1 = std::min(xs[col + 1], image.width - 1);
      if (x1 <= x0)
        continue;
      const auto area = std::uint64_t(x1 - x0) * (y1 - y0);
      for (unsigned c = 0; c < 3; ++c) {
        const auto sum = (integral[x1 * 3 + c] - integral[x0 * 3 + c]) & 0xffffffffu;
        put16(result, (std::size_t(row) * 640 * 3 + col * 3 + c) * 2,
              static_cast<Sample>(sum / area));
      }
    }
  }
  return result;
}

void write_nefx(Source &source, const fs::path &path, const Rgb &image, unsigned count,
                const Check &check) {
  output_groups(count, 1);
  validate_rgb(image);
  if (source.hash.empty())
    throw std::logic_error("Serialize only a decoded source.");
  // Serialize from bytes that reproduce the hash recorded at decoding.
  auto root = read_tiff(source.read(check));
  auto &subimages = root.at(330).children;
  if (subimages.size() != 6)
    throw std::invalid_argument("Expected the observed six-SubIFD Nikon layout.");
  auto &raw = subimages[1];
  if (raw.at(262).number() != 32803 || raw.images.size() != 1 || !raw.images.contains(273) ||
      raw.images.at(273).size() != 1)
    throw std::invalid_argument("Expected the observed single-strip source Bayer RAW.");
  auto &maker = child(child(root.at(34665)).at(37500));
  auto &info = child(raw.at(0xc7d5));
  const auto crop = maker.at(0x45).integers();
  const unsigned scale = count >= 16 ? 2 : 1;
  if (crop.size() != 4 || image.width != crop[2] * scale || image.height != crop[3] * scale)
    throw std::invalid_argument("RGB dimensions do not match the source crop and mode.");
  raw.numbers(256, 4, {image.width});
  raw.numbers(257, 4, {image.height});
  raw.numbers(258, 3, {14, 14, 14});
  raw.numbers(259, 3, {34713});
  raw.numbers(262, 3, {2});
  raw.numbers(277, 3, {3});
  raw.numbers(278, 4, {image.height});
  raw.numbers(284, 3, {1});
  raw.images[273] = {packed14(image)};
  root.set(305, 2, text("cNEFX " CNEFX_VERSION));
  const auto now = std::time(nullptr);
  std::tm date{};
#ifdef _WIN32
  if (localtime_s(&date, &now))
    throw std::runtime_error("Cannot format output timestamp.");
#else
  if (!localtime_r(&now, &date))
    throw std::runtime_error("Cannot format output timestamp.");
#endif
  std::array<char, 32> formatted{};
  if (!std::strftime(formatted.data(), formatted.size(), "%Y:%m:%d %H:%M:%S", &date))
    throw std::runtime_error("Cannot format output timestamp.");
  root.set(306, 2, text(formatted.data()));
  if (root.tags.contains(0xcd41))
    root.set(0xcd41, root.at(0xcd41).kind, {});
  maker.numbers(0x45, 3, {0, 0, image.width, image.height});
  auto compression = maker.at(0x51).data;
  if (compression.size() != 24 || !std::equal(compression.begin(), compression.begin() + 4, "0102"))
    throw std::invalid_argument("Unsupported Nikon compression metadata.");
  put16(compression, 10, 10);
  maker.set(0x51, 7, std::move(compression));
  maker.tags.erase(0x58);
  maker.tags.erase(0x96);
  if (maker.tags.contains(0x68))
    patch(maker, 0x68, {{4, Bytes{0}}});
  if (scale == 2) {
    for (auto [node, code] :
         std::array<std::pair<Directory *, unsigned>, 2>{{{&maker, 0x54}, {&info, 8}}}) {
      std::vector<std::pair<unsigned, Bytes>> updates;
      for (unsigned offset : {4u, 12u}) {
        const auto value = std::int64_t(i32(node->at(code).data, offset)) * 2;
        if (value < INT32_MIN || value > INT32_MAX)
          throw std::invalid_argument("Scaled Nikon metadata exceeds signed 32-bit range.");
        Bytes data(4);
        put32(data, 0, std::bit_cast<std::uint32_t>(static_cast<std::int32_t>(value)));
        updates.emplace_back(offset, std::move(data));
      }
      patch(*node, code, updates, node == &info);
    }
    if (maker.tags.contains(0xb7)) {
      auto data = maker.at(0xb7).data;
      if (data.size() < 8)
        throw std::invalid_argument("Truncated autofocus metadata.");
      std::fill(data.begin() + 8, data.end(), 0);
      maker.set(0xb7, maker.at(0xb7).kind, std::move(data));
    }
  }
  const auto &selector = maker.at(0x5e).data;
  if (selector.size() != 28)
    throw std::invalid_argument("Invalid pixel-shift marker source.");
  Bytes marker{'0', '1', '0',         '0',
               2,   0,   selector[4], static_cast<std::uint8_t>(std::countr_zero(count) - 1)};
  marker.resize(16);
  maker.set(0xe2d, 7, std::move(marker));
  Bytes private_preview{'0', '1', '0', '0', 16, 0, 0, 0, 0x80, 2, 0xa8, 1, 3, 14, 14, 14};
  const auto pixels = preview(image, check);
  private_preview.insert(private_preview.end(), pixels.begin(), pixels.end());
  maker.set(0xe2e, 7, std::move(private_preview));
  info.tags.erase(7);
  info.tags.erase(15);
  Bytes dimensions(8);
  put16(dimensions, 4, image.width);
  put16(dimensions, 6, image.height);
  patch(info, 11, {{4, dimensions}}, true);
  patch(info, 13, {{4, Bytes{10, 0}}}, true);
  File output(path, FileMode::create);
  write_tiff(root, output, check);
  source.check();
}
} // namespace cnefx::detail
