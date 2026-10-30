#include "raw.hpp"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <numeric>
#include <string_view>

namespace cnefx::detail {
namespace {
void require(bool ok, const char *reason) {
  if (!ok)
    throw std::invalid_argument(reason);
}
const Directory &child(const Tag &tag) {
  require(tag.children.size() == 1, "Expected exactly one linked TIFF directory.");
  return tag.children.front();
}
const Bytes &block(const Directory &directory, unsigned code, std::string_view version,
                   std::size_t size) {
  const auto &tag = directory.at(code);
  require(tag.kind == 7 && tag.count == size && tag.data.size() == size &&
              std::equal(version.begin(), version.end(), tag.data.begin()),
          "Unsupported Nikon metadata version or length.");
  return tag.data;
}
int decimal(const std::string &value, std::size_t start, std::size_t size) {
  int n = 0;
  const auto *p = value.data() + start;
  require(std::all_of(p, p + size, [](auto c) { return c >= '0' && c <= '9'; }),
          "Invalid capture timestamp.");
  const auto result = std::from_chars(p, p + size, n);
  require(result.ec == std::errc{} && result.ptr == p + size && n >= 0,
          "Invalid capture timestamp.");
  return n;
}
std::string timestamp(const Directory &exif) {
  auto text = exif.at(36867).text();
  require(text.size() == 19 && text[4] == ':' && text[7] == ':' && text[10] == ' ' &&
              text[13] == ':' && text[16] == ':',
          "Invalid capture timestamp.");
  const int year = decimal(text, 0, 4), month = decimal(text, 5, 2), day = decimal(text, 8, 2);
  require(year >= 1 &&
              std::chrono::year_month_day{std::chrono::year{year},
                                          std::chrono::month{unsigned(month)},
                                          std::chrono::day{unsigned(day)}}
                  .ok() &&
              decimal(text, 11, 2) < 24 && decimal(text, 14, 2) < 60 && decimal(text, 17, 2) < 60,
          "Invalid capture timestamp.");
  auto subsecond = exif.tags.contains(37521) ? exif.at(37521).text() : "0";
  require(!subsecond.empty() && std::all_of(subsecond.begin(), subsecond.end(),
                                            [](auto c) { return c >= '0' && c <= '9'; }),
          "Invalid capture subsecond timestamp.");
  text[4] = text[7] = '-';
  text[10] = 'T';
  return text + "." + (subsecond + "000000").substr(0, 6);
}
std::pair<std::int64_t, std::int64_t> rational(const Tag &tag, bool positive) {
  require((tag.kind == 5 || tag.kind == 10) && tag.count == 1 && tag.data.size() == 8,
          "Missing or malformed exposure metadata.");
  const std::int64_t a = tag.kind == 5 ? std::int64_t(u32(tag.data, 0)) : i32(tag.data, 0);
  const std::int64_t b = tag.kind == 5 ? std::int64_t(u32(tag.data, 4)) : i32(tag.data, 4);
  require(b > 0 && (!positive || a > 0), "Invalid exposure metadata.");
  const auto gcd = std::gcd(a, b);
  return {a / gcd, b / gcd};
}
RadialCorrection radial(const Directory &maker, const Directory &raw) {
  const auto &center = block(maker, 0x54, "0100", 20);
  const auto &tag = raw.at(0xc7d5);
  require(tag.nikon_header == Bytes{'N', 'i', 'k', 'o', 'n', 0, 1, 3, 0, 0},
          "Unsupported Nikon radial coefficient header.");
  const auto &coefficients = block(child(tag), 7, "0100", 104);
  require(u32(center, 8) == 256 && u32(center, 16) == 256,
          "Unsupported radial center fixed-point scale.");
  RadialCorrection result;
  result.center = {i32(center, 4), i32(center, 12)};
  constexpr unsigned positions[2][3] = {{44, 28, 20}, {80, 64, 56}};
  for (unsigned color = 0; color < 2; ++color)
    for (unsigned i = 0; i < 3; ++i) {
      const auto offset = positions[color][i];
      require(u32(coefficients, offset + 4) == 1048576,
              "Unsupported radial coefficient fixed-point scale.");
      result.coefficients[color][i] = i32(coefficients, offset);
    }
  return result;
}
} // namespace
Metadata read_metadata(const Directory &root) {
  require(!root.tags.contains(50706), "Expected a source NEF, not DNG.");
  Metadata result;
  auto &m = result.sequence;
  m.make = root.at(271).text();
  m.model = root.at(272).text();
  require(m.make == "NIKON CORPORATION" && (m.model == "NIKON Z 8" || m.model == "NIKON Z f" ||
                                            m.model == "NIKON Z6_3" || m.model == "NIKON Z5_2"),
          "Unsupported camera model; expected Nikon Z8, Zf, Z6III or Z5II.");
  m.orientation = root.at(274).number();
  require(m.orientation >= 1 && m.orientation <= 8, "Unsupported TIFF orientation.");
  const auto &exif = child(root.at(34665));
  const auto &note = exif.at(37500);
  require(note.nikon_header == Bytes{'N', 'i', 'k', 'o', 'n', 0, 2, 17, 0, 0},
          "Unsupported Nikon MakerNote header.");
  const auto &maker = child(note);
  const auto &compression = block(maker, 0x51, "0102", 24);
  m.compression = u16(compression, 10);
  require(m.compression != 13 && m.compression != 14 && m.compression != 15,
          "Nikon HE/HE* is unsupported; use lossless-compressed 14-bit NEF.");
  require(m.compression == 3 || m.compression == 10, "Unsupported Nikon RAW compression code.");
  require(!maker.tags.contains(0x0e2d), "Expected a source NEF, not merged NEFX.");
  block(maker, 0x97, "0803", 166);
  const auto &capture = block(maker, 0x56, "0100", 16);
  const auto &shift = block(maker, 0x5e, "0100", 28);
  m.mode = shift[4];
  require(m.mode >= 1 && m.mode <= 4 && capture[12] == 1,
          "Expected active 4, 8, 16 or 32-frame pixel-shift metadata.");
  m.selector = shift;
  m.count = 2u << m.mode;
  m.burst = u32(capture, 4);
  result.shot = u32(capture, 8);
  const auto &crop_tag = maker.at(0x45);
  require(crop_tag.kind == 3 && crop_tag.count == 4, "Missing or unsupported Nikon crop metadata.");
  const auto crop = crop_tag.integers();
  m.crop = {crop[0], crop[1], crop[2], crop[3]};
  const Directory *raw = nullptr;
  for (const auto &page : root.at(330).children) {
    if (page.tags.contains(262) && page.at(262).number() == 32803) {
      require(!raw, "Expected exactly one Bayer RAW SubIFD.");
      raw = &page;
    }
  }
  require(raw, "Expected exactly one Bayer RAW SubIFD.");
  require(raw->at(258).number() == 14 && raw->at(277).number() == 1 &&
              raw->at(259).number() == 34713,
          "Expected a 14-bit Nikon Bayer RAW image.");
  require(raw->at(33421).integers() == std::vector<std::uint32_t>({2, 2}) &&
              raw->at(33422).data == Bytes{0, 1, 1, 2},
          "Only the observed RGGB source CFA layout is supported.");
  m.width = raw->at(256).number();
  m.height = raw->at(257).number();
  require(m.width >= 8 && m.height >= 8 && !(m.width % 2) && !(m.height % 2),
          "Unsupported RAW geometry.");
  sample_count(m.width, m.height, 1);
  validate_crop(m.crop, m.width, m.height);
  m.iso = exif.at(34855).number();
  require(m.iso > 0 && m.iso < 65535, "Missing or unsupported EXIF ISO value.");
  std::string serial = exif.tags.contains(42033) ? exif.at(42033).text() : "";
  const auto trim = [](std::string s) {
    const auto first = s.find_first_not_of(" \t\r\n\f\v");
    if (first == std::string::npos)
      return std::string{};
    return s.substr(first, s.find_last_not_of(" \t\r\n\f\v") - first + 1);
  };
  serial = trim(serial);
  if (serial.empty() && maker.tags.contains(0x1d))
    serial = trim(maker.at(0x1d).text());
  require(!serial.empty(), "Missing camera serial.");
  m.identity_hash = sha256(Bytes(serial.begin(), serial.end()));
  result.timestamp = timestamp(exif);
  if (exif.tags.contains(36881))
    m.timezone = exif.at(36881).text();
  const auto &shutter = maker.at(0xa7);
  require(shutter.kind == 4 && shutter.count == 1 && shutter.number() >= result.shot,
          "Shutter count is inconsistent with the shot number.");
  m.shutter_anchor = shutter.number() - result.shot;
  m.exposure = {rational(exif.at(33434), true), rational(exif.at(33437), true),
                rational(exif.at(37380), false)};
  try {
    result.radial = radial(maker, *raw);
  } catch (const std::invalid_argument &e) {
    result.radial_error = e.what();
  }
  return result;
}
void validate_sequence(const std::vector<Metadata> &metadata, bool radial) {
  output_groups(metadata.size(), 1);
  const auto &first = metadata.front();
  if (first.sequence.count != metadata.size())
    throw std::invalid_argument("Input count does not match the capture mode.");
  for (std::size_t i = 0; i < metadata.size(); ++i) {
    const auto &item = metadata[i];
    if (std::uint64_t(item.shot) != std::uint64_t(first.shot) + i)
      throw std::invalid_argument("Inputs must have consecutive shot numbers in capture order.");
    if (i && item.timestamp < metadata[i - 1].timestamp)
      throw std::invalid_argument("Capture timestamps move backwards within the sequence.");
    if (item.sequence != first.sequence)
      throw std::invalid_argument("Sequence metadata differs.");
    if (radial && !item.radial)
      throw std::invalid_argument("Unsupported radial correction metadata: " + item.radial_error);
    if (radial && item.radial != first.radial)
      throw std::invalid_argument("Sequence radial correction parameters differ.");
  }
}
} // namespace cnefx::detail
