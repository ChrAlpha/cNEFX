#pragma once

#include "tiff.hpp"

namespace cnefx::detail {
struct Calibration {
  std::array<std::uint32_t, 4> black{};
  std::array<double, 4> gains{};
  std::array<double, 9> matrix{};
  std::uint32_t white{};
  bool missing_matrix{};
  bool operator==(const Calibration &) const = default;
};
struct SequenceMetadata {
  std::string make, model, identity_hash;
  std::optional<std::string> timezone;
  std::uint32_t orientation{}, width{}, height{}, iso{}, compression{}, mode{}, count{}, burst{},
      shutter_anchor{};
  Crop crop{};
  Selection selection;
  std::array<std::pair<std::int64_t, std::int64_t>, 3> exposure{};
  Calibration calibration;
  bool operator==(const SequenceMetadata &) const = default;
};
struct Metadata {
  SequenceMetadata sequence;
  std::uint32_t shot{};
  std::string timestamp;
  std::optional<RadialCorrection> radial;
  std::string radial_error;
  bool operator==(const Metadata &) const = default;
};
struct Source {
  std::shared_ptr<File> file;
  FileState initial;
  std::string hash;
  explicit Source(const fs::path &path);
  void check() const;
  // Reads the whole input once. The first read records its hash; later reads
  // must reproduce it. Callers parse and decode the returned hashed bytes.
  std::shared_ptr<Memory> read(const Check &check_cancel = {});
  void verify(const Check &check_cancel = {}) const;
};
Metadata read_metadata(const Directory &root);
std::pair<Plane, Metadata> read_frame(Source &source, const Check &check = {});
void verify_rgb(const fs::path &path, const Rgb &expected, const Check &check = {});
const char *decoder_version();
void validate_sequence(std::span<const Metadata> metadata, bool radial);
// Explicit exposure count; every consecutive block remains a complete sequence.
// This validates observed fields, not an undocumented Nikon bracket-program tag.
void validate_brackets(std::span<const Metadata> metadata, unsigned count, bool radial);
} // namespace cnefx::detail
