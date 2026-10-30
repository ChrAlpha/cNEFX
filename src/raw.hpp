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
  // Nikon HE and HE* store a TicoRAW codestream that LibRaw cannot unpack.
  bool high_efficiency() const { return compression == 13 || compression == 14; }
  bool operator==(const SequenceMetadata &) const = default;
};
struct Metadata {
  SequenceMetadata sequence;
  std::uint32_t shot{};
  // Offset and length of the single RAW strip within the source bytes.
  std::pair<std::uint64_t, std::uint64_t> strip{};
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
// Null for a source capture with active pixel-shift metadata, else the reason to
// skip the file.
const char *skip_reason(const Directory &root);
// Camera and burst identity of a merged NEFX, used to recognize earlier outputs.
std::optional<std::pair<std::string, std::uint32_t>> merged_identity(const Directory &root);
// Indices grouped by camera and burst identity in first-appearance order, each
// ordered by shot number. Callers validate every group with validate_sequence.
std::vector<std::vector<unsigned>> group_sequences(std::span<const Metadata> metadata);
} // namespace cnefx::detail
