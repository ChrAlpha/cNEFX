#include "cnefx/merge.hpp"
#include "nefx.hpp"

#include <algorithm>
#include <set>

#ifndef CNEFX_VERSION
#define CNEFX_VERSION "unknown"
#endif

namespace cnefx {
using namespace detail;
const char *version() { return CNEFX_VERSION; }

MergeReport merge(std::span<const fs::path> paths, const MergeOptions &options, const Check &check,
                  const Progress &progress) {
  if (paths.size() > 32)
    throw std::invalid_argument("Expected at most 32 ordered input captures.");
  const auto groups = output_groups(paths.size(), options.outputs);
  if (options.output.empty())
    throw std::invalid_argument("An output destination is required.");
  if (!options.directory) {
    auto extension = options.output.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [](unsigned char c) { return c >= 'A' && c <= 'Z' ? char(c + 32) : char(c); });
    if (options.outputs != 1 || extension != ".nefx")
      throw std::invalid_argument("Single output requires one output and a .nefx extension.");
  }
  const auto output = fs::absolute(options.output);
  if (path_exists(output))
    throw std::invalid_argument("Output already exists; overwriting is not supported.");
  if (check)
    check();
  std::vector<Source> sources;
  sources.reserve(paths.size());
  std::set<std::pair<std::uint64_t, std::uint64_t>> identities;
  for (const auto &path : paths) {
    if (check)
      check();
    sources.emplace_back(path);
    const auto &state = sources.back().initial;
    if (!identities.emplace(state.device, state.inode).second)
      throw std::invalid_argument("Duplicate input file or hard link.");
  }
  // Validate the complete sequence from its TIFF metadata before any decode.
  std::vector<Metadata> preflight;
  preflight.reserve(sources.size());
  for (const auto &source : sources) {
    if (check)
      check();
    preflight.push_back(read_metadata(read_tiff(source.file)));
  }
  validate_sequence(preflight, options.chromatic_aberration_correction);
  const auto &information = preflight.front();
  if (progress)
    progress(Stage::preflight, paths.size(), paths.size());
  Publication pending(output, options.directory);
  std::vector<bool> consumed(paths.size());
  std::optional<Calibration> calibration;
  unsigned decoded = 0;
  auto load = [&](unsigned index) {
    if (consumed.at(index))
      throw std::logic_error("Capture decoded more than once.");
    auto frame = read_frame(sources.at(index), check);
    // Decoded bytes must carry the validated metadata; LibRaw adds calibration.
    auto expected = preflight[index];
    expected.sequence.calibration = frame.second.sequence.calibration;
    if (frame.second != expected)
      throw std::invalid_argument("Input metadata changed after preflight.");
    if (!calibration)
      calibration = frame.second.sequence.calibration;
    else if (*calibration != frame.second.sequence.calibration)
      throw std::invalid_argument("Sequence calibration differs.");
    consumed[index] = true;
    ++decoded;
    if (progress)
      progress(Stage::decode, decoded, paths.size());
    return std::move(frame.first);
  };
  MergeReport report;
  report.make = information.sequence.make;
  report.model = information.sequence.model;
  report.decoder_version = decoder_version();
  report.options = options;
  report.options.output = output;
  for (unsigned index = 0; index < groups.size(); ++index) {
    const auto &members = groups[index];
    auto pixels = reconstruct(
        members.size(),
        [&](std::span<const unsigned> batch) {
          std::vector<Plane> planes;
          for (const auto i : batch)
            planes.push_back(load(members[i]));
          return planes;
        },
        information.sequence.crop, Selection::parse(information.sequence.selector),
        information.sequence.iso,
        options.chromatic_aberration_correction ? information.radial : std::nullopt, check);
    const auto name =
        "merged-" + std::string(index < 9 ? "0" : "") + std::to_string(index + 1) + ".nefx";
    const auto candidate = options.directory ? pending.temporary() / name : pending.temporary();
    write_nefx(sources[members[0]], candidate, pixels, members.size(), check);
    verify_rgb(candidate, pixels, check);
    File file(candidate);
    report.outputs.push_back({options.directory ? output / name : output, sha256(file, check),
                              pixels.width, pixels.height, pixels.pixels.size(), members});
    if (progress)
      progress(Stage::write, index + 1, groups.size());
  }
  if (decoded != paths.size())
    throw std::logic_error("Every capture must be decoded exactly once.");
  report.limitations = {
      "Experimental: authentic pixel-shift captures and Lightroom are not validated.",
      "Adobe XMP defaults are preserved without updates.",
      "HE/HE* decoding is not implemented; use lossless-compressed 14-bit NEF inputs.",
      "Input order must follow consecutive capture numbers in a single sequence."};
  if (calibration->missing_matrix)
    report.limitations.push_back(
        "LibRaw has no XYZ-to-camera matrix for this input; NEFX preserves source color metadata. "
        "Color rendering is unvalidated.");
  if (!options.chromatic_aberration_correction)
    report.limitations.push_back("Chromatic aberration correction is disabled.");
  for (unsigned i = 0; i < sources.size(); ++i) {
    sources[i].verify(check);
    report.inputs.push_back({sources[i].file->path(), sources[i].hash, preflight[i].shot});
    if (progress)
      progress(Stage::verify, i + 1, sources.size());
  }
  if (options.directory) {
    const auto manifest = report_json(report) + "\n";
    File file(pending.temporary() / "manifest.json", FileMode::create);
    file.write(0,
               std::span(reinterpret_cast<const std::uint8_t *>(manifest.data()), manifest.size()));
    file.sync();
  }
  if (progress)
    progress(Stage::publish, 0, 1);
  pending.publish([&] {
    if (check)
      check();
    for (const auto &source : sources)
      source.check();
  });
  pending.cleanup();
  report.warnings = pending.warnings();
  return report;
}
} // namespace cnefx
