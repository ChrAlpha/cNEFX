#include "cnefx/merge.hpp"
#include "nefx.hpp"
#include "parallel.hpp"

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
  if (!options.brackets || options.brackets > 9)
    throw std::invalid_argument("Expected 1 to 9 exposure sequences.");
  if (options.brackets > 1 && !options.directory)
    throw std::invalid_argument("AE bracketing requires an output directory.");
  if (paths.size() > 32 * options.brackets || paths.size() % options.brackets)
    throw std::invalid_argument("Expected complete exposure sequences of at most 32 captures.");
  const unsigned count = paths.size() / options.brackets;
  const auto local_groups = output_groups(count, options.outputs);
  Groups groups;
  for (unsigned bracket = 0; bracket < options.brackets; ++bracket)
    for (auto members : local_groups) {
      for (auto &member : members)
        member += bracket * count;
      groups.push_back(std::move(members));
    }
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
  // Validate every exposure sequence from its TIFF metadata before any decode.
  std::vector<Metadata> preflight;
  preflight.reserve(sources.size());
  for (const auto &source : sources) {
    if (check)
      check();
    preflight.push_back(read_metadata(read_tiff(source.file)));
  }
  if (options.brackets == 1)
    validate_sequence(preflight, options.chromatic_aberration_correction);
  else
    validate_brackets(preflight, options.brackets, options.chromatic_aberration_correction);
  const auto &information = preflight.front();
  if (progress)
    progress(Stage::preflight, paths.size(), paths.size());
  Publication pending(output, options.directory);
  std::vector<bool> consumed(paths.size());
  std::vector<std::optional<Calibration>> calibrations(options.brackets);
  unsigned decoded = 0;
  auto accept = [&](unsigned index, std::pair<Plane, Metadata> frame) {
    // Decoded bytes must carry the validated metadata; LibRaw adds calibration.
    auto expected = preflight[index];
    expected.sequence.calibration = frame.second.sequence.calibration;
    if (frame.second != expected)
      throw std::invalid_argument("Input metadata changed after preflight.");
    auto &calibration = calibrations[index / count];
    if (!calibration)
      calibration = frame.second.sequence.calibration;
    else if (*calibration != frame.second.sequence.calibration)
      throw std::invalid_argument("Sequence calibration differs.");
    ++decoded;
    if (progress)
      progress(Stage::decode, decoded, paths.size());
    return std::move(frame.first);
  };
  // Decode a batch concurrently. Claims, validation and progress stay on the
  // caller's thread, in capture order.
  auto load = [&](const std::vector<unsigned> &indices) {
    for (const auto index : indices) {
      if (consumed.at(index))
        throw std::logic_error("Capture decoded more than once.");
      consumed[index] = true;
    }
    std::vector<std::pair<Plane, Metadata>> frames(indices.size());
    const auto count = static_cast<unsigned>(indices.size());
    parallel_for(count, options.jobs, check, [&](unsigned i, const Check &poll) {
      frames[i] = read_frame(sources.at(indices[i]), poll);
    });
    std::vector<Plane> planes;
    planes.reserve(frames.size());
    for (unsigned i = 0; i < frames.size(); ++i)
      planes.push_back(accept(indices[i], std::move(frames[i])));
    return planes;
  };
  MergeReport report;
  report.make = information.sequence.make;
  report.model = information.sequence.model;
  report.decoder_version = decoder_version();
  report.options = options;
  report.options.output = output;
  if (options.brackets > 1)
    for (unsigned bracket = 0; bracket < options.brackets; ++bracket) {
      const auto &sequence = preflight[bracket * count].sequence;
      BracketReport item{sequence.iso, sequence.exposure, {}};
      for (unsigned i = 0; i < count; ++i)
        item.input_indices.push_back(bracket * count + i);
      report.brackets.push_back(std::move(item));
    }
  for (unsigned index = 0; index < groups.size(); ++index) {
    const auto &members = groups[index];
    const auto &sequence = preflight[members[0]].sequence;
    const auto bracket = index / options.outputs;
    auto pixels = reconstruct(
        members.size(),
        [&](std::span<const unsigned> batch) {
          std::vector<unsigned> captures;
          for (const auto i : batch)
            captures.push_back(members.at(i));
          return load(captures);
        },
        sequence.crop, sequence.selection, sequence.iso,
        options.chromatic_aberration_correction ? preflight[members[0]].radial : std::nullopt,
        check, options.jobs);
    const auto name =
        (options.brackets > 1 ? "bracket-0" + std::to_string(bracket + 1) + "-" : "") + "merged-0" +
        std::to_string(index % options.outputs + 1) + ".nefx";
    const auto candidate = options.directory ? pending.temporary() / name : pending.temporary();
    write_nefx(sources[members[0]], candidate, pixels, members.size(), check);
    verify_rgb(candidate, pixels, check);
    File file(candidate);
    report.outputs.push_back({options.directory ? output / name : output, sha256(file, check),
                              pixels.width, pixels.height, pixels.pixels.size(), members, bracket});
    if (progress)
      progress(Stage::write, index + 1, groups.size());
  }
  if (decoded != paths.size())
    throw std::logic_error("Every capture must be decoded exactly once.");
  report.limitations = {
      "Experimental: authentic capture and Lightroom validation is partial.",
      "Adobe XMP defaults are preserved without updates.",
      "HE/HE* decoding is not implemented; use lossless-compressed 14-bit NEF inputs.",
      options.brackets > 1
          ? "AE brackets require complete exposure sequences in consecutive shutter order."
          : "Input order must follow consecutive capture numbers in a single sequence."};
  if (options.brackets > 1)
    report.limitations.push_back(
        "AE bracket grouping uses the supplied exposure count and observed metadata; authentic "
        "AE capture/reference and downstream HDR acceptance remain unvalidated.");
  if (std::any_of(calibrations.begin(), calibrations.end(),
                  [](const auto &calibration) { return calibration->missing_matrix; }))
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
