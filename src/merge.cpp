#include "cnefx/merge.hpp"
#include "nefx.hpp"
#include "parallel.hpp"

#include <algorithm>
#include <memory>
#include <set>

#ifndef CNEFX_VERSION
#define CNEFX_VERSION "unknown"
#endif

namespace cnefx {
using namespace detail;
namespace {
std::u8string folded(const fs::path &path) {
  auto text = path.u8string();
  std::transform(text.begin(), text.end(), text.begin(),
                 [](char8_t c) { return c >= u8'A' && c <= u8'Z' ? char8_t(c + 32) : c; });
  return text;
}
bool has_extension(const fs::path &path, std::u8string_view expected) {
  return folded(path.extension()) == expected;
}
// Directories contribute their immediate .NEF files in path order; hidden files
// such as macOS ._ resource forks on exFAT cards are not captures.
std::vector<fs::path> expand(std::span<const fs::path> inputs) {
  std::vector<fs::path> paths;
  for (const auto &input : inputs) {
    if (!fs::is_directory(input)) {
      paths.push_back(input);
      continue;
    }
    std::vector<fs::path> entries;
    for (const auto &entry : fs::directory_iterator(input))
      if (entry.is_regular_file() && has_extension(entry.path(), u8".nef") &&
          !entry.path().filename().u8string().starts_with(u8'.'))
        entries.push_back(entry.path());
    std::sort(entries.begin(), entries.end());
    paths.insert(paths.end(), entries.begin(), entries.end());
  }
  return paths;
}
std::u8string number(unsigned value) {
  const auto digits = std::to_string(value);
  return std::u8string(value < 10 ? u8"0" : u8"") + std::u8string(digits.begin(), digits.end());
}
} // namespace
const char *version() { return CNEFX_VERSION; }

MergeReport merge(std::span<const fs::path> inputs, const MergeOptions &options, const Check &check,
                  const Progress &progress) {
  if (inputs.empty())
    throw std::invalid_argument("Expected input captures.");
  if (!options.outputs)
    throw std::invalid_argument("Expected at least one output per sequence.");
  if (options.directory && options.output.empty())
    throw std::invalid_argument("An output directory path is required.");
  const bool single = !options.directory && !options.output.empty();
  if (single && (options.outputs != 1 || !has_extension(options.output, u8".nefx")))
    throw std::invalid_argument("Single output requires one output and a .nefx extension.");
  if (options.directory && path_exists(fs::absolute(options.output)))
    throw std::invalid_argument("Output already exists; overwriting is not supported.");
  if (check)
    check();
  MergeReport report;
  // Classify and validate every capture from its TIFF metadata before any decode.
  // Only captures stay open, so skipped files do not count against file limits.
  std::vector<Source> captures;
  std::vector<Metadata> found;
  std::set<std::pair<std::uint64_t, std::uint64_t>> identities;
  for (const auto &path : expand(inputs)) {
    if (check)
      check();
    Source source(path);
    if (!identities.emplace(source.initial.device, source.initial.inode).second)
      throw std::invalid_argument("Duplicate input file or hard link.");
    const auto root = read_tiff(source.file);
    if (const auto reason = skip_reason(root)) {
      report.skipped.push_back({source.file->path(), reason});
      continue;
    }
    found.push_back(read_metadata(root));
    captures.push_back(std::move(source));
  }
  if (captures.empty())
    throw std::invalid_argument("No pixel-shift captures found.");
  // Synthesis order: sequences by first appearance, captures by shot number.
  std::vector<Source> sources;
  std::vector<Metadata> preflight;
  std::vector<std::pair<unsigned, unsigned>> sequences;
  std::vector<Groups> outputs;
  const auto context = [&](unsigned first) {
    return "Sequence starting at " + path_utf8(sources[first].file->path()) + ": ";
  };
  for (const auto &group : group_sequences(found)) {
    const auto first = static_cast<unsigned>(sources.size());
    for (const auto index : group) {
      sources.push_back(std::move(captures[index]));
      preflight.push_back(found[index]);
    }
    try {
      validate_sequence(std::span(preflight).subspan(first),
                        options.chromatic_aberration_correction);
      outputs.push_back(output_groups(group.size(), options.outputs));
    } catch (const std::invalid_argument &e) {
      throw std::invalid_argument(context(first) + e.what());
    }
    sequences.emplace_back(first, static_cast<unsigned>(group.size()));
  }
  // Each sequence produces its own outputs; groups hold synthesis-order indices
  // and placements the sequence and output number within it.
  Groups groups;
  std::vector<std::pair<unsigned, unsigned>> placements;
  std::vector<unsigned> owners(sources.size());
  for (unsigned sequence = 0; sequence < sequences.size(); ++sequence) {
    const auto [first, count] = sequences[sequence];
    std::fill_n(owners.begin() + first, count, sequence);
    const auto &local = outputs[sequence];
    for (unsigned output = 0; output < local.size(); ++output) {
      groups.push_back(local[output]);
      for (auto &member : groups.back())
        member += first;
      placements.emplace_back(sequence, output);
    }
  }
  if (single && groups.size() != 1)
    throw std::invalid_argument("Inputs contain " + std::to_string(sequences.size()) +
                                " pixel-shift sequences; a single output file needs one.");
  // Outputs are named after their sequence's first capture, as NX Studio does.
  std::vector<fs::path> destinations;
  std::set<std::u8string> unique;
  for (const auto &[sequence, output] : placements) {
    const auto &first = sources[sequences[sequence].first].file->path();
    const auto name =
        fs::path(first.stem().u8string() + u8"_merged" +
                 (options.outputs > 1 ? u8"_" + number(output + 1) : std::u8string()) + u8".NEFX");
    destinations.push_back(single              ? fs::absolute(options.output)
                           : options.directory ? fs::absolute(options.output) / name
                                               : first.parent_path() / name);
    // Case-insensitive filesystems would otherwise collide only when writing.
    if (!unique.insert(folded(destinations.back())).second)
      throw std::invalid_argument("Outputs would share the name " + path_utf8(name) + ".");
    if (path_exists(destinations.back()))
      throw std::invalid_argument("Output already exists; overwriting is not supported: " +
                                  path_utf8(destinations.back()));
  }
  if (progress)
    progress(Stage::preflight, sources.size(), sources.size());
  // A directory publishes every output at once; otherwise each output file is
  // staged beside its destination and published after all outputs are verified.
  std::vector<std::unique_ptr<Publication>> publications;
  if (options.directory)
    publications.push_back(std::make_unique<Publication>(fs::absolute(options.output), true));
  else
    for (const auto &destination : destinations)
      publications.push_back(std::make_unique<Publication>(destination, false));
  std::vector<bool> consumed(sources.size());
  std::vector<std::optional<Calibration>> calibrations(sequences.size());
  unsigned decoded = 0;
  auto accept = [&](unsigned index, std::pair<Plane, Metadata> frame) {
    // Decoded bytes must carry the validated metadata; LibRaw adds calibration.
    auto expected = preflight[index];
    expected.sequence.calibration = frame.second.sequence.calibration;
    if (frame.second != expected)
      throw std::invalid_argument("Input metadata changed after preflight.");
    auto &calibration = calibrations[owners[index]];
    if (!calibration)
      calibration = frame.second.sequence.calibration;
    else if (*calibration != frame.second.sequence.calibration)
      throw std::invalid_argument(context(sequences[owners[index]].first) +
                                  "Sequence calibration differs.");
    ++decoded;
    if (progress)
      progress(Stage::decode, decoded, sources.size());
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
  const auto &information = preflight.front();
  report.make = information.sequence.make;
  report.model = information.sequence.model;
  report.decoder_version = decoder_version();
  report.options = options;
  if (!report.options.output.empty())
    report.options.output = fs::absolute(options.output);
  report.sequence_count = static_cast<unsigned>(sequences.size());
  for (unsigned index = 0; index < groups.size(); ++index) {
    const auto &members = groups[index];
    const auto &first = preflight[members[0]];
    auto pixels = reconstruct(
        members.size(),
        [&](std::span<const unsigned> batch) {
          std::vector<unsigned> captures;
          for (const auto i : batch)
            captures.push_back(members.at(i));
          return load(captures);
        },
        first.sequence.crop, first.sequence.selection, first.sequence.iso,
        options.chromatic_aberration_correction ? first.radial : std::nullopt, check, options.jobs);
    const auto candidate = options.directory
                               ? publications.front()->temporary() / destinations[index].filename()
                               : publications[index]->temporary();
    write_nefx(sources[members[0]], candidate, pixels, members.size(), check);
    verify_rgb(candidate, pixels, check);
    File file(candidate);
    report.outputs.push_back({destinations[index], sha256(file, check), pixels.width, pixels.height,
                              pixels.pixels.size(), members, placements[index].first});
    if (progress)
      progress(Stage::write, index + 1, groups.size());
  }
  if (decoded != sources.size())
    throw std::logic_error("Every capture must be decoded exactly once.");
  report.limitations = {
      "Experimental: authentic capture and Lightroom validation is partial.",
      "Adobe XMP defaults are preserved without updates.",
      "HE/HE* decoding is not implemented; use lossless-compressed 14-bit NEF inputs.",
      "Sequences are grouped by Nikon burst identity and ordered by shot number; each must be "
      "complete."};
  if (std::any_of(calibrations.begin(), calibrations.end(),
                  [](const auto &calibration) { return calibration->missing_matrix; }))
    report.limitations.push_back(
        "LibRaw has no XYZ-to-camera matrix for this input; NEFX preserves source color metadata. "
        "Color rendering is unvalidated.");
  if (!options.chromatic_aberration_correction)
    report.limitations.push_back("Chromatic aberration correction is disabled.");
  for (unsigned i = 0; i < sources.size(); ++i) {
    sources[i].verify(check);
    report.inputs.push_back(
        {sources[i].file->path(), sources[i].hash, preflight[i].shot, owners[i]});
    if (progress)
      progress(Stage::verify, i + 1, sources.size());
  }
  if (options.directory) {
    const auto manifest = report_json(report) + "\n";
    File file(publications.front()->temporary() / "manifest.json", FileMode::create);
    file.write(0,
               std::span(reinterpret_cast<const std::uint8_t *>(manifest.data()), manifest.size()));
    file.sync();
  }
  if (progress)
    progress(Stage::publish, 0, publications.size());
  for (std::size_t i = 0; i < publications.size(); ++i) {
    try {
      publications[i]->publish([&] {
        // Cancellation can stop publication only before the first output appears.
        if (check && !i)
          check();
        for (const auto &source : sources)
          source.check();
      });
    } catch (const std::exception &e) {
      if (!i)
        throw;
      std::string published;
      for (std::size_t j = 0; j < i; ++j)
        published += (j ? ", " : "") + path_utf8(destinations[j]);
      throw std::runtime_error(std::to_string(i) + " of " + std::to_string(publications.size()) +
                               " outputs were published and remain valid (" + published +
                               "); publication then failed: " + e.what());
    }
    if (progress)
      progress(Stage::publish, i + 1, publications.size());
  }
  for (auto &publication : publications) {
    publication->cleanup();
    report.warnings.insert(report.warnings.end(), publication->warnings().begin(),
                           publication->warnings().end());
  }
  return report;
}
} // namespace cnefx
