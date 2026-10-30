#include "cnefx/merge.hpp"
#include "discover.hpp"
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
std::u8string number(unsigned value) {
  const auto digits = std::to_string(value);
  return std::u8string(value < 10 ? u8"0" : u8"") + std::u8string(digits.begin(), digits.end());
}
struct Plan {
  std::vector<unsigned> captures; // Discovery indices in shot order.
  Groups groups;                  // Positions within captures.
  std::vector<fs::path> destinations;
  std::vector<unsigned> pending; // Outputs to synthesize.
};
} // namespace
const char *version() { return CNEFX_VERSION; }
bool MergeReport::complete() const {
  return errors.empty() && std::none_of(sequences.begin(), sequences.end(), [](const auto &item) {
           return item.status == SequenceStatus::failed;
         });
}

MergeReport merge(std::span<const fs::path> inputs, const MergeOptions &options, const Check &check,
                  const Progress &progress) {
  if (inputs.empty())
    throw std::invalid_argument("Expected input captures.");
  if (!options.outputs)
    throw std::invalid_argument("Expected at least one output per sequence.");
  if (options.directory && options.output.empty())
    throw std::invalid_argument("An output directory path is required.");
  const bool single = !options.directory && !options.output.empty();
  if (single && (options.outputs != 1 || folded(options.output.extension()) != u8".nefx"))
    throw std::invalid_argument("Single output requires one output and a .nefx extension.");
  if (options.directory && path_exists(fs::absolute(options.output)))
    throw std::invalid_argument("Output already exists; overwriting is not supported.");
  if (check)
    check();
  MergeReport report;
  report.decoder_version = decoder_version();
  report.options = options;
  if (!options.output.empty())
    report.options.output = fs::absolute(options.output);
  auto found = discover(inputs, check);
  report.skipped = std::move(found.skipped);
  report.errors = std::move(found.errors);
  if (found.captures.empty())
    throw std::invalid_argument(report.errors.empty() ? "No pixel-shift captures found."
                                                      : path_utf8(report.errors.front().path) +
                                                            ": " + report.errors.front().reason);
  // Plan every sequence from metadata before any decode.
  std::vector<Metadata> metadata;
  for (const auto &capture : found.captures)
    metadata.push_back(capture.metadata);
  std::vector<Plan> plans;
  std::set<std::u8string> names;
  for (const auto &group : group_sequences(metadata)) {
    const auto &first = found.captures[group.front()];
    auto &sequence = report.sequences.emplace_back();
    sequence.make = first.metadata.sequence.make;
    sequence.model = first.metadata.sequence.model;
    sequence.first_capture = first.path;
    sequence.captures = static_cast<unsigned>(group.size());
    auto &plan = plans.emplace_back(Plan{group, {}, {}, {}});
    try {
      plan.groups = output_groups(first.metadata.sequence.count, options.outputs);
      for (unsigned output = 0; output < plan.groups.size(); ++output) {
        const auto name = fs::path(
            first.path.stem().u8string() + u8"_merged" +
            (options.outputs > 1 ? u8"_" + number(output + 1) : std::u8string()) + u8".NEFX");
        plan.destinations.push_back(single              ? report.options.output
                                    : options.directory ? report.options.output / name
                                                        : first.path.parent_path() / name);
        // Case-insensitive filesystems would otherwise collide only when writing.
        if (!names.insert(folded(plan.destinations.back())).second)
          throw std::invalid_argument("Outputs would share the name " + path_utf8(name) + ".");
      }
      // Outputs of an earlier run, by cNEFX or NX Studio, count as done, so an
      // interrupted sequence resumes with its missing outputs. Anything else at a
      // destination is never replaced.
      const std::pair identity(first.metadata.sequence.identity_hash,
                               first.metadata.sequence.burst);
      const auto earlier = [&](const fs::path &path) {
        try {
          return merged_identity(read_tiff(std::make_shared<File>(path))) == identity;
        } catch (const std::exception &) {
          return false;
        }
      };
      for (unsigned output = 0; output < plan.destinations.size(); ++output) {
        const auto &destination = plan.destinations[output];
        if (!path_exists(destination))
          plan.pending.push_back(output);
        else if (earlier(destination))
          sequence.existing.push_back(destination);
        else
          throw std::invalid_argument("Output exists but is not this sequence's merged NEFX: " +
                                      path_utf8(destination));
      }
      if (plan.pending.empty()) {
        sequence.status = SequenceStatus::already_merged;
        continue;
      }
      std::vector<Metadata> ordered;
      for (const auto index : group)
        ordered.push_back(metadata[index]);
      validate_sequence(ordered, options.chromatic_aberration_correction);
    } catch (const std::exception &e) {
      sequence.status = SequenceStatus::failed;
      sequence.error = e.what();
    }
  }
  if (single && report.sequences.size() != 1)
    throw std::invalid_argument("Inputs contain " + std::to_string(report.sequences.size()) +
                                " pixel-shift sequences; a single output file needs one.");
  // An output set holds every sequence or none.
  if (options.directory && !report.complete()) {
    for (const auto &sequence : report.sequences)
      if (sequence.status == SequenceStatus::failed)
        throw std::invalid_argument("Sequence starting at " + path_utf8(sequence.first_capture) +
                                    ": " + sequence.error);
    throw std::invalid_argument(path_utf8(report.errors.front().path) + ": " +
                                report.errors.front().reason);
  }
  unsigned captures = 0, outputs = 0;
  for (unsigned i = 0; i < plans.size(); ++i)
    if (report.sequences[i].status == SequenceStatus::merged) {
      for (const auto output : plans[i].pending) {
        captures += plans[i].groups[output].size();
        ++outputs;
      }
    }
  if (progress)
    progress(Stage::preflight, found.captures.size(), found.captures.size());
  // A directory publishes every output at once. Otherwise each sequence is opened,
  // synthesized, verified and published on its own, so staging space and open
  // files are bounded by one sequence.
  const auto set =
      options.directory ? std::make_unique<Publication>(report.options.output, true) : nullptr;
  std::vector<std::pair<fs::path, FileState>> retained;
  unsigned decoded = 0, written = 0, verified = 0, published = 0;
  bool missing_matrix = false;
  const auto synthesize = [&](unsigned index) {
    const auto &plan = plans[index];
    std::vector<Source> sources;
    std::vector<Metadata> preflight;
    for (const auto capture : plan.captures) {
      if (check)
        check();
      sources.emplace_back(found.captures[capture].path);
      if (sources.back().initial != found.captures[capture].state)
        throw std::invalid_argument("Input changed after discovery: " +
                                    path_utf8(found.captures[capture].path));
      preflight.push_back(found.captures[capture].metadata);
    }
    std::vector<std::unique_ptr<Publication>> files;
    if (!set)
      for (const auto output : plan.pending)
        files.push_back(std::make_unique<Publication>(plan.destinations[output], false));
    std::vector<bool> consumed(sources.size());
    std::optional<Calibration> calibration;
    auto accept = [&](unsigned position, std::pair<Plane, Metadata> frame) {
      // Decoded bytes must carry the validated metadata; LibRaw adds calibration.
      auto expected = preflight[position];
      expected.sequence.calibration = frame.second.sequence.calibration;
      if (frame.second != expected)
        throw std::invalid_argument("Input metadata changed after preflight.");
      if (!calibration)
        calibration = frame.second.sequence.calibration;
      else if (*calibration != frame.second.sequence.calibration)
        throw std::invalid_argument("Sequence calibration differs.");
      if (progress)
        progress(Stage::decode, ++decoded, captures);
      return std::move(frame.first);
    };
    // Decode a batch concurrently. Claims, validation and progress stay on the
    // caller's thread, in capture order.
    auto load = [&](const std::vector<unsigned> &positions) {
      for (const auto position : positions) {
        if (consumed.at(position))
          throw std::logic_error("Capture decoded more than once.");
        consumed[position] = true;
      }
      std::vector<std::pair<Plane, Metadata>> frames(positions.size());
      const auto count = static_cast<unsigned>(positions.size());
      parallel_for(count, options.jobs, check, [&](unsigned i, const Check &poll) {
        frames[i] = read_frame(sources.at(positions[i]), poll);
      });
      std::vector<Plane> planes;
      planes.reserve(frames.size());
      for (unsigned i = 0; i < frames.size(); ++i)
        planes.push_back(accept(positions[i], std::move(frames[i])));
      return planes;
    };
    std::vector<OutputReport> results;
    for (unsigned item = 0; item < plan.pending.size(); ++item) {
      const auto &members = plan.groups[plan.pending[item]];
      const auto &first = preflight[members[0]];
      auto pixels = reconstruct(
          members.size(),
          [&](std::span<const unsigned> batch) {
            std::vector<unsigned> positions;
            for (const auto i : batch)
              positions.push_back(members.at(i));
            return load(positions);
          },
          first.sequence.crop, first.sequence.selection, first.sequence.iso,
          options.chromatic_aberration_correction ? first.radial : std::nullopt, check,
          options.jobs);
      const auto &destination = plan.destinations[plan.pending[item]];
      const auto candidate =
          set ? set->temporary() / destination.filename() : files[item]->temporary();
      write_nefx(sources[members[0]], candidate, pixels, members.size(), check);
      verify_rgb(candidate, pixels, check);
      File file(candidate);
      results.push_back({destination, sha256(file, check), pixels.width, pixels.height,
                         pixels.pixels.size(), members, index});
      if (progress)
        progress(Stage::write, ++written, outputs);
    }
    // Report the decoded captures; output indices refer to that list.
    std::vector<InputReport> used;
    std::vector<unsigned> positions(sources.size());
    for (unsigned i = 0; i < sources.size(); ++i) {
      if (!consumed[i])
        continue;
      sources[i].verify(check);
      positions[i] = static_cast<unsigned>(report.inputs.size() + used.size());
      used.push_back({sources[i].file->path(), sources[i].hash, preflight[i].shot, index});
      if (progress)
        progress(Stage::verify, ++verified, captures);
    }
    std::size_t needed = 0;
    for (auto &result : results) {
      needed += result.input_indices.size();
      for (auto &i : result.input_indices)
        i = positions[i];
    }
    if (used.size() != needed)
      throw std::logic_error("Every needed capture must be decoded exactly once.");
    missing_matrix |= calibration->missing_matrix;
    const auto commit = [&](std::size_t count) {
      report.inputs.insert(report.inputs.end(), used.begin(), used.end());
      report.outputs.insert(report.outputs.end(), results.begin(),
                            results.begin() + std::ptrdiff_t(count));
    };
    if (set) {
      for (const auto &source : sources)
        retained.emplace_back(source.file->path(), source.initial);
      commit(results.size());
      return;
    }
    for (std::size_t i = 0; i < files.size(); ++i) {
      try {
        if (progress)
          progress(Stage::publish, published, outputs);
        files[i]->publish([&] {
          // Cancellation can stop a sequence only before its first output appears.
          if (check && !i)
            check();
          for (const auto &source : sources)
            source.check();
        });
      } catch (const std::exception &e) {
        if (!i)
          throw;
        for (std::size_t j = 0; j < i; ++j) {
          files[j]->cleanup();
          report.warnings.insert(report.warnings.end(), files[j]->warnings().begin(),
                                 files[j]->warnings().end());
        }
        commit(i);
        std::string paths;
        for (std::size_t j = 0; j < i; ++j)
          paths += (j ? ", " : "") + path_utf8(results[j].path);
        throw std::runtime_error(std::to_string(i) + " of " + std::to_string(files.size()) +
                                 " outputs were published and remain valid (" + paths +
                                 "); publication then failed: " + e.what());
      }
      ++published;
    }
    for (auto &file : files) {
      file->cleanup();
      report.warnings.insert(report.warnings.end(), file->warnings().begin(),
                             file->warnings().end());
    }
    commit(results.size());
  };
  for (unsigned index = 0; index < plans.size(); ++index) {
    auto &sequence = report.sequences[index];
    if (sequence.status != SequenceStatus::merged)
      continue;
    try {
      synthesize(index);
    } catch (const Cancelled &) {
      throw;
    } catch (const std::exception &e) {
      if (set)
        throw;
      sequence.status = SequenceStatus::failed;
      sequence.error = e.what();
    }
  }
  report.limitations = {
      "Experimental: authentic capture and Lightroom validation is partial.",
      "Adobe XMP defaults are preserved without updates.",
      "HE/HE* decoding is not implemented; use lossless-compressed 14-bit NEF inputs.",
      "Sequences are grouped by Nikon burst identity and ordered by shot number; each must be "
      "complete."};
  if (missing_matrix)
    report.limitations.push_back(
        "LibRaw has no XYZ-to-camera matrix for this input; NEFX preserves source color metadata. "
        "Color rendering is unvalidated.");
  if (!options.chromatic_aberration_correction)
    report.limitations.push_back("Chromatic aberration correction is disabled.");
  if (set) {
    const auto manifest = report_json(report) + "\n";
    File file(set->temporary() / "manifest.json", FileMode::create);
    file.write(0,
               std::span(reinterpret_cast<const std::uint8_t *>(manifest.data()), manifest.size()));
    file.sync();
    if (progress)
      progress(Stage::publish, 0, outputs);
    set->publish([&] {
      if (check)
        check();
      for (const auto &[path, state] : retained)
        if (file_state(path) != state)
          throw std::invalid_argument("Input changed during synthesis.");
    });
    if (progress)
      progress(Stage::publish, outputs, outputs);
    set->cleanup();
    report.warnings.insert(report.warnings.end(), set->warnings().begin(), set->warnings().end());
  }
  return report;
}
} // namespace cnefx
