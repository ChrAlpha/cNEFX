#pragma once

#include "core.hpp"

#include <filesystem>
#include <string>

namespace cnefx {
class Cancelled : public std::runtime_error {
public:
  Cancelled() : std::runtime_error("Synthesis cancelled.") {}
};
enum class Stage { preflight, decode, write, verify, publish };
using Progress = std::function<void(Stage, unsigned completed, unsigned total)>;
struct MergeOptions {
  // Empty writes each output beside its sequence's first capture as
  // <stem>_merged.NEFX (<stem>_merged_NN.NEFX for several outputs). A file
  // requires exactly one output; a directory receives every output at once.
  std::filesystem::path output;
  // Outputs per pixel-shift sequence.
  unsigned outputs = 1;
  bool directory = false, chromatic_aberration_correction = false;
  // Worker threads for decoding a batch of up to four captures and for radial
  // correction; zero selects the hardware concurrency. Outputs do not depend on
  // it; one keeps all work on the caller's thread.
  unsigned jobs = 0;
};
struct InputReport {
  std::filesystem::path path;
  std::string sha256;
  std::uint32_t shot_number{};
  unsigned sequence_index{};
};
// A path that is not a pixel-shift capture, or one that could not be read.
struct FileReport {
  std::filesystem::path path;
  std::string reason;
};
enum class SequenceStatus { merged, already_merged, failed };
struct SequenceReport {
  std::string make, model;
  std::filesystem::path first_capture;
  unsigned captures{};
  SequenceStatus status{};
  // Outputs of an earlier run; the reason for failed.
  std::vector<std::filesystem::path> existing;
  std::string error;
};
struct OutputReport {
  std::filesystem::path path;
  std::string sha256;
  std::uint32_t width{}, height{};
  std::uint64_t samples_verified{};
  std::vector<unsigned> input_indices;
  unsigned sequence_index{};
};
struct MergeReport {
  std::string decoder_version;
  MergeOptions options;
  // Sequences by first appearance; captures decoded and outputs published by
  // this run, including those of a sequence that failed after publishing some.
  std::vector<SequenceReport> sequences;
  std::vector<InputReport> inputs;
  std::vector<OutputReport> outputs;
  std::vector<FileReport> skipped, errors;
  std::vector<std::string> limitations, warnings;
  // False when a sequence failed or a file could not be read; other sequences
  // are still merged and published.
  bool complete() const;
};

// Callbacks run synchronously on the caller's thread. Throw Cancelled from check
// to cancel; outputs already published stay. Inputs are NEF files, directories
// (their immediate .NEF files) or wildcard file-name patterns. Captures without
// active pixel-shift metadata and merged NEFX files are skipped; the rest are
// grouped by camera and burst identity and ordered by shot number, as NX Studio
// does. Without an output directory each sequence is synthesized and published
// on its own; a sequence whose outputs already exist is not merged again.
MergeReport merge(std::span<const std::filesystem::path> inputs, const MergeOptions &options,
                  const Check &check = {}, const Progress &progress = {});
std::string report_json(const MergeReport &report);
std::string error_json(const std::string &code, const std::string &message);
std::string path_utf8(const std::filesystem::path &path);
const char *version();
} // namespace cnefx
