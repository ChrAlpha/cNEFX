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
struct SkippedReport {
  std::filesystem::path path;
  std::string reason;
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
  std::string make, model, decoder_version;
  MergeOptions options;
  unsigned sequence_count{};
  // Inputs in synthesis order: sequences by first appearance, captures by shot.
  std::vector<InputReport> inputs;
  std::vector<SkippedReport> skipped;
  std::vector<OutputReport> outputs;
  std::vector<std::string> limitations, warnings;
};

// Callbacks run synchronously on the caller's thread. Throw Cancelled from check
// to cancel. Inputs are NEF files or directories, whose immediate .NEF files are
// considered. Captures without active pixel-shift metadata are skipped; the rest
// are grouped by Nikon burst identity and ordered by shot number, as NX Studio
// does. Every sequence must be complete, and inputs must stay unchanged.
MergeReport merge(std::span<const std::filesystem::path> inputs, const MergeOptions &options,
                  const Check &check = {}, const Progress &progress = {});
std::string report_json(const MergeReport &report);
std::string error_json(const std::string &code, const std::string &message);
std::string path_utf8(const std::filesystem::path &path);
const char *version();
} // namespace cnefx
