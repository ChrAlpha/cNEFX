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
  std::filesystem::path output;
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
};
struct OutputReport {
  std::filesystem::path path;
  std::string sha256;
  std::uint32_t width{}, height{};
  std::uint64_t samples_verified{};
  std::vector<unsigned> input_indices;
};
struct MergeReport {
  std::string make, model, decoder_version;
  MergeOptions options;
  std::vector<InputReport> inputs;
  std::vector<OutputReport> outputs;
  std::vector<std::string> limitations, warnings;
};

// Callbacks run synchronously on the caller's thread. Throw Cancelled from check
// to cancel. Input paths are explicit capture order and must stay unchanged.
MergeReport merge(std::span<const std::filesystem::path> inputs, const MergeOptions &options,
                  const Check &check = {}, const Progress &progress = {});
std::string report_json(const MergeReport &report);
std::string error_json(const std::string &code, const std::string &message);
std::string path_utf8(const std::filesystem::path &path);
const char *version();
} // namespace cnefx
