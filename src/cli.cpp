#include "cnefx/merge.hpp"

#include <atomic>
#include <charconv>
#include <csignal>
#include <iostream>
#include <string_view>

namespace {
// Worker threads may receive the signal; a lock-free atomic is safe in a
// handler and visible to the caller's thread.
std::atomic<bool> interrupted{false};
static_assert(std::atomic<bool>::is_always_lock_free);
extern "C" void signal_handler(int) { interrupted.store(true, std::memory_order_relaxed); }
std::filesystem::path from_utf8(const std::string &text) {
  return std::filesystem::path(std::u8string(text.begin(), text.end()));
}
void help() {
  std::cout
      << "cNEFX " << cnefx::version()
      << " - native Nikon pixel-shift NEFX synthesis\n\n"
         "Usage: cnefx merge FILE... --output FILE.nefx [--json]\n"
         "       cnefx merge FILE... --outputs N --output-directory DIR [--json]\n"
         "       cnefx merge FILE... --ae-bracketing N --output-directory DIR [--json]\n\n"
         "Pass complete sequences in explicit capture order. Destinations must not exist.\n"
         "Exposure groups are detected from metadata; multiple groups require a directory.\n"
         "Modes: 4->1, 8->1/2, 16->1/4, 32->1/2/4/8.\n\n"
         "Options:\n"
         "  -o, --output FILE.nefx             Write one verified NEFX\n"
         "  --output-directory DIR            Publish a complete output set\n"
         "  --outputs N                       Outputs per exposure sequence (default 1)\n"
         "  --ae-bracketing N                  Require 2-9 exposures (default: detect metadata)\n"
         "  --jobs N                          Worker threads, 1-256 (default: all cores)\n"
         "  --chromatic-aberration-correction  Apply supported source radial correction\n"
         "  --json                            Write a JSON report to stdout\n"
         "  --                                Treat remaining arguments as input paths\n"
         "  -h, --help                        Show this help\n"
         "  --version                         Show the version\n\n"
         "Experimental: authentic capture/reference and Lightroom acceptance remain open.\n";
}
int run(const std::vector<std::string> &args) {
  cnefx::MergeOptions options;
  std::vector<std::filesystem::path> inputs;
  bool json = false;
  try {
    if (args.empty()) {
      help();
      return 2;
    }
    if (args[0] == "--help" || args[0] == "-h") {
      help();
      return 0;
    }
    if (args[0] == "--version") {
      if (args.size() != 1)
        throw std::invalid_argument("Unexpected argument after --version.");
      std::cout << "cNEFX " << cnefx::version() << '\n';
      return 0;
    }
    if (args[0] != "merge")
      throw std::invalid_argument("Expected the merge command.");
    bool positional = false, have_output = false, have_count = false, have_jobs = false,
         have_brackets = false;
    for (std::size_t i = 1; i < args.size(); ++i) {
      const auto &arg = args[i];
      if (!positional && (arg == "--help" || arg == "-h")) {
        help();
        return 0;
      }
      if (!positional && arg == "--") {
        positional = true;
        continue;
      }
      if (!positional && arg == "--json") {
        json = true;
        continue;
      }
      if (!positional && arg == "--chromatic-aberration-correction") {
        options.chromatic_aberration_correction = true;
        continue;
      }
      if (!positional && (arg == "-o" || arg == "--output" || arg == "--output-directory" ||
                          arg == "--outputs" || arg == "--jobs" || arg == "--ae-bracketing")) {
        if (i + 1 >= args.size() || args[i + 1].empty() || args[i + 1].starts_with("--"))
          throw std::invalid_argument("Missing value for " + arg + ".");
        const auto &value = args[++i];
        if (arg == "--ae-bracketing") {
          if (have_brackets)
            throw std::invalid_argument("--ae-bracketing was specified more than once.");
          const auto result =
              std::from_chars(value.data(), value.data() + value.size(), options.brackets);
          if (result.ec != std::errc{} || result.ptr != value.data() + value.size() ||
              options.brackets < 2 || options.brackets > 9)
            throw std::invalid_argument("--ae-bracketing requires an integer from 2 to 9.");
          have_brackets = true;
        } else if (arg == "--jobs") {
          if (have_jobs)
            throw std::invalid_argument("--jobs was specified more than once.");
          const auto result =
              std::from_chars(value.data(), value.data() + value.size(), options.jobs);
          if (result.ec != std::errc{} || result.ptr != value.data() + value.size() ||
              !options.jobs || options.jobs > 256)
            throw std::invalid_argument("--jobs requires an integer from 1 to 256.");
          have_jobs = true;
        } else if (arg == "--outputs") {
          if (have_count)
            throw std::invalid_argument("--outputs was specified more than once.");
          const auto result =
              std::from_chars(value.data(), value.data() + value.size(), options.outputs);
          if (result.ec != std::errc{} || result.ptr != value.data() + value.size() ||
              !options.outputs)
            throw std::invalid_argument("--outputs requires a positive integer.");
          have_count = true;
        } else {
          if (have_output)
            throw std::invalid_argument("Specify exactly one output destination.");
          options.output = from_utf8(value);
          options.directory = arg == "--output-directory";
          have_output = true;
        }
        continue;
      }
      if (!positional && arg.starts_with('-'))
        throw std::invalid_argument("Unknown option: " + arg);
      inputs.push_back(from_utf8(arg));
    }
    if (!have_output || inputs.empty())
      throw std::invalid_argument("Input files and an output destination are required.");
    if (!options.directory && options.outputs != 1)
      throw std::invalid_argument("Multiple outputs require --output-directory.");
    if (have_brackets && !options.directory)
      throw std::invalid_argument("AE bracketing requires --output-directory.");
  } catch (const std::exception &e) {
    std::cerr << "cnefx: " << e.what() << '\n';
    return 2;
  }
  std::signal(SIGINT, signal_handler);
  std::signal(SIGTERM, signal_handler);
  try {
    const auto report = cnefx::merge(inputs, options, [] {
      if (interrupted)
        throw cnefx::Cancelled();
    });
    if (json)
      std::cout << cnefx::report_json(report) << '\n';
    else {
      for (const auto &output : report.outputs)
        std::cout << cnefx::path_utf8(output.path) << " (" << output.samples_verified
                  << " samples verified)\n";
      for (const auto &limitation : report.limitations)
        std::cerr << limitation << '\n';
      for (const auto &warning : report.warnings)
        std::cerr << "Warning: " << warning << '\n';
    }
    return 0;
  } catch (const cnefx::Cancelled &e) {
    if (json)
      std::cout << cnefx::error_json("cancelled", e.what()) << '\n';
    else
      std::cerr << e.what() << '\n';
    return 130;
  } catch (const std::exception &e) {
    if (json)
      std::cout << cnefx::error_json("merge_failed", e.what()) << '\n';
    else
      std::cerr << e.what() << '\n';
    return 1;
  }
}
} // namespace
#ifdef _WIN32
int wmain(int argc, wchar_t **argv) {
  std::vector<std::string> args;
  for (int i = 1; i < argc; ++i)
    args.push_back(cnefx::path_utf8(std::filesystem::path(argv[i])));
  return run(args);
}
#else
int main(int argc, char **argv) { return run(std::vector<std::string>(argv + 1, argv + argc)); }
#endif
