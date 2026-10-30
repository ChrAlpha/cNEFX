#include "cnefx/merge.hpp"

#include <atomic>
#include <charconv>
#include <csignal>
#include <iostream>
#include <string_view>

#ifdef _WIN32
#include <cstdio>
#else
#include <algorithm>
#include <climits>
#include <sys/resource.h>
#endif

namespace {
// Worker threads may receive the signal; a lock-free atomic is safe in a
// handler and visible to the caller's thread.
std::atomic<bool> interrupted{false};
static_assert(std::atomic<bool>::is_always_lock_free);
extern "C" void signal_handler(int) { interrupted.store(true, std::memory_order_relaxed); }
// Every capture of a run stays open until publication, so a folder of several
// sequences needs more descriptors than the default soft limit on some systems.
void raise_file_limit() {
#ifdef _WIN32
  _setmaxstdio(8192);
#else
  rlimit limit{};
  if (getrlimit(RLIMIT_NOFILE, &limit) || limit.rlim_cur >= limit.rlim_max)
    return;
  limit.rlim_cur = limit.rlim_max;
#ifdef __APPLE__
  limit.rlim_cur = std::min<rlim_t>(limit.rlim_max, OPEN_MAX);
#endif
  setrlimit(RLIMIT_NOFILE, &limit);
#endif
}
std::filesystem::path from_utf8(const std::string &text) {
  return std::filesystem::path(std::u8string(text.begin(), text.end()));
}
void help() {
  std::cout << "cNEFX " << cnefx::version()
            << " - native Nikon pixel-shift NEFX synthesis\n\n"
               "Usage: cnefx merge PATH... [--outputs N] [--output FILE.nefx | --output-directory "
               "DIR]\n\n"
               "PATH is a NEF file or a directory whose immediate .NEF files are used. Captures\n"
               "without pixel-shift metadata are skipped; the rest are grouped into sequences by\n"
               "burst identity and ordered by shot number. Each complete sequence is merged;\n"
               "AE-bracketed exposures are separate sequences. By default each output is written\n"
               "beside its first capture as NAME_merged.NEFX. Destinations must not exist.\n"
               "Modes: 4->1, 8->1/2, 16->1/4, 32->1/2/4/8.\n\n"
               "Options:\n"
               "  -o, --output FILE.nefx             Write the only output to this file\n"
               "  --output-directory DIR             Publish all outputs and a manifest together\n"
               "  --outputs N                        Outputs per sequence (default 1)\n"
               "  --jobs N                           Worker threads, 1-256 (default: all cores)\n"
               "  --chromatic-aberration-correction  Apply supported source radial correction\n"
               "  --json                             Write a JSON report to stdout\n"
               "  --                                 Treat remaining arguments as input paths\n"
               "  -h, --help                         Show this help\n"
               "  --version                          Show the version\n\n"
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
    bool positional = false, have_output = false, have_count = false, have_jobs = false;
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
                          arg == "--outputs" || arg == "--jobs")) {
        if (i + 1 >= args.size() || args[i + 1].empty() || args[i + 1].starts_with("--"))
          throw std::invalid_argument("Missing value for " + arg + ".");
        const auto &value = args[++i];
        if (arg == "--jobs") {
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
    if (inputs.empty())
      throw std::invalid_argument("Input files or directories are required.");
    if (have_output && !options.directory && options.outputs != 1)
      throw std::invalid_argument("--output writes one file; use --output-directory or omit it.");
  } catch (const std::exception &e) {
    std::cerr << "cnefx: " << e.what() << '\n';
    return 2;
  }
  raise_file_limit();
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
      for (const auto &skipped : report.skipped)
        std::cerr << "Skipped " << cnefx::path_utf8(skipped.path) << ": " << skipped.reason << '\n';
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
