#include "cnefx/merge.hpp"
#include "io.hpp"

#include <iostream>
#include <set>
#include <utility>

using namespace cnefx;
using namespace cnefx::detail;
namespace {
void require(bool ok) {
  if (!ok)
    throw std::runtime_error("Merge test failed.");
}
void mutate(const fs::path &path, bool restore) {
  const auto time = fs::last_write_time(path);
  File file(path, FileMode::update);
  const auto byte = file.read(file.size() - 1, 1);
  file.write(file.size() - 1, Bytes{static_cast<std::uint8_t>(byte[0] ^ 1)});
  file.sync();
  if (restore) {
    file.write(file.size() - 1, byte);
    file.sync();
    fs::last_write_time(path, time);
  }
}
// TARGET "-" writes each output beside its first capture. During publication
// race_last claims the last 4-frame sequence's output and race_second the second
// output of the first sequence.
void inject(int argc, char **argv) {
  const std::string mode = argv[2], target = argv[3];
  const bool beside = target == "-";
  MergeOptions options{beside ? fs::path() : fs::path(target),
                       static_cast<unsigned>(std::stoul(argv[4])), !beside, false};
  std::vector<fs::path> paths(argv + 5, argv + argc);
  const auto folder = paths.front().parent_path();
  const auto last =
      mode == "race_second"
          ? folder / (paths.front().stem().string() + "_merged_02.NEFX")
          : paths.back().parent_path() / (paths[paths.size() - 4].stem().string() + "_merged.NEFX");
  bool fired = false, cancel = false, rejected = false, progressed = false;
  MergeReport report;
  if (mode == "incomplete") {
    fired = true;
    paths.pop_back();
  }
  try {
    report = merge(
        paths, options,
        [&] {
          if (cancel)
            throw Cancelled();
        },
        [&](Stage stage, unsigned done, unsigned) {
          progressed = true;
          if (fired)
            return;
          if (mode == "preflight" && stage == Stage::preflight) {
            fired = true;
            mutate(paths.back(), false);
          }
          if (stage == Stage::decode && done == 1) {
            if (mode == "unread") {
              fired = true;
              mutate(paths.back(), false);
            }
            if (mode == "replacement") {
              fired = true;
              const auto time = fs::last_write_time(paths.back());
              Bytes bytes;
              {
                File file(paths.back());
                bytes = file.read(0, file.size());
              }
              fs::rename(paths.back(), paths.back().string() + ".backup");
              {
                File file(paths.back(), FileMode::create);
                file.write(0, bytes);
                file.sync();
              }
              fs::last_write_time(paths.back(), time);
            }
            if (mode == "cancel_decode") {
              fired = true;
              cancel = true;
            }
          }
          if (stage == Stage::write && done == 1) {
            if (mode == "write_failure") {
              fired = true;
              throw std::runtime_error("Injected output failure.");
            }
            if (mode == "after_write") {
              fired = true;
              mutate(paths.front(), false);
            }
            if (mode == "cancel") {
              fired = true;
              cancel = true;
            }
          }
          // The first sequence is published before the second one is written.
          if (stage == Stage::write && done == 2 && mode == "cancel_later") {
            fired = true;
            cancel = true;
          }
          if (stage == Stage::verify && done == (mode == "late_hash" ? 2u : 1u) &&
              (mode == "restored" || mode == "late_hash")) {
            fired = true;
            mutate(paths.front(), true);
          }
          if (stage == Stage::publish && mode == "target_race") {
            fired = true;
            fs::create_directory(options.output);
          }
          if (stage == Stage::publish && (mode == "race_last" || mode == "race_second")) {
            fired = true;
            File(last, FileMode::create).sync();
          }
        });
  } catch (const Cancelled &) {
    require(mode == "cancel" || mode == "cancel_decode" || mode == "cancel_later");
    rejected = true;
  } catch (const std::exception &) {
    require(mode != "cancel" && mode != "cancel_decode" && mode != "cancel_later");
    rejected = true;
  }
  // Beside the captures each sequence stands alone: one failure leaves the others
  // merged and the run reports partial completion instead of throwing.
  const bool partial = beside && (mode == "write_failure" || mode == "race_last" ||
                                  mode == "race_second" || mode == "preflight");
  require(fired && rejected != partial);
  // Sequence completeness is metadata-only; it must fail before preflight completes.
  require(mode != "incomplete" || !progressed);
  const auto beside_output = [](const fs::path &first) {
    return first.parent_path() / (first.stem().string() + "_merged.NEFX");
  };
  if (mode == "race_second") {
    // The first output of the failed sequence stays published and is reported.
    const auto &first = report.sequences.front();
    require(first.status == SequenceStatus::failed &&
            first.error.find("1 of 2 outputs were published") != std::string::npos &&
            report.outputs.front().path ==
                folder / (paths.front().stem().string() + "_merged_01.NEFX") &&
            fs::file_size(last) == 0 && !report.complete());
  } else if (partial) {
    // write_failure fails the first sequence; preflight changes the last capture.
    const unsigned failed = mode == "write_failure" ? 0 : 2;
    require(!report.complete() && report.outputs.size() == 2);
    for (unsigned sequence = 0; sequence < 3; ++sequence) {
      const bool merged = sequence != failed;
      require((report.sequences[sequence].status == SequenceStatus::merged) == merged);
      require(path_exists(beside_output(paths[4 * sequence])) == (merged || mode == "race_last"));
    }
    // The claimed name is untouched.
    require(mode != "race_last" || fs::file_size(last) == 0);
  } else if (mode == "cancel_later")
    // Sequences finished before cancellation stay published.
    require(path_exists(beside_output(paths[0])) && !path_exists(beside_output(paths[4])) &&
            !path_exists(beside_output(paths[8])));
  else if (beside)
    for (const auto &path : paths)
      require(!path_exists(beside_output(path)));
  else
    require(mode == "target_race" ? fs::is_directory(options.output) && fs::is_empty(options.output)
                                  : !path_exists(options.output));
  std::set<fs::path> folders{beside ? folder : options.output.parent_path()};
  for (const auto &path : paths)
    if (beside)
      folders.insert(path.parent_path());
  for (const auto &directory : folders)
    for (const auto &entry : fs::directory_iterator(directory))
      require(!entry.path().filename().string().starts_with(".cnefx-"));
}
void tests() {
  require(error_json("a\"b", "a\nb").find("a\\u000ab") != std::string::npos);
  require(error_json("error", "\xf0\x9f\x98\x80").find("\\ud83d\\ude00") != std::string::npos);
  require(error_json("error", "\xff").find("\\udcff") != std::string::npos);
  // Option and input errors precede any file access.
  for (const auto &[inputs, options] : std::vector<std::pair<std::vector<fs::path>, MergeOptions>>{
           {{}, {}},
           {{"absent.NEF"}, {"output.nefx", 2, false, false}},
           {{"absent.NEF"}, {"output.tif", 1, false, false}},
           {{"absent.NEF"}, {"", 1, true, false}},
           {{"absent.NEF"}, {"", 0, false, false}}}) {
    bool rejected = false;
    try {
      merge(inputs, options);
    } catch (const std::invalid_argument &) {
      rejected = true;
    }
    require(rejected);
  }
}
} // namespace
int main(int argc, char **argv) {
  try {
    if (argc == 1)
      tests();
    else if (argc >= 9 && std::string(argv[1]) == "--inject")
      inject(argc, argv);
    else
      return 2;
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
