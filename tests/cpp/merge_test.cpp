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
// TARGET "-" writes each output beside its first capture; race_last then expects
// 4-frame sequences and claims the last sequence's destination during publication.
void inject(int argc, char **argv) {
  const std::string mode = argv[2], target = argv[3];
  const bool beside = target == "-";
  MergeOptions options{beside ? fs::path() : fs::path(target),
                       static_cast<unsigned>(std::stoul(argv[4])), !beside, false};
  std::vector<fs::path> paths(argv + 5, argv + argc);
  const auto folder = paths.front().parent_path();
  const auto last =
      paths.back().parent_path() / (paths[paths.size() - 4].stem().string() + "_merged.NEFX");
  bool fired = false, cancel = false, rejected = false, progressed = false;
  std::string message;
  if (mode == "incomplete") {
    fired = true;
    paths.pop_back();
  }
  try {
    merge(
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
          if (stage == Stage::verify && done == (mode == "late_hash" ? 2u : 1u) &&
              (mode == "restored" || mode == "late_hash")) {
            fired = true;
            mutate(paths.front(), true);
          }
          if (stage == Stage::publish && mode == "target_race") {
            fired = true;
            fs::create_directory(options.output);
          }
          if (stage == Stage::publish && mode == "race_last") {
            fired = true;
            File(last, FileMode::create).sync();
          }
        });
  } catch (const Cancelled &) {
    require(mode == "cancel" || mode == "cancel_decode");
    rejected = true;
  } catch (const std::exception &e) {
    require(mode != "cancel" && mode != "cancel_decode");
    rejected = true;
    message = e.what();
  }
  require(fired && rejected);
  // Sequence completeness is metadata-only; it must fail before preflight completes.
  require(mode != "incomplete" || !progressed);
  const auto beside_output = [](const fs::path &first) {
    return first.parent_path() / (first.stem().string() + "_merged.NEFX");
  };
  if (mode == "race_last")
    // Earlier outputs stay published and valid; the claimed name is untouched.
    require(path_exists(beside_output(paths[0])) && path_exists(beside_output(paths[4])) &&
            fs::file_size(last) == 0 &&
            message.find("2 of 3 outputs were published") != std::string::npos);
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
