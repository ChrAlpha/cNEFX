#include "cnefx/merge.hpp"
#include "io.hpp"

#include <iostream>
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
void inject(int argc, char **argv, bool bracketed) {
  const std::string mode = argv[2];
  MergeOptions options{argv[3], bracketed ? 1u : 2u, true, false};
  std::vector<fs::path> paths(argv + 4, argv + argc);
  bool fired = false, cancel = false, rejected = false, progressed = false;
  if (mode == "order") {
    fired = true;
    std::swap(paths[0], paths[1]);
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
        });
  } catch (const Cancelled &) {
    require(mode == "cancel" || mode == "cancel_decode");
    rejected = true;
  } catch (const std::exception &) {
    require(mode != "cancel" && mode != "cancel_decode");
    rejected = true;
  }
  require(fired && rejected);
  // Capture order is metadata-only; it must fail before preflight completes.
  require(mode != "order" || !progressed);
  require(mode == "target_race" ? fs::is_directory(options.output) && fs::is_empty(options.output)
                                : !path_exists(options.output));
  for (const auto &entry : fs::directory_iterator(options.output.parent_path()))
    require(!entry.path().filename().string().starts_with(".cnefx-"));
}
void tests() {
  require(error_json("a\"b", "a\nb").find("a\\u000ab") != std::string::npos);
  require(error_json("error", "\xf0\x9f\x98\x80").find("\\ud83d\\ude00") != std::string::npos);
  require(error_json("error", "\xff").find("\\udcff") != std::string::npos);
  MergeOptions options{"output.nefx", 1, false, false};
  bool rejected = false;
  try {
    merge(std::vector<fs::path>{"a", "b", "c"}, options);
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  require(rejected);
}
} // namespace
int main(int argc, char **argv) {
  try {
    if (argc == 1)
      tests();
    else if ((argc == 12 && std::string(argv[1]) == "--inject") ||
             (argc == 16 && std::string(argv[1]) == "--inject-brackets"))
      inject(argc, argv, std::string(argv[1]) == "--inject-brackets");
    else
      return 2;
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
