#include "tiff.hpp"

#include <algorithm>
#include <chrono>
#include <iostream>

#ifndef _WIN32
#include <unistd.h>
#endif

using namespace cnefx;
using namespace cnefx::detail;
namespace {
#define CHECK(condition)                                                                           \
  do {                                                                                             \
    if (!(condition))                                                                              \
      throw std::runtime_error("Failed: " #condition);                                             \
  } while (false)
template <class F> void rejects(F action) {
  bool rejected = false;
  try {
    action();
  } catch (const std::exception &) {
    rejected = true;
  }
  CHECK(rejected);
}
void create(const fs::path &path, const Bytes &bytes) {
  File f(path, FileMode::create);
  f.write(0, bytes);
  f.sync();
}
Payload payload(Bytes bytes) {
  return {bytes.size(),
          [bytes = std::move(bytes)](const Sink &sink, const Check &) { sink(bytes); }};
}
void hashing(const fs::path &root) {
  std::cout << "SHA-256 hardware available: " << sha256_hardware_available() << '\n';
  CHECK(sha256(Bytes{}) == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
  CHECK(sha256(Bytes{'a', 'b', 'c'}) ==
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  Bytes million(1000000, 'a');
  CHECK(sha256(million) == "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
  CHECK(sha256(million, false) ==
        "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
  Bytes boundary(129);
  for (unsigned i = 0; i < boundary.size(); ++i)
    boundary[i] = (i * 131 + i / 7) & 255;
  for (unsigned size = 0; size <= boundary.size(); ++size) {
    const auto data = std::span(boundary).first(size);
    CHECK(sha256(data) == sha256(data, false));
  }
  // Hardware block runs must agree with the portable path at any alignment.
  Bytes blocks(8 + 64 * 70);
  for (unsigned i = 0; i < blocks.size(); ++i)
    blocks[i] = (i * 197 + i / 13) & 255;
  for (unsigned start = 0; start < 8; ++start)
    for (const std::size_t size : {63u, 64u, 65u, 128u, 191u, 1000u, 64u * 70})
      CHECK(sha256(std::span(blocks).subspan(start, size)) ==
            sha256(std::span(blocks).subspan(start, size), false));
  Bytes multiple_chunks(2 * 1024 * 1024 + 37);
  for (unsigned i = 0; i < multiple_chunks.size(); ++i)
    multiple_chunks[i] = (i * 131 + i / 7) & 255;
  create(root / "chunks.bin", multiple_chunks);
  File chunks(root / "chunks.bin");
  CHECK(sha256(chunks) == sha256(multiple_chunks, false));
  CHECK(sha256(chunks, {}, false) == sha256(multiple_chunks));
  const auto contents = read_contents(chunks);
  CHECK(contents.sha256 == sha256(multiple_chunks, false));
  CHECK(std::ranges::equal(contents.memory->bytes(), multiple_chunks));
  rejects([&] { contents.memory->read(multiple_chunks.size(), 1); });
  const auto path = root / "hash.bin";
  create(path, million);
  File file(path);
  CHECK(sha256(file) == sha256(million));
  CHECK(file.state() == file_state(path));
  rejects([&] { create(path, Bytes{1}); });
  rejects([&] { file.read(million.size(), 1); });
}
Directory graph() {
  Directory root;
  root.set(271, 2, Bytes{'N', 'I', 'K', 'O', 'N', 0});
  root.numbers(330, 4, {0});
  Directory raw;
  raw.numbers(256, 4, {32});
  raw.numbers(257, 4, {16});
  raw.numbers(273, 4, {0});
  raw.numbers(279, 4, {0});
  raw.images[273].push_back(payload(Bytes{0, 1, 2, 3, 4, 5, 6, 7, 8}));
  root.at(330).children.push_back(std::move(raw));
  root.numbers(34665, 4, {0});
  Directory exif, maker;
  maker.numbers(0x45, 3, {1, 1, 28, 12});
  maker.set(0x97, 7, Bytes{'0', '8', '0', '3'});
  exif.set(37500, 7, {});
  auto &tag = exif.at(37500);
  tag.nikon_header = Bytes{'N', 'i', 'k', 'o', 'n', 0, 2, 17, 0, 0};
  tag.children.push_back(std::move(maker));
  root.at(34665).children.push_back(std::move(exif));
  return root;
}
void tiff(const fs::path &root) {
  auto tree = graph();
  const auto first = root / "first.tif", second = root / "second.tif";
  {
    File file(first, FileMode::create);
    write_tiff(tree, file);
  }
  auto input = std::make_shared<File>(first);
  auto restored = read_tiff(input);
  CHECK(restored.at(271).text() == "NIKON");
  CHECK(restored.at(330).children.front().at(256).number() == 32);
  CHECK(restored.at(34665).children[0].at(37500).children[0].at(0x45).integers() ==
        std::vector<std::uint32_t>({1, 1, 28, 12}));
  {
    File file(second, FileMode::create);
    write_tiff(restored, file);
  }
  File output(second);
  CHECK(sha256(*input) == sha256(output));
  // A graph read from memory streams the same payloads as one read from a file.
  const auto memory = read_contents(*input).memory;
  {
    File file(root / "memory.tif", FileMode::create);
    write_tiff(read_tiff(memory), file);
  }
  File from_memory(root / "memory.tif");
  CHECK(sha256(*input) == sha256(from_memory));
  auto malformed = root / "bad.tif";
  create(malformed, Bytes{'I', 'I', 42, 0, 8, 0, 0, 0});
  rejects([&] { read_tiff(std::make_shared<File>(malformed)); });
  auto bytes = input->read(0, input->size());
  // The root's second tag is a one-pointer SubIFD. Point it at its parent.
  put32(bytes, 8 + 2 + 12 + 8, 8);
  const auto cycle = root / "cycle.tif";
  create(cycle, bytes);
  rejects([&] { read_tiff(std::make_shared<File>(cycle)); });
  bytes = input->read(0, input->size());
  put16(bytes, 8 + 2 + 12, 271);
  const auto duplicate = root / "duplicate.tif";
  create(duplicate, bytes);
  rejects([&] { read_tiff(std::make_shared<File>(duplicate)); });
  bytes = input->read(0, input->size());
  put32(bytes, 8 + 2 + 3 * 12, 8);
  const auto chain = root / "chain.tif";
  create(chain, bytes);
  rejects([&] { read_tiff(std::make_shared<File>(chain)); });
  tree.at(330).children[0].images[273][0].length = 10;
  File wrong(root / "wrong.tif", FileMode::create);
  rejects([&] { write_tiff(tree, wrong); });
  File nonempty(first, FileMode::update);
  rejects([&] { write_tiff(graph(), nonempty); });
}
void publication(const fs::path &root) {
  const auto destination = root / "output.nefx";
  {
    Publication pending(destination, false);
    create(pending.temporary(), Bytes{1, 2, 3});
    pending.publish();
    pending.cleanup();
    CHECK(pending.warnings().empty());
  }
  CHECK(file_state(destination).size == 3);
  rejects([&] { Publication pending(destination, false); });
  const auto conflict = root / "conflict.nefx";
  {
    Publication pending(conflict, false);
    create(pending.temporary(), Bytes{1});
    create(conflict, Bytes{2, 3});
    rejects([&] { pending.publish(); });
  }
  CHECK(file_state(conflict).size == 2);
  const auto cancelled = root / "cancelled.nefx";
  {
    Publication pending(cancelled, false);
    create(pending.temporary(), Bytes{1});
    rejects([&] { pending.publish([] { throw std::runtime_error("Cancelled."); }); });
  }
  CHECK(!path_exists(cancelled));
  const auto completed = root / "completed.nefx";
  {
    Publication pending(completed, false);
    create(pending.temporary(), Bytes{1});
    unsigned checks = 0;
    pending.publish([&] {
      if (++checks == 3)
        throw std::runtime_error("Cancelled.");
    });
    pending.cleanup();
    CHECK(pending.warnings().size() == 1);
  }
  CHECK(path_exists(completed));
  const auto set = root / "outputs";
  {
    Publication pending(set, true);
    create(pending.temporary() / "one.nefx", Bytes{42});
    pending.publish();
    pending.cleanup();
  }
  CHECK(file_state(set / "one.nefx").size == 1);
  const auto directory_conflict = root / "directory-conflict";
  {
    Publication pending(directory_conflict, true);
    create(pending.temporary() / "one.nefx", Bytes{42});
    fs::create_directory(directory_conflict);
    rejects([&] { pending.publish(); });
  }
  CHECK(fs::is_empty(directory_conflict));
#ifndef _WIN32
  const auto link = root / "dangling.nefx";
  fs::create_symlink(root / "absent", link);
  rejects([&] { Publication pending(link, false); });
  rejects([&] { File file(root); });
  const auto denied = root / "denied";
  fs::create_directory(denied);
  {
    Publication pending(denied / "output.nefx", false);
    create(pending.temporary(), Bytes{1});
    fs::permissions(denied, fs::perms::owner_write | fs::perms::owner_exec);
    try {
      pending.publish();
      fs::permissions(denied, fs::perms::owner_all);
    } catch (...) {
      fs::permissions(denied, fs::perms::owner_all);
      throw;
    }
    CHECK(path_exists(denied / "output.nefx"));
    pending.cleanup();
  }
  for (const bool directory : {false, true}) {
    const auto locked = root / (directory ? "locked-set" : "locked-file");
    fs::create_directory(locked);
    const auto target = locked / (directory ? "outputs" : "output.nefx");
    std::string error;
    {
      Publication pending(target, directory);
      create(directory ? pending.temporary() / "one.nefx" : pending.temporary(), Bytes{1});
      fs::permissions(locked, fs::perms::owner_read | fs::perms::owner_exec);
      try {
        pending.publish();
      } catch (const std::exception &e) {
        error = e.what();
      }
      fs::permissions(locked, fs::perms::owner_all);
    }
    // The transfer error must survive the absent destination's lookup error.
    CHECK(geteuid() == 0 ? path_exists(target)
                         : !path_exists(target) && error.starts_with("Publish output"));
  }
#endif
}
} // namespace
int main(int argc, char **argv) {
  fs::path root;
  try {
    if (argc != 2)
      return 2;
    fs::create_directories(argv[1]);
    root = fs::path(argv[1]) /
           std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    CHECK(fs::create_directory(root));
    hashing(root);
    tiff(root);
    publication(root);
    fs::remove_all(root);
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    if (!root.empty()) {
      std::error_code error;
      fs::remove_all(root, error);
    }
    return 1;
  }
}
