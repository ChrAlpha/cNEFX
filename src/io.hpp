#pragma once

#include "cnefx/core.hpp"

#include <cstdio>
#include <filesystem>
#include <memory>
#include <string>

namespace cnefx::detail {
namespace fs = std::filesystem;
using Bytes = std::vector<std::uint8_t>;
struct FileState {
  std::uint64_t device{}, inode{}, size{}, modified{}, changed{};
  bool regular{};
  bool operator==(const FileState &) const = default;
};
enum class FileMode { read, create, update };
class File {
public:
  explicit File(const fs::path &path, FileMode mode = FileMode::read);
  ~File();
  File(const File &) = delete;
  File &operator=(const File &) = delete;
  std::FILE *stream() const { return stream_; }
  const fs::path &path() const { return path_; }
  FileState state() const;
  std::uint64_t size() const { return state().size; }
  void seek(std::uint64_t offset);
  Bytes read(std::uint64_t offset, std::size_t size);
  void read(std::uint64_t offset, std::span<std::uint8_t> bytes);
  void write(std::uint64_t offset, std::span<const std::uint8_t> bytes);
  void sync();

private:
  fs::path path_;
  std::FILE *stream_ = nullptr;
};
FileState file_state(const fs::path &path);
bool path_exists(const fs::path &path);
std::string sha256(std::span<const std::uint8_t> bytes);
std::string sha256(File &file, const Check &check = {});

class Publication {
public:
  Publication(const fs::path &destination, bool directory);
  ~Publication();
  Publication(const Publication &) = delete;
  Publication &operator=(const Publication &) = delete;
  const fs::path &temporary() const { return temporary_; }
  void publish(const Check &check = {});
  void cleanup();
  const std::vector<std::string> &warnings() const { return warnings_; }

private:
  fs::path destination_, staging_, temporary_;
  bool directory_ = false, published_ = false;
  std::vector<std::string> warnings_;
};
} // namespace cnefx::detail
