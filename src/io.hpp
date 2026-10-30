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
using Sink = std::function<void(std::span<const std::uint8_t>)>;
// Random-access bytes for bounded TIFF parsing and payload streaming.
class Input {
public:
  virtual ~Input() = default;
  virtual std::uint64_t size() const = 0;
  virtual void read(std::uint64_t offset, std::span<std::uint8_t> bytes) = 0;
  Bytes read(std::uint64_t offset, std::size_t size);
  // Passes [offset, offset + length) to sink in bounded chunks.
  virtual void stream(std::uint64_t offset, std::uint64_t length, const Sink &sink,
                      const Check &check);
};
class File final : public Input {
public:
  explicit File(const fs::path &path, FileMode mode = FileMode::read);
  ~File() override;
  File(const File &) = delete;
  File &operator=(const File &) = delete;
  using Input::read;
  std::FILE *stream() const { return stream_; }
  const fs::path &path() const { return path_; }
  FileState state() const;
  std::uint64_t size() const override { return state().size; }
  void seek(std::uint64_t offset);
  void read(std::uint64_t offset, std::span<std::uint8_t> bytes) override;
  void write(std::uint64_t offset, std::span<const std::uint8_t> bytes);
  void sync();

private:
  fs::path path_;
  std::FILE *stream_ = nullptr;
};
class Memory final : public Input {
public:
  explicit Memory(std::shared_ptr<const Bytes> bytes);
  using Input::read;
  std::span<const std::uint8_t> bytes() const { return *bytes_; }
  std::uint64_t size() const override { return bytes_->size(); }
  void read(std::uint64_t offset, std::span<std::uint8_t> bytes) override;
  void stream(std::uint64_t offset, std::uint64_t length, const Sink &sink,
              const Check &check) override;

private:
  std::shared_ptr<const Bytes> bytes_;
};
FileState file_state(const fs::path &path);
bool path_exists(const fs::path &path);
bool sha256_hardware_available();
std::string sha256(std::span<const std::uint8_t> bytes, bool accelerate = true);
std::string sha256(File &file, const Check &check = {}, bool accelerate = true);
struct Contents {
  std::shared_ptr<Memory> memory;
  std::string sha256;
};
// Reads the whole file once, hashing exactly the bytes that are returned.
Contents read_contents(File &file, const Check &check = {});

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
