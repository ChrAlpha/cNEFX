#include "io.hpp"

#include <cerrno>
#include <limits>
#include <random>
#include <system_error>

#ifdef _WIN32
#define NOMINMAX
#include <fcntl.h>
#include <io.h>
#include <sys/stat.h>
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#ifdef __linux__
#include <sys/syscall.h>
#endif
#endif

namespace cnefx::detail {
namespace {
[[noreturn]] void io_error(const char *message) {
  throw std::system_error(errno, std::generic_category(), message);
}
#ifdef _WIN32
FileState state_of(HANDLE handle) {
  BY_HANDLE_FILE_INFORMATION info{};
  FILE_BASIC_INFO basic{};
  if (!GetFileInformationByHandle(handle, &info) ||
      !GetFileInformationByHandleEx(handle, FileBasicInfo, &basic, sizeof(basic)))
    throw std::system_error(GetLastError(), std::system_category(), "File identity");
  return {info.dwVolumeSerialNumber,
          (std::uint64_t(info.nFileIndexHigh) << 32) | info.nFileIndexLow,
          (std::uint64_t(info.nFileSizeHigh) << 32) | info.nFileSizeLow,
          std::uint64_t(basic.LastWriteTime.QuadPart),
          std::uint64_t(basic.ChangeTime.QuadPart),
          !(info.dwFileAttributes & (FILE_ATTRIBUTE_DIRECTORY | FILE_ATTRIBUTE_REPARSE_POINT))};
}
void sync_directory(const fs::path &) {
  // Windows has no portable directory fsync. MoveFileEx uses WRITE_THROUGH.
}
#else
FileState state_of(const struct stat &state) {
#ifdef __APPLE__
  auto m = state.st_mtimespec, c = state.st_ctimespec;
#else
  auto m = state.st_mtim, c = state.st_ctim;
#endif
  return {std::uint64_t(state.st_dev),
          std::uint64_t(state.st_ino),
          std::uint64_t(state.st_size),
          std::uint64_t(m.tv_sec) * 1000000000 + std::uint64_t(m.tv_nsec),
          std::uint64_t(c.tv_sec) * 1000000000 + std::uint64_t(c.tv_nsec),
          S_ISREG(state.st_mode)};
}
void sync_directory(const fs::path &path) {
  const int fd = open(path.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
  if (fd < 0)
    io_error("Open directory for sync");
  const int result = fsync(fd), error = errno;
  close(fd);
  if (result) {
    errno = error;
    io_error("Sync directory");
  }
}
#endif
bool same_identity(const FileState &a, const FileState &b) {
  return a.device == b.device && a.inode == b.inode && a.regular == b.regular;
}
bool published(const fs::path &path, const FileState &identity) {
  try {
    return same_identity(identity, file_state(path));
  } catch (const std::exception &) {
    return false;
  }
}
void transfer(const fs::path &source, const fs::path &target, bool directory) {
  const char *action = directory ? "Publish output directory without replacement"
                                 : "Publish output without replacement";
#ifdef _WIN32
  if (!MoveFileExW(source.c_str(), target.c_str(), MOVEFILE_WRITE_THROUGH))
    throw std::system_error(GetLastError(), std::system_category(), action);
#else
#if defined(__linux__) && defined(SYS_renameat2)
  constexpr unsigned rename_noreplace = 1;
  if (!syscall(SYS_renameat2, AT_FDCWD, source.c_str(), AT_FDCWD, target.c_str(), rename_noreplace))
    return;
  const bool unsupported = errno == EINVAL || errno == ENOSYS;
#elif defined(__APPLE__)
  if (!renamex_np(source.c_str(), target.c_str(), RENAME_EXCL))
    return;
  const bool unsupported = errno == ENOTSUP || errno == EINVAL;
#else
  const bool unsupported = true;
  errno = ENOTSUP;
#endif
  // Without an exclusive rename, a hard link still publishes a file without
  // replacement. No equivalent exists for a directory.
  if (!unsupported || directory)
    io_error(action);
  if (link(source.c_str(), target.c_str()))
    io_error(action);
#endif
}
} // namespace

File::File(const fs::path &path, FileMode mode) : path_(path) {
  const bool create = mode == FileMode::create, writable = mode != FileMode::read;
#ifdef _WIN32
  const int fd =
      _wopen(path.c_str(),
             (writable ? _O_RDWR : _O_RDONLY) | (create ? _O_CREAT | _O_EXCL : 0) | _O_BINARY,
             _S_IREAD | _S_IWRITE);
  if (fd < 0)
    io_error("Open file");
  stream_ = _fdopen(fd, writable ? "r+b" : "rb");
  if (!stream_) {
    _close(fd);
    io_error("Open file stream");
  }
#else
  const int fd = open(path.c_str(),
                      (writable ? O_RDWR : O_RDONLY) | (create ? O_CREAT | O_EXCL : 0) | O_CLOEXEC |
                          O_NOFOLLOW | O_NONBLOCK,
                      0600);
  if (fd < 0)
    io_error("Open file");
  stream_ = fdopen(fd, writable ? "r+b" : "rb");
  if (!stream_) {
    close(fd);
    io_error("Open file stream");
  }
#endif
  try {
    if (!state().regular)
      throw std::invalid_argument("Expected a regular file.");
  } catch (...) {
    std::fclose(stream_);
    stream_ = nullptr;
    throw;
  }
}
File::~File() {
  if (stream_)
    std::fclose(stream_);
}
FileState File::state() const {
#ifdef _WIN32
  return state_of(reinterpret_cast<HANDLE>(_get_osfhandle(_fileno(stream_))));
#else
  struct stat st{};
  if (fstat(fileno(stream_), &st))
    io_error("Read file identity");
  return state_of(st);
#endif
}
FileState file_state(const fs::path &path) {
#ifdef _WIN32
  HANDLE h = CreateFileW(
      path.c_str(), FILE_READ_ATTRIBUTES, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
      nullptr, OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OPEN_REPARSE_POINT, nullptr);
  if (h == INVALID_HANDLE_VALUE)
    throw std::system_error(GetLastError(), std::system_category(), "Read path identity");
  try {
    auto state = state_of(h);
    CloseHandle(h);
    return state;
  } catch (...) {
    CloseHandle(h);
    throw;
  }
#else
  struct stat st{};
  if (lstat(path.c_str(), &st))
    io_error("Read path identity");
  return state_of(st);
#endif
}
bool path_exists(const fs::path &path) {
  return fs::symlink_status(path).type() != fs::file_type::not_found;
}
void File::seek(std::uint64_t offset) {
  if (offset > std::uint64_t(std::numeric_limits<std::int64_t>::max()))
    throw std::invalid_argument("File offset overflow.");
#ifdef _WIN32
  if (_fseeki64(stream_, static_cast<std::int64_t>(offset), SEEK_SET))
    io_error("Seek file");
#else
  if (fseeko(stream_, static_cast<off_t>(offset), SEEK_SET))
    io_error("Seek file");
#endif
}
Bytes File::read(std::uint64_t offset, std::size_t size) {
  Bytes data(size);
  read(offset, data);
  return data;
}
void File::read(std::uint64_t offset, std::span<std::uint8_t> bytes) {
  seek(offset);
  if (std::fread(bytes.data(), 1, bytes.size(), stream_) != bytes.size())
    throw std::runtime_error("Truncated or unreadable file region.");
}
void File::write(std::uint64_t offset, std::span<const std::uint8_t> bytes) {
  seek(offset);
  if (std::fwrite(bytes.data(), 1, bytes.size(), stream_) != bytes.size())
    io_error("Write file");
}
void File::sync() {
  if (std::fflush(stream_))
    io_error("Flush file");
#ifdef _WIN32
  if (_commit(_fileno(stream_)))
    io_error("Sync file");
#else
  if (fsync(fileno(stream_)))
    io_error("Sync file");
#endif
}

Publication::Publication(const fs::path &destination, bool directory)
    : destination_(fs::absolute(destination)), directory_(directory) {
  if (path_exists(destination_))
    throw std::invalid_argument("Output already exists; overwriting is not supported.");
  std::random_device random;
  for (unsigned attempt = 0; attempt < 64; ++attempt) {
    staging_ = destination_.parent_path() /
               (".cnefx-" + std::to_string(random()) + "-" + std::to_string(random()));
#ifdef _WIN32
    if (CreateDirectoryW(staging_.c_str(), nullptr))
      break;
    if (GetLastError() != ERROR_ALREADY_EXISTS)
      throw std::system_error(GetLastError(), std::system_category(), "Create staging directory");
#else
    if (!mkdir(staging_.c_str(), 0700))
      break;
    if (errno != EEXIST)
      io_error("Create staging directory");
#endif
    staging_.clear();
  }
  if (staging_.empty())
    throw std::runtime_error("Cannot allocate private staging directory.");
  temporary_ = staging_ / (directory ? "outputs" : "output.nefx");
  try {
    if (directory)
      fs::create_directory(temporary_);
  } catch (...) {
    std::error_code ignored;
    fs::remove_all(staging_, ignored);
    throw;
  }
}
Publication::~Publication() {
  std::error_code ignored;
  fs::remove_all(staging_, ignored);
}
void Publication::publish(const Check &check) {
  if (published_)
    throw std::logic_error("Output is already published.");
  if (check)
    check();
  const auto identity = file_state(temporary_);
  if (directory_ ? !fs::is_directory(fs::symlink_status(temporary_)) : !identity.regular)
    throw std::invalid_argument("Unexpected staging file type.");
  if (directory_)
    sync_directory(temporary_);
  else {
    File file(temporary_, FileMode::update);
    file.sync();
  }
  if (check)
    check();
  try {
    transfer(temporary_, destination_, directory_);
  } catch (...) {
    // A reported failure can follow a completed transfer. Otherwise keep the
    // original error rather than the absent destination's lookup error.
    if (!published(destination_, identity))
      throw;
    warnings_.push_back("Output published; transfer reported a completion failure.");
  }
  published_ = true;
  try {
    if (check)
      check();
  } catch (const std::exception &e) {
    warnings_.push_back(std::string("Output published; ") + e.what());
  }
  try {
    sync_directory(destination_.parent_path());
  } catch (const std::exception &e) {
    warnings_.push_back(std::string("Output published; parent directory sync failed: ") + e.what());
  }
}
void Publication::cleanup() {
  try {
    fs::remove_all(staging_);
  } catch (const std::exception &e) {
    if (!published_)
      throw;
    warnings_.push_back(std::string("Output published; temporary cleanup failed: ") + e.what());
  }
}
} // namespace cnefx::detail
