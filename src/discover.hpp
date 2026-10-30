#pragma once

#include "cnefx/merge.hpp"
#include "raw.hpp"

namespace cnefx::detail {
struct Capture {
  fs::path path;
  FileState state;
  Metadata metadata;
};
struct Discovery {
  std::vector<Capture> captures;
  std::vector<FileReport> skipped, errors;
};
// ASCII case folding for extensions, patterns and destination names.
std::u8string folded(const fs::path &path);
// Expands NEF files, directories (their immediate non-hidden .NEF files) and
// wildcard file-name patterns. Each file is opened once to classify it from
// metadata and closed again; unreadable captures are reported, not fatal.
Discovery discover(std::span<const fs::path> inputs, const Check &check);
} // namespace cnefx::detail
