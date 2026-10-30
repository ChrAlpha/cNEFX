#include "discover.hpp"

#include <algorithm>
#include <set>

namespace cnefx::detail {
namespace {
bool wildcard(std::u8string_view text) { return text.find_first_of(u8"*?") != text.npos; }
bool hidden(const fs::path &path) { return path.filename().u8string().starts_with(u8'.'); }
// Bytes in the UTF-8 character that starts at the given offset.
std::size_t width(std::u8string_view text, std::size_t at) {
  std::size_t n = 1;
  while (at + n < text.size() && (text[at + n] & 0xc0) == 0x80)
    ++n;
  return n;
}
// '*' matches any run and '?' one character; both sides are case-folded.
bool matches(std::u8string_view pattern, std::u8string_view name) {
  std::size_t p = 0, n = 0, star = pattern.npos, resume = 0;
  while (n < name.size()) {
    if (p < pattern.size() && pattern[p] == u8'*') {
      star = p++;
      resume = n;
    } else if (p < pattern.size() && pattern[p] == u8'?') {
      ++p;
      n += width(name, n);
    } else if (p < pattern.size() && pattern[p] == name[n]) {
      ++p;
      ++n;
    } else if (star != pattern.npos) {
      p = star + 1;
      resume += width(name, resume);
      n = resume;
    } else
      return false;
  }
  while (p < pattern.size() && pattern[p] == u8'*')
    ++p;
  return p == pattern.size();
}
// Folder listings hold regular files only; pattern matches may name folders.
std::vector<fs::path> sorted(const fs::path &directory, std::u8string_view pattern = u8"*") {
  std::vector<fs::path> paths;
  for (const auto &entry : fs::directory_iterator(directory))
    if ((!hidden(entry.path()) || pattern.starts_with(u8'.')) &&
        (pattern != u8"*" || entry.is_regular_file()) &&
        matches(pattern, folded(entry.path().filename())))
      paths.push_back(entry.path());
  std::sort(paths.begin(), paths.end());
  return paths;
}
} // namespace
std::u8string folded(const fs::path &path) {
  auto text = path.u8string();
  std::transform(text.begin(), text.end(), text.begin(),
                 [](char8_t c) { return c >= u8'A' && c <= u8'Z' ? char8_t(c + 32) : c; });
  return text;
}
Discovery discover(std::span<const fs::path> inputs, const Check &check) {
  Discovery result;
  std::set<fs::path> seen;
  std::set<std::pair<std::uint64_t, std::uint64_t>> identities;
  // Named files that are not NEF are reported; folder listings consider .NEF only.
  const auto consider = [&](const fs::path &path, bool named) {
    const auto extension = folded(path.extension());
    if (extension != u8".nef" && (!named || extension != u8".nefx")) {
      if (named)
        result.skipped.push_back({path, "Not a NEF file."});
      return;
    }
    if (check)
      check();
    auto reported = path;
    try {
      const auto canonical = fs::canonical(path);
      reported = canonical;
      // A file reached twice, e.g. through a folder and a pattern, counts once.
      if (!seen.insert(canonical).second)
        return;
      const auto state = file_state(canonical);
      if (!state.regular)
        throw std::invalid_argument("Input must be a regular file.");
      if (!identities.emplace(state.device, state.inode).second)
        throw std::invalid_argument("Hard link to another input.");
      const auto file = std::make_shared<File>(canonical);
      if (file->state() != state)
        throw std::invalid_argument("Input changed while opening.");
      // A file without a parseable pixel-shift MakerNote, such as another
      // camera's NEF, is skipped; a pixel-shift capture must pass strict checks.
      Directory root;
      try {
        root = read_tiff(file);
      } catch (const std::exception &e) {
        result.skipped.push_back({canonical, std::string("Unsupported NEF: ") + e.what()});
        return;
      }
      if (const auto reason = skip_reason(root))
        result.skipped.push_back({canonical, reason});
      else
        result.captures.push_back({canonical, state, read_metadata(root)});
    } catch (const Cancelled &) {
      throw;
    } catch (const std::exception &e) {
      result.errors.push_back({reported, e.what()});
    }
  };
  for (const auto &input : inputs) {
    const auto pattern = folded(input.filename());
    const bool exists = path_exists(input);
    // The root name may hold the Windows \\?\ long-path prefix.
    if (!exists && wildcard(input.parent_path().relative_path().u8string()))
      throw std::invalid_argument("Wildcards are supported only in file names: " +
                                  path_utf8(input));
    if (!exists && !wildcard(pattern))
      throw std::invalid_argument("Input not found: " + path_utf8(input));
    if (!exists) {
      // Shells on Windows pass patterns through; elsewhere a quoted pattern does.
      const auto parent = input.parent_path().empty() ? fs::path(".") : input.parent_path();
      const auto paths =
          fs::is_directory(parent) ? sorted(parent, pattern) : std::vector<fs::path>{};
      if (paths.empty())
        throw std::invalid_argument("No files match " + path_utf8(input) + ".");
      for (const auto &path : paths)
        if (fs::is_directory(path))
          for (const auto &entry : sorted(path))
            consider(entry, false);
        else
          consider(path, true);
    } else if (fs::is_directory(input))
      for (const auto &entry : sorted(input))
        consider(entry, false);
    else
      consider(input, true);
  }
  return result;
}
} // namespace cnefx::detail
