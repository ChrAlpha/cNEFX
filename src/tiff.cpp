#include "tiff.hpp"

#include <algorithm>
#include <bit>
#include <limits>
#include <set>

namespace cnefx::detail {
namespace {
void bounds(std::size_t length, std::size_t offset, std::size_t count) {
  if (offset > length || count > length - offset)
    throw std::invalid_argument("Truncated TIFF value.");
}
constexpr std::pair<unsigned, unsigned> images[] = {{513, 514}, {273, 279}, {324, 325}};
constexpr std::uint8_t header[] = {'I', 'I', 42, 0, 8, 0, 0, 0};
} // namespace
std::uint16_t u16(std::span<const std::uint8_t> data, std::size_t offset) {
  bounds(data.size(), offset, 2);
  return data[offset] | (unsigned(data[offset + 1]) << 8);
}
std::uint32_t u32(std::span<const std::uint8_t> data, std::size_t offset) {
  bounds(data.size(), offset, 4);
  return std::uint32_t(u16(data, offset)) | (std::uint32_t(u16(data, offset + 2)) << 16);
}
std::int32_t i32(std::span<const std::uint8_t> data, std::size_t offset) {
  return std::bit_cast<std::int32_t>(u32(data, offset));
}
void put16(std::span<std::uint8_t> data, std::size_t offset, std::uint16_t value) {
  bounds(data.size(), offset, 2);
  data[offset] = value & 255;
  data[offset + 1] = value >> 8;
}
void put32(std::span<std::uint8_t> data, std::size_t offset, std::uint32_t value) {
  bounds(data.size(), offset, 4);
  put16(data, offset, value & 65535);
  put16(data, offset + 2, value >> 16);
}
unsigned type_width(unsigned kind) {
  constexpr unsigned widths[] = {0, 1, 1, 2, 4, 8, 1, 1, 2, 4, 8, 4, 8};
  if (!kind || kind >= std::size(widths))
    throw std::invalid_argument("Unsupported TIFF type.");
  return widths[kind];
}
std::vector<std::uint32_t> Tag::integers() const {
  if ((kind != 3 && kind != 4) || data.size() != std::uint64_t(count) * type_width(kind))
    throw std::invalid_argument("Unsupported TIFF integer type or length.");
  std::vector<std::uint32_t> values(count);
  for (unsigned i = 0; i < count; ++i)
    values[i] = kind == 3 ? u16(data, i * 2) : u32(data, i * 4);
  return values;
}
std::uint32_t Tag::number() const {
  if (count != 1)
    throw std::invalid_argument("Expected one TIFF integer.");
  return integers().front();
}
std::string Tag::text() const {
  if (kind != 2 || data.empty() || data.back() != 0)
    throw std::invalid_argument("Expected terminated TIFF ASCII.");
  const auto end = std::find(data.begin(), data.end(), 0);
  if (std::any_of(data.begin(), end, [](auto c) { return c > 127; }))
    throw std::invalid_argument("Invalid TIFF ASCII.");
  return std::string(data.begin(), end);
}
void Directory::set(unsigned code, unsigned kind, Bytes data) {
  const auto width = type_width(kind);
  if (code > 65535 || data.size() % width || data.size() / width > UINT32_MAX)
    throw std::invalid_argument("Invalid TIFF value width.");
  const auto count = static_cast<std::uint32_t>(data.size() / width);
  tags[code] = Tag{code, kind, count, std::move(data), {}, {}};
}
void Directory::numbers(unsigned code, unsigned kind, std::initializer_list<std::uint32_t> values) {
  if (kind != 3 && kind != 4)
    throw std::invalid_argument("Expected TIFF SHORT or LONG.");
  Bytes data(values.size() * type_width(kind));
  std::size_t i = 0;
  for (auto v : values) {
    if (kind == 3) {
      if (v > 65535)
        throw std::invalid_argument("TIFF SHORT overflow.");
      put16(data, i * 2, v);
    } else
      put32(data, i * 4, v);
    ++i;
  }
  set(code, kind, std::move(data));
}
Tag &Directory::at(unsigned code) { return const_cast<Tag &>(std::as_const(*this).at(code)); }
const Tag &Directory::at(unsigned code) const {
  auto tag = tags.find(code);
  if (tag == tags.end())
    throw std::invalid_argument("Missing required TIFF tag " + std::to_string(code) + ".");
  return tag->second;
}

Directory read_tiff(std::shared_ptr<Input> input) {
  const auto size = input->size();
  if (size >= (std::uint64_t{1} << 32))
    throw std::invalid_argument("Input exceeds classic TIFF size.");
  std::set<std::pair<std::uint64_t, std::uint32_t>> seen;
  std::uint64_t metadata_bytes = 0, payload_bytes = 0;
  const auto region = [&](std::uint64_t offset, std::uint64_t count) {
    if (offset > size || count > size - offset)
      throw std::invalid_argument("TIFF region exceeds file bounds.");
  };
  const auto block = [&](std::uint64_t offset, std::size_t count) {
    region(offset, count);
    metadata_bytes += count;
    if (metadata_bytes > 32u * 1024 * 1024)
      throw std::invalid_argument("TIFF metadata exceeds work budget.");
    return input->read(offset, count);
  };
  const auto first = [&](std::uint64_t base) {
    const auto h = block(base, 8);
    if (!std::equal(h.begin(), h.begin() + 4, std::begin(header)))
      throw std::invalid_argument("Only classic little-endian TIFF is supported.");
    return u32(h, 4);
  };
  std::function<Directory(std::uint64_t, std::uint32_t, bool)> walk;
  walk = [&](std::uint64_t base, std::uint32_t relative, bool maker) -> Directory {
    if (relative < 8 || seen.size() >= 64 || !seen.emplace(base, relative).second)
      throw std::invalid_argument("Repeated or excessive TIFF directory references.");
    const auto offset = base + relative;
    const auto count = u16(block(offset, 2), 0);
    const auto entries = block(offset + 2, std::size_t(count) * 12 + 4);
    if (u32(entries, count * 12))
      throw std::invalid_argument("Nonzero TIFF next-directory chains are unsupported.");
    Directory directory;
    for (unsigned i = 0; i < count; ++i) {
      const unsigned code = u16(entries, i * 12), kind = u16(entries, i * 12 + 2);
      const auto n = u32(entries, i * 12 + 4), value = u32(entries, i * 12 + 8);
      const std::uint64_t length = std::uint64_t(n) * type_width(kind);
      const auto position = length <= 4 ? offset + 2 + i * 12 + 8 : base + value;
      region(position, length);
      if (directory.tags.contains(code))
        throw std::invalid_argument("Duplicate TIFF tag.");
      Tag tag{code, kind, n, {}, {}, {}};
      if (code == 37500 || code == 0xc7d5) {
        if (kind != 7 || length < 18)
          throw std::invalid_argument("Unsupported embedded Nikon TIFF.");
        tag.nikon_header = block(position, 10);
        const Bytes prefix{'N', 'i', 'k', 'o', 'n', 0};
        if (!std::equal(prefix.begin(), prefix.end(), tag.nikon_header.begin()))
          throw std::invalid_argument("Unsupported Nikon header.");
        tag.children.push_back(walk(position + 10, first(position + 10), code == 37500));
      } else {
        if (length > 32u * 1024 * 1024)
          throw std::invalid_argument("TIFF value exceeds work budget.");
        tag.data = block(position, static_cast<std::size_t>(length));
        if (code == 330 || code == 34665 || code == 34853 || code == 40965 ||
            (maker && code == 17)) {
          if (kind != 4 || n < 1 || n > 16)
            throw std::invalid_argument("Unsupported linked TIFF directory layout.");
          for (auto pointer : tag.integers())
            tag.children.push_back(walk(base, pointer, false));
        }
      }
      directory.tags.emplace(code, std::move(tag));
    }
    for (auto [offset_tag, length_tag] : images) {
      if (!directory.tags.contains(offset_tag))
        continue;
      const auto offsets = directory.at(offset_tag).integers(),
                 lengths = directory.at(length_tag).integers();
      if (offsets.size() != lengths.size() || offsets.size() > 32)
        throw std::invalid_argument("Unsupported TIFF strip or tile array.");
      for (std::size_t i = 0; i < offsets.size(); ++i) {
        const auto position = base + offsets[i], length = std::uint64_t(lengths[i]);
        region(position, length);
        payload_bytes += length;
        if (payload_bytes > 2 * size + 4u * 1024 * 1024)
          throw std::invalid_argument("TIFF payload references exceed work budget.");
        directory.images[offset_tag].push_back(
            {length, [input, position, length](const Sink &sink, const Check &check) {
               input->stream(position, length, sink, check);
             }});
      }
    }
    return directory;
  };
  return walk(0, first(0), false);
}

void write_tiff(const Directory &root, File &output, const Check &check) {
  if (output.size() != 0)
    throw std::invalid_argument("TIFF output must be empty.");
  output.write(0, header);
  std::uint64_t end = 8;
  unsigned directories = 0;
  const auto allocate = [&](std::uint64_t size, bool align = true) {
    const auto offset = align ? (end + 15) & ~std::uint64_t{15} : end;
    if (size >= (std::uint64_t{1} << 32) || offset + size >= (std::uint64_t{1} << 32))
      throw std::invalid_argument("Output exceeds classic TIFF offset limits.");
    if (offset + size > end) {
      const std::uint8_t zero = 0;
      output.write(offset + size - 1, std::span(&zero, 1));
    }
    end = offset + size;
    return offset;
  };
  const auto write = [&](std::span<const std::uint8_t> bytes) {
    const auto offset = bytes.empty() ? end : allocate(bytes.size());
    output.write(offset, bytes);
    return offset;
  };
  std::function<std::uint64_t(const Directory &, std::uint64_t, bool)> walk;
  walk = [&](const Directory &directory, std::uint64_t base, bool adjacent) {
    if (++directories > 64 || directory.tags.size() > 65535)
      throw std::invalid_argument("Excessive TIFF directory count.");
    const auto size = 2 + directory.tags.size() * 12 + 4;
    const auto start = allocate(size, !adjacent);
    Bytes entries(size);
    put16(entries, 0, directory.tags.size());
    std::map<unsigned, Bytes> replacements;
    for (auto [offset_tag, length_tag] : images) {
      if (!directory.images.contains(offset_tag))
        continue;
      if (!directory.tags.contains(offset_tag) || !directory.tags.contains(length_tag))
        throw std::invalid_argument("Missing TIFF image tags.");
      const auto &payloads = directory.images.at(offset_tag);
      if (payloads.size() > 32)
        throw std::invalid_argument("Too many TIFF image strips.");
      Bytes offsets(payloads.size() * 4), lengths(payloads.size() * 4);
      for (std::size_t i = 0; i < payloads.size(); ++i) {
        const auto &payload = payloads[i];
        const auto position = payload.length ? allocate(payload.length) : end;
        std::uint64_t written = 0;
        if (!payload.chunks)
          throw std::invalid_argument("Missing TIFF image payload reader.");
        payload.chunks(
            [&](std::span<const std::uint8_t> chunk) {
              if (check)
                check();
              if (chunk.size() > payload.length - written)
                throw std::invalid_argument("TIFF chunks exceed declared length.");
              output.write(position + written, chunk);
              written += chunk.size();
            },
            check);
        if (written != payload.length)
          throw std::invalid_argument("TIFF chunks disagree with declared length.");
        put32(offsets, i * 4, payload.length ? position - base : 0);
        put32(lengths, i * 4, payload.length);
      }
      replacements[offset_tag] = std::move(offsets);
      replacements[length_tag] = std::move(lengths);
    }
    unsigned index = 0;
    for (const auto &[code, tag] : directory.tags) {
      if (check)
        check();
      auto kind = tag.kind;
      auto count = tag.count;
      auto payload = tag.data;
      if (replacements.contains(code)) {
        payload = replacements.at(code);
        kind = 4;
        count = payload.size() / 4;
      }
      std::uint32_t value = 0;
      if (!tag.nikon_header.empty()) {
        if (tag.nikon_header.size() != 10 || tag.children.size() != 1)
          throw std::invalid_argument("Invalid embedded Nikon TIFF.");
        const auto position = allocate(18);
        output.write(position, tag.nikon_header);
        output.write(position + 10, header);
        walk(tag.children.front(), position + 10, true);
        kind = 7;
        count = end - position;
        value = position - base;
      } else {
        if (!tag.children.empty()) {
          kind = 4;
          count = tag.children.size();
          payload.resize(count * 4);
          for (unsigned i = 0; i < count; ++i)
            put32(payload, i * 4, walk(tag.children[i], base, false) - base);
        }
        if (payload.size() != std::uint64_t(count) * type_width(kind))
          throw std::invalid_argument("TIFF value length disagrees with its type.");
        if (payload.size() > 4)
          value = write(payload) - base;
        else {
          Bytes small(4);
          std::copy(payload.begin(), payload.end(), small.begin());
          value = u32(small, 0);
        }
      }
      put16(entries, 2 + index * 12, code);
      put16(entries, 4 + index * 12, kind);
      put32(entries, 6 + index * 12, count);
      put32(entries, 10 + index * 12, value);
      ++index;
    }
    output.write(start, entries);
    return start;
  };
  walk(root, 0, true);
  output.sync();
}
} // namespace cnefx::detail
