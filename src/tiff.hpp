#pragma once

#include "io.hpp"

#include <map>

namespace cnefx::detail {
std::uint16_t u16(std::span<const std::uint8_t> data, std::size_t offset);
std::uint32_t u32(std::span<const std::uint8_t> data, std::size_t offset);
std::int32_t i32(std::span<const std::uint8_t> data, std::size_t offset);
void put16(std::span<std::uint8_t> data, std::size_t offset, std::uint16_t value);
void put32(std::span<std::uint8_t> data, std::size_t offset, std::uint32_t value);
unsigned type_width(unsigned kind);
struct Directory;
struct Tag {
  unsigned code{}, kind{};
  std::uint32_t count{};
  Bytes data;
  std::vector<Directory> children;
  Bytes nikon_header;
  std::vector<std::uint32_t> integers() const;
  std::uint32_t number() const;
  std::string text() const;
};
struct Payload {
  std::uint64_t length{};
  std::function<void(const Sink &, const Check &)> chunks;
};
struct Directory {
  std::map<unsigned, Tag> tags;
  std::map<unsigned, std::vector<Payload>> images;
  void set(unsigned code, unsigned kind, Bytes data);
  void numbers(unsigned code, unsigned kind, std::initializer_list<std::uint32_t> values);
  Tag &at(unsigned code);
  const Tag &at(unsigned code) const;
};
Directory read_tiff(std::shared_ptr<Input> input);
void write_tiff(const Directory &directory, File &output, const Check &check = {});
} // namespace cnefx::detail
