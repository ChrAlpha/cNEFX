#pragma once

#include "raw.hpp"

namespace cnefx::detail {
Bytes preview(const Rgb &image, const Check &check = {});
Payload packed14(const Rgb &image);
std::uint16_t crc16(std::span<const std::uint8_t> bytes);
void write_nefx(Source &source, const fs::path &path, const Rgb &image, unsigned count,
                const Check &check = {});
} // namespace cnefx::detail
