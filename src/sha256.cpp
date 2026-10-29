#include "io.hpp"

#include <algorithm>
#include <bit>
#include <cstring>

namespace cnefx::detail {
namespace {
// SHA-256 as specified in FIPS 180-4. No platform crypto runtime is required.
constexpr std::uint32_t constants[] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
class Hash {
  std::array<std::uint32_t, 8> h_{0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                                  0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
  std::array<std::uint8_t, 64> block_{};
  std::size_t used_ = 0;
  std::uint64_t total_ = 0;
  void compress(const std::uint8_t *p) {
    std::uint32_t w[64]{};
    for (unsigned i = 0; i < 16; ++i)
      for (unsigned j = 0; j < 4; ++j)
        w[i] = (w[i] << 8) | p[4 * i + j];
    for (unsigned i = 16; i < 64; ++i) {
      const auto a = w[i - 15], b = w[i - 2];
      w[i] = w[i - 16] + (std::rotr(a, 7) ^ std::rotr(a, 18) ^ (a >> 3)) + w[i - 7] +
             (std::rotr(b, 17) ^ std::rotr(b, 19) ^ (b >> 10));
    }
    auto [a, b, c, d, e, f, g, h] = h_;
    for (unsigned i = 0; i < 64; ++i) {
      const auto t1 = h + (std::rotr(e, 6) ^ std::rotr(e, 11) ^ std::rotr(e, 25)) +
                      ((e & f) ^ (~e & g)) + constants[i] + w[i];
      const auto t2 =
          (std::rotr(a, 2) ^ std::rotr(a, 13) ^ std::rotr(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
      h = g;
      g = f;
      f = e;
      e = d + t1;
      d = c;
      c = b;
      b = a;
      a = t1 + t2;
    }
    h_[0] += a;
    h_[1] += b;
    h_[2] += c;
    h_[3] += d;
    h_[4] += e;
    h_[5] += f;
    h_[6] += g;
    h_[7] += h;
  }

public:
  void update(std::span<const std::uint8_t> data) {
    total_ += data.size();
    while (!data.empty()) {
      if (!used_ && data.size() >= 64) {
        compress(data.data());
        data = data.subspan(64);
        continue;
      }
      const auto n = std::min(data.size(), 64 - used_);
      std::memcpy(block_.data() + used_, data.data(), n);
      used_ += n;
      data = data.subspan(n);
      if (used_ == 64) {
        compress(block_.data());
        used_ = 0;
      }
    }
  }
  std::string finish() {
    const auto bits = total_ * 8;
    block_[used_++] = 0x80;
    if (used_ > 56) {
      std::fill(block_.begin() + used_, block_.end(), 0);
      compress(block_.data());
      used_ = 0;
    }
    std::fill(block_.begin() + used_, block_.begin() + 56, 0);
    for (unsigned i = 0; i < 8; ++i)
      block_[63 - i] = (bits >> (8 * i)) & 255;
    compress(block_.data());
    const char *digits = "0123456789abcdef";
    std::string result;
    for (auto v : h_)
      for (int shift = 28; shift >= 0; shift -= 4)
        result += digits[(v >> shift) & 15];
    return result;
  }
};
} // namespace
std::string sha256(std::span<const std::uint8_t> bytes) {
  Hash hash;
  hash.update(bytes);
  return hash.finish();
}
std::string sha256(File &file, const Check &check) {
  Hash hash;
  const auto size = file.size();
  Bytes buffer(1024 * 1024);
  for (std::uint64_t offset = 0; offset < size;) {
    if (check)
      check();
    auto chunk = std::span(buffer).first(
        static_cast<std::size_t>(std::min<std::uint64_t>(size - offset, buffer.size())));
    file.read(offset, chunk);
    hash.update(chunk);
    offset += chunk.size();
  }
  return hash.finish();
}
} // namespace cnefx::detail
