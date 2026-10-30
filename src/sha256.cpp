#include "io.hpp"

#include <algorithm>
#include <bit>
#include <cstring>
#include <limits>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#ifdef _MSC_VER
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#elif defined(__aarch64__) && !defined(_MSC_VER)
#include <arm_neon.h>
#ifdef __linux__
#include <asm/hwcap.h>
#include <sys/auxv.h>
#endif
#endif

namespace cnefx::detail {
bool sha256_hardware_available() {
  static const bool available = [] {
#if defined(_M_X64) && defined(_MSC_VER)
    int registers[4]{};
    __cpuid(registers, 0);
    if (registers[0] < 7)
      return false;
    __cpuidex(registers, 7, 0);
    return (static_cast<unsigned>(registers[1]) & (1u << 29)) != 0;
#elif defined(__x86_64__)
    unsigned a{}, b{}, c{}, d{};
    return __get_cpuid_count(7, 0, &a, &b, &c, &d) && (b & (1u << 29));
#elif defined(__aarch64__) && defined(__linux__)
    return (getauxval(AT_HWCAP) & HWCAP_SHA2) != 0;
#elif defined(__aarch64__) && defined(__APPLE__)
    // SHA-256 instructions are part of the Apple AArch64 baseline.
    return true;
#else
    return false;
#endif
  }();
  return available;
}
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

// Keep ISA-specific rounds behind the capability check. Padding and the message
// schedule share the portable implementation, including its uint32 arithmetic.
#if defined(__x86_64__) || defined(_M_X64)
#ifdef _MSC_VER
__declspec(noinline)
#else
__attribute__((target("sha"), noinline))
#endif
void hardware_rounds(std::array<std::uint32_t, 8>& h, const std::uint32_t* w) {
  auto abef = _mm_set_epi32(h[0], h[1], h[4], h[5]);
  auto cdgh = _mm_set_epi32(h[2], h[3], h[6], h[7]);
  for (unsigned i = 0; i < 64; i += 4) {
    const auto message =
        _mm_add_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i *>(w + i)),
                      _mm_loadu_si128(reinterpret_cast<const __m128i *>(constants + i)));
    cdgh = _mm_sha256rnds2_epu32(cdgh, abef, message);
    abef = _mm_sha256rnds2_epu32(abef, cdgh, _mm_srli_si128(message, 8));
  }
  std::uint32_t a[4], b[4];
  _mm_storeu_si128(reinterpret_cast<__m128i *>(a), abef);
  _mm_storeu_si128(reinterpret_cast<__m128i *>(b), cdgh);
  h[0] += a[3];
  h[1] += a[2];
  h[2] += b[3];
  h[3] += b[2];
  h[4] += a[1];
  h[5] += a[0];
  h[6] += b[1];
  h[7] += b[0];
}
#elif defined(__aarch64__) && !defined(_MSC_VER)
#if defined(__clang__)
__attribute__((target("sha2"), noinline))
#else
__attribute__((target("+sha2"), noinline))
#endif
void hardware_rounds(std::array<std::uint32_t, 8>& h, const std::uint32_t* w) {
  auto abcd = vld1q_u32(h.data()), efgh = vld1q_u32(h.data() + 4);
  const auto original_abcd = abcd, original_efgh = efgh;
  for (unsigned i = 0; i < 64; i += 4) {
    const auto message = vaddq_u32(vld1q_u32(w + i), vld1q_u32(constants + i));
    const auto previous = abcd;
    abcd = vsha256hq_u32(abcd, efgh, message);
    efgh = vsha256h2q_u32(efgh, previous, message);
  }
  vst1q_u32(h.data(), vaddq_u32(abcd, original_abcd));
  vst1q_u32(h.data() + 4, vaddq_u32(efgh, original_efgh));
}
#else
void hardware_rounds(std::array<std::uint32_t, 8> &, const std::uint32_t *) {}
#endif

class Hash {
  std::array<std::uint32_t, 8> h_{0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                                  0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
  std::array<std::uint8_t, 64> block_{};
  std::size_t used_ = 0;
  std::uint64_t total_ = 0;
  bool accelerated_;
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
    if (accelerated_) {
      hardware_rounds(h_, w);
      return;
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
  explicit Hash(bool accelerate) : accelerated_(accelerate && sha256_hardware_available()) {}
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
std::string sha256(std::span<const std::uint8_t> bytes, bool accelerate) {
  Hash hash(accelerate);
  hash.update(bytes);
  return hash.finish();
}
namespace {
// Reads the file in bounded chunks; target supplies the destination of each.
template <class Target>
std::string hash_file(File &file, const Check &check, bool accelerate, Target target) {
  Hash hash(accelerate);
  const auto size = file.size();
  for (std::uint64_t offset = 0; offset < size;) {
    if (check)
      check();
    const auto chunk = target(
        offset, static_cast<std::size_t>(std::min<std::uint64_t>(size - offset, 1024 * 1024)));
    file.read(offset, chunk);
    hash.update(chunk);
    offset += chunk.size();
  }
  return hash.finish();
}
} // namespace
std::string sha256(File &file, const Check &check, bool accelerate) {
  Bytes buffer(1024 * 1024);
  return hash_file(file, check, accelerate,
                   [&](std::uint64_t, std::size_t size) { return std::span(buffer).first(size); });
}
Contents read_contents(File &file, const Check &check) {
  const auto size = file.size();
  if (size > std::numeric_limits<std::size_t>::max())
    throw std::invalid_argument("Input exceeds address space.");
  auto bytes = std::make_shared<Bytes>(static_cast<std::size_t>(size));
  auto digest = hash_file(file, check, true, [&](std::uint64_t offset, std::size_t length) {
    return std::span(*bytes).subspan(static_cast<std::size_t>(offset), length);
  });
  if (file.size() != bytes->size())
    throw std::invalid_argument("Input changed size while reading.");
  return {std::make_shared<Memory>(std::move(bytes)), std::move(digest)};
}
} // namespace cnefx::detail
