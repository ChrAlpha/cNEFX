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
    __cpuidex(registers, 1, 0);
    const auto features = static_cast<unsigned>(registers[2]);
    if (!(features & (1u << 9)) || !(features & (1u << 19)))
      return false;
    __cpuidex(registers, 7, 0);
    return (static_cast<unsigned>(registers[1]) & (1u << 29)) != 0;
#elif defined(__x86_64__)
    unsigned a{}, b{}, c{}, d{};
    // SHA-NI plus the SSSE3 shuffles and SSE4.1 blends used around it.
    return __get_cpuid(1, &a, &b, &c, &d) && (c & (1u << 9)) && (c & (1u << 19)) &&
           __get_cpuid_count(7, 0, &a, &b, &c, &d) && (b & (1u << 29));
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

// ISA-specific code runs only behind the capability check. It computes the
// message schedule and rounds for consecutive blocks with the state in vector
// registers; padding is shared with the portable implementation.
#if defined(__x86_64__) || defined(_M_X64)
#ifdef _MSC_VER
#define CNEFX_SHA_TARGET
#else
#define CNEFX_SHA_TARGET __attribute__((target("sha,sse4.1")))
#endif
namespace x86 {
CNEFX_SHA_TARGET inline void rounds(__m128i &abef, __m128i &cdgh, __m128i message, unsigned group) {
  const auto schedule = _mm_add_epi32(
      message, _mm_loadu_si128(reinterpret_cast<const __m128i *>(constants + 4 * group)));
  cdgh = _mm_sha256rnds2_epu32(cdgh, abef, schedule);
  abef = _mm_sha256rnds2_epu32(abef, cdgh, _mm_shuffle_epi32(schedule, 0x0e));
}
// W[t..t+3] from W[t-16..t-13], W[t-12..t-9], W[t-8..t-5] and W[t-4..t-1].
CNEFX_SHA_TARGET inline __m128i next(__m128i a, __m128i b, __m128i c, __m128i d) {
  return _mm_sha256msg2_epu32(_mm_add_epi32(_mm_sha256msg1_epu32(a, b), _mm_alignr_epi8(d, c, 4)),
                              d);
}
CNEFX_SHA_TARGET inline __m128i load(const std::uint8_t *data, unsigned i) {
  const auto order = _mm_set_epi64x(0x0c0d0e0f08090a0bULL, 0x0405060700010203ULL);
  return _mm_shuffle_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i *>(data + 16 * i)), order);
}
} // namespace x86
#ifdef _MSC_VER
__declspec(noinline)
#else
__attribute__((target("sha,sse4.1"), noinline))
#endif
void hardware_blocks(std::array<std::uint32_t, 8> &h, const std::uint8_t *data,
                     std::size_t blocks) {
  // Rearrange ABCD/EFGH into the ABEF/CDGH lanes used by sha256rnds2.
  auto cdab = _mm_shuffle_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i *>(h.data())), 0xb1);
  auto hgfe =
      _mm_shuffle_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i *>(h.data() + 4)), 0x1b);
  auto abef = _mm_alignr_epi8(cdab, hgfe, 8), cdgh = _mm_blend_epi16(hgfe, cdab, 0xf0);
  for (; blocks; --blocks, data += 64) {
    const auto saved_abef = abef, saved_cdgh = cdgh;
    auto m0 = x86::load(data, 0), m1 = x86::load(data, 1), m2 = x86::load(data, 2),
         m3 = x86::load(data, 3);
    x86::rounds(abef, cdgh, m0, 0);
    x86::rounds(abef, cdgh, m1, 1);
    x86::rounds(abef, cdgh, m2, 2);
    x86::rounds(abef, cdgh, m3, 3);
    for (unsigned group = 4; group < 16; group += 4) {
      m0 = x86::next(m0, m1, m2, m3);
      x86::rounds(abef, cdgh, m0, group);
      m1 = x86::next(m1, m2, m3, m0);
      x86::rounds(abef, cdgh, m1, group + 1);
      m2 = x86::next(m2, m3, m0, m1);
      x86::rounds(abef, cdgh, m2, group + 2);
      m3 = x86::next(m3, m0, m1, m2);
      x86::rounds(abef, cdgh, m3, group + 3);
    }
    abef = _mm_add_epi32(abef, saved_abef);
    cdgh = _mm_add_epi32(cdgh, saved_cdgh);
  }
  const auto feba = _mm_shuffle_epi32(abef, 0x1b), dchg = _mm_shuffle_epi32(cdgh, 0xb1);
  _mm_storeu_si128(reinterpret_cast<__m128i *>(h.data()), _mm_blend_epi16(feba, dchg, 0xf0));
  _mm_storeu_si128(reinterpret_cast<__m128i *>(h.data() + 4), _mm_alignr_epi8(dchg, feba, 8));
}
#elif defined(__aarch64__) && !defined(_MSC_VER)
#if defined(__clang__)
#define CNEFX_SHA_TARGET __attribute__((target("sha2")))
#else
#define CNEFX_SHA_TARGET __attribute__((target("+sha2")))
#endif
namespace arm {
CNEFX_SHA_TARGET inline void rounds(uint32x4_t &abcd, uint32x4_t &efgh, uint32x4_t message,
                                    unsigned group) {
  const auto schedule = vaddq_u32(message, vld1q_u32(constants + 4 * group));
  const auto previous = abcd;
  abcd = vsha256hq_u32(abcd, efgh, schedule);
  efgh = vsha256h2q_u32(efgh, previous, schedule);
}
CNEFX_SHA_TARGET inline uint32x4_t next(uint32x4_t a, uint32x4_t b, uint32x4_t c, uint32x4_t d) {
  return vsha256su1q_u32(vsha256su0q_u32(a, b), c, d);
}
} // namespace arm
#if defined(__clang__)
__attribute__((target("sha2"), noinline))
#else
__attribute__((target("+sha2"), noinline))
#endif
void hardware_blocks(std::array<std::uint32_t, 8> &h, const std::uint8_t *data,
                     std::size_t blocks) {
  auto abcd = vld1q_u32(h.data()), efgh = vld1q_u32(h.data() + 4);
  for (; blocks; --blocks, data += 64) {
    const auto saved_abcd = abcd, saved_efgh = efgh;
    const auto load = [&](unsigned i) {
      return vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(data + 16 * i)));
    };
    auto m0 = load(0), m1 = load(1), m2 = load(2), m3 = load(3);
    arm::rounds(abcd, efgh, m0, 0);
    arm::rounds(abcd, efgh, m1, 1);
    arm::rounds(abcd, efgh, m2, 2);
    arm::rounds(abcd, efgh, m3, 3);
    for (unsigned group = 4; group < 16; group += 4) {
      m0 = arm::next(m0, m1, m2, m3);
      arm::rounds(abcd, efgh, m0, group);
      m1 = arm::next(m1, m2, m3, m0);
      arm::rounds(abcd, efgh, m1, group + 1);
      m2 = arm::next(m2, m3, m0, m1);
      arm::rounds(abcd, efgh, m2, group + 2);
      m3 = arm::next(m3, m0, m1, m2);
      arm::rounds(abcd, efgh, m3, group + 3);
    }
    abcd = vaddq_u32(abcd, saved_abcd);
    efgh = vaddq_u32(efgh, saved_efgh);
  }
  vst1q_u32(h.data(), abcd);
  vst1q_u32(h.data() + 4, efgh);
}
#else
void hardware_blocks(std::array<std::uint32_t, 8> &, const std::uint8_t *, std::size_t) {}
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
  void blocks(const std::uint8_t *p, std::size_t count) {
    if (accelerated_) {
      hardware_blocks(h_, p, count);
      return;
    }
    for (; count; --count, p += 64)
      compress(p);
  }

public:
  explicit Hash(bool accelerate) : accelerated_(accelerate && sha256_hardware_available()) {}
  void update(std::span<const std::uint8_t> data) {
    total_ += data.size();
    while (!data.empty()) {
      if (!used_ && data.size() >= 64) {
        const auto count = data.size() / 64;
        blocks(data.data(), count);
        data = data.subspan(count * 64);
        continue;
      }
      const auto n = std::min(data.size(), 64 - used_);
      std::memcpy(block_.data() + used_, data.data(), n);
      used_ += n;
      data = data.subspan(n);
      if (used_ == 64) {
        blocks(block_.data(), 1);
        used_ = 0;
      }
    }
  }
  std::string finish() {
    const auto bits = total_ * 8;
    block_[used_++] = 0x80;
    if (used_ > 56) {
      std::fill(block_.begin() + used_, block_.end(), 0);
      blocks(block_.data(), 1);
      used_ = 0;
    }
    std::fill(block_.begin() + used_, block_.begin() + 56, 0);
    for (unsigned i = 0; i < 8; ++i)
      block_[63 - i] = (bits >> (8 * i)) & 255;
    blocks(block_.data(), 1);
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
