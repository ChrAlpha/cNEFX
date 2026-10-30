#include "ticoraw.hpp"

#include <algorithm>
#include <bit>

// Nikon HE and HE* store the Bayer mosaic as an intoPIX TicoRAW codestream: JPEG XS
// (ISO/IEC 21122-1) entropy coding and 5/3 wavelets over four half-resolution
// Star-Tetrix components, with vendor header extensions, packet order and output
// curve. Every observed camera writes the same main header except for the frame
// size and codestream length.
namespace cnefx::detail {
namespace {
constexpr unsigned band_count = 25, packet_count = 8, precinct_header = 12, packet_header = 7;
constexpr int bit_depth = 18, fraction_bits = 4;
constexpr std::uint8_t capability[] = {'C', 'O', 'N', 'T', 'A', 'C',  'T',  '_', 'I', 'N', 'T',
                                       'O', 'P', 'I', 'X', '_', 0xef, 0xc0, 0,   0,   0,   0,
                                       0,   0,   0,   0,   0,   0,    0,    0,   0,   0};
// Nc, Ng, Ss, Bw, Fq/Br, Fslc/Ppoc/Cpih, NLx/NLy, Lh/Rl/Qpih/Fs/Rm.
constexpr std::uint8_t profile[] = {4, 4, 8, bit_depth, 0x44, 0x10, 0x51, 0x14};
// Per component: bit depth, sampling and vendor decomposition levels. Component 2
// carries no wavelet.
constexpr std::uint8_t components[] = {14, 0x11, 0x51, 14, 0x11, 0x51, 14, 0x11, 0, 14, 0x11, 0x51};

void require(bool ok, const char *reason) {
  if (!ok)
    throw std::invalid_argument(reason);
}
void corrupt(bool ok) { require(ok, "Corrupt Nikon HE codestream."); }
unsigned be16(std::span<const std::uint8_t> data, std::size_t offset) {
  corrupt(offset + 2 <= data.size());
  return unsigned(data[offset]) << 8 | data[offset + 1];
}

// MSB-first reader bounded to one sub-packet.
class Bits {
  std::span<const std::uint8_t> data_;
  std::size_t bit_ = 0;

  std::uint64_t window() const {
    std::uint64_t word = 0;
    for (std::size_t i = bit_ / 8, end = i + 8; i < end; ++i)
      word = word << 8 | (i < data_.size() ? data_[i] : 0);
    return word << (bit_ % 8);
  }

public:
  explicit Bits(std::span<const std::uint8_t> data) : data_(data) {}
  unsigned read(unsigned n) {
    corrupt(bit_ + n <= data_.size() * 8);
    const auto value = n ? unsigned(window() >> (64 - n)) : 0;
    bit_ += n;
    return value;
  }
  int unary() {
    const auto ones = unsigned(std::countl_one(window()));
    corrupt(ones <= 2 * bit_depth + 16 && bit_ + ones + 1 <= data_.size() * 8);
    bit_ += ones + 1;
    return int(ones);
  }
  std::size_t bytes() const { return (bit_ + 7) / 8; }
};

struct Band {
  unsigned component, x, width, row, lines;
  unsigned groups() const { return (width + 3) / 4; }
};
struct Layout {
  std::vector<Band> bands;
  std::array<std::vector<std::pair<unsigned, unsigned>>, packet_count> packets;
  std::array<unsigned, band_count> weight{};
};
// Low-pass widths after each of the five horizontal splits.
std::array<unsigned, 6> lows(unsigned width) {
  std::array<unsigned, 6> result{width};
  for (unsigned level = 1; level < result.size(); ++level)
    result[level] = result[level - 1] - result[level - 1] / 2;
  return result;
}
Layout layout(unsigned width) {
  Layout result;
  const auto low = lows(width);
  for (unsigned c = 0; c < 4; ++c) {
    if (c == 2) {
      result.bands.push_back({c, 0, width, 0, 2});
      continue;
    }
    // Vertical-low line [L5 | H5 | H4 | H3 | H2 | H1], then vertical-high [L1 | H1].
    result.bands.push_back({c, 0, low[5], 0, 1});
    for (unsigned level = 5; level >= 1; --level)
      result.bands.push_back({c, low[level], low[level - 1] - low[level], 0, 1});
    result.bands.push_back({c, 0, low[1], 1, 1});
    result.bands.push_back({c, low[1], width - low[1], 1, 1});
  }
  // Nikon orders packets by vertical half, then component; weights follow the
  // order in which bands first appear.
  std::array<bool, band_count> seen{};
  unsigned next = 0;
  for (unsigned half = 0; half < 2; ++half)
    for (unsigned c = 0; c < 4; ++c)
      for (unsigned b = 0; b < band_count; ++b) {
        const auto &band = result.bands[b];
        if (band.component != c || (band.lines == 1 && band.row != half))
          continue;
        result.packets[half * 4 + c].emplace_back(b, band.lines == 2 ? half : 0);
        if (!seen[b]) {
          seen[b] = true;
          result.weight[b] = next++;
        }
      }
  return result;
}

struct Header {
  std::array<std::pair<int, int>, band_count> weights{};
  unsigned slice_height{};
  std::int64_t black{}, gain{}, knee{};
  std::size_t body{};
};
Header header(std::span<const std::uint8_t> data, std::uint32_t width, std::uint32_t height) {
  const auto segment = [&](std::size_t &position, unsigned marker, unsigned length) {
    require(be16(data, position) == marker && be16(data, position + 2) == length,
            "Unsupported Nikon HE codestream profile.");
    corrupt(length <= data.size() - position - 2);
    const auto payload = data.subspan(position + 4, length - 2);
    position += 2 + length;
    return payload;
  };
  corrupt(data.size() >= 4);
  require(be16(data, 0) == 0xff10, "Expected a Nikon HE codestream.");
  std::size_t position = 2;
  const auto cap = segment(position, 0xff50, 34);
  require(std::equal(cap.begin(), cap.end(), capability),
          "Unsupported Nikon HE codestream profile.");
  const auto pih = segment(position, 0xff12, 39);
  const auto cdt = segment(position, 0xff13, 14);
  const auto wgt = segment(position, 0xff14, 52);
  const auto length = std::uint64_t(be16(pih, 0)) << 16 | be16(pih, 2);
  require(length == data.size() && std::uint64_t(width) * height <= 16 * length,
          "Nikon HE codestream length disagrees with its RAW strip.");
  require(be16(pih, 4) == 0 && be16(pih, 6) == 0 && be16(pih, 8) == width &&
              be16(pih, 10) == height && be16(pih, 12) == 0 && be16(pih, 14) &&
              std::equal(std::begin(profile), std::end(profile), pih.begin() + 16) &&
              std::equal(std::begin(components), std::end(components), cdt.begin()),
          "Unsupported Nikon HE codestream profile.");
  require(width % 2 == 0 && height % 4 == 0 && width / 2 >= 32,
          "Unsupported Nikon HE frame geometry.");
  Header result;
  for (unsigned b = 0; b < band_count; ++b)
    result.weights[b] = {wgt[2 * b], wgt[2 * b + 1]};
  result.slice_height = be16(pih, 14);
  const auto *t = pih.data() + 24;
  result.black = (t[3] & 0x0f) << 8 | t[4];
  result.gain = (t[7] & 0x03) << 16 | t[8] << 8 | t[9];
  result.knee = t[10] << 8 | t[11];
  result.body = position;
  return result;
}

using Planes = std::array<std::vector<std::int32_t>, 4>;
// Entropy-decodes every slice into scaled wavelet coefficients.
void entropy(std::span<const std::uint8_t> data, const Header &header, const Layout &layout,
             unsigned width, unsigned height, Planes &planes, const Check &check) {
  const unsigned precincts = height / 2,
                 slices = (precincts + header.slice_height - 1) / header.slice_height;
  std::array<std::vector<int>, band_count> gcli, top;
  std::array<std::vector<std::uint32_t>, band_count> magnitude;
  std::array<std::vector<std::uint8_t>, band_count> insignificant;
  for (unsigned b = 0; b < band_count; ++b) {
    const auto groups = layout.bands[b].groups();
    gcli[b].resize(groups);
    magnitude[b].resize(groups * 4);
    insignificant[b].resize((groups + 7) / 8);
  }
  std::array<int, band_count> gtli{}, top_gtli{};
  std::array<unsigned, band_count> mode{};
  std::array<bool, band_count> reference{};
  auto position = header.body;
  for (unsigned slice = 0; slice < slices; ++slice) {
    if (check)
      check();
    corrupt(be16(data, position) == 0xff20 && be16(data, position + 2) == 4 &&
            be16(data, position + 4) == slice);
    position += 6;
    reference.fill(false);
    const auto first = slice * header.slice_height,
               last = std::min(precincts, first + header.slice_height);
    for (unsigned p = first; p < last; ++p) {
      corrupt(position + precinct_header <= data.size());
      Bits head(data.subspan(position, precinct_header));
      const auto size = std::size_t(head.read(24));
      const int q = int(head.read(8)), r = int(head.read(8));
      for (unsigned b = 0; b < band_count; ++b) {
        mode[b] = head.read(2);
        const auto [gain, priority] = header.weights[layout.weight[b]];
        const int pump = priority < r ? 1 : 0;
        gtli[b] = q < gain + pump ? 0 : std::min(q - gain - pump, 15);
      }
      position += precinct_header;
      corrupt(size <= data.size() - position);
      const auto end = position + size;
      for (const auto &packet : layout.packets) {
        corrupt(position + packet_header <= end);
        Bits lengths(data.subspan(position, packet_header));
        const bool raw = lengths.read(1);
        const std::size_t data_bytes = lengths.read(20), gcli_bytes = lengths.read(20),
                          sign_bytes = lengths.read(15);
        require(!raw, "Unsupported raw-GCLI Nikon HE packet.");
        position += packet_header;
        const auto take = [&](std::size_t bytes) {
          corrupt(bytes <= end - position);
          const auto part = data.subspan(position, bytes);
          position += bytes;
          return Bits(part);
        };
        std::size_t flags = 0;
        for (const auto &[b, line] : packet)
          if (mode[b] & 2)
            flags += insignificant[b].size();
        auto significance = take((flags + 7) / 8);
        for (const auto &[b, line] : packet)
          if (mode[b] & 2)
            for (auto &flag : insignificant[b])
              flag = std::uint8_t(significance.read(1));
        auto codes = take(gcli_bytes);
        for (const auto &[b, line] : packet) {
          const bool vertical = mode[b] & 1, coded = mode[b] & 2;
          // The first precinct of a slice predicts from a virtual all-zero line.
          const int above = reference[b] ? top_gtli[b] : gtli[b];
          for (unsigned i = 0; i < gcli[b].size(); ++i) {
            const bool skip = coded && insignificant[b][i / 8];
            int value;
            if (!vertical) {
              value = skip ? gtli[b] : gtli[b] + codes.unary();
            } else {
              const int predicted =
                  std::max(reference[b] ? top[b][i] : 0, std::max(above, gtli[b]));
              value = predicted;
              if (!skip) {
                const int x = codes.unary(), threshold = std::max(predicted - gtli[b], 0);
                value += x > 2 * threshold ? x - threshold : x % 2 ? -(x + 1) / 2 : x / 2;
              }
            }
            corrupt(value <= bit_depth);
            gcli[b][i] = value;
          }
          top[b] = gcli[b];
          top_gtli[b] = gtli[b];
          reference[b] = true;
        }
        corrupt(codes.bytes() == gcli_bytes);
        auto bits = take(data_bytes);
        for (const auto &[b, line] : packet)
          for (unsigned i = 0; i < gcli[b].size(); ++i) {
            const int planes_coded = gcli[b][i] - gtli[b];
            auto *group = magnitude[b].data() + i * 4;
            std::fill_n(group, 4, 0u);
            if (planes_coded <= 0)
              continue;
            corrupt(planes_coded <= 15);
            for (int k = 0; k < planes_coded; ++k) {
              const auto nibble = bits.read(4);
              for (unsigned j = 0; j < 4; ++j)
                group[j] = group[j] << 1 | (nibble >> (3 - j) & 1);
            }
            for (unsigned j = 0; j < 4; ++j)
              group[j] <<= gtli[b];
          }
        corrupt(bits.bytes() == data_bytes);
        auto signs = take(sign_bytes);
        for (const auto &[b, line] : packet) {
          const auto &band = layout.bands[b];
          auto *row = planes[band.component].data() + std::size_t(2 * p + band.row + line) * width;
          for (unsigned k = 0; k < magnitude[b].size(); ++k) {
            auto v = magnitude[b][k];
            // Coefficients past the band end are discarded, but their signs are coded.
            const bool negative = v && signs.read(1);
            if (k >= band.width)
              continue;
            const int planes_coded = gcli[b][k / 4] - gtli[b];
            if (v && gtli[b] > 0) {
              std::uint32_t sum = 0;
              for (; v; v >>= planes_coded + 1)
                sum += v;
              v = sum;
            }
            const auto value = std::int32_t(v) * (1 << fraction_bits);
            row[band.x + k] = negative ? -value : value;
          }
        }
        corrupt(signs.bytes() == sign_bytes);
      }
      // Constant-rate precincts end in zero padding; sub-packets carry none.
      corrupt(std::all_of(data.begin() + position, data.begin() + end, [](auto v) { return !v; }));
      position = end;
    }
  }
  corrupt(be16(data, position) == 0xff11 && position + 2 == data.size());
}

std::int32_t wrap(std::int64_t value) { return static_cast<std::int32_t>(value); }
// Reversible LeGall 5/3 synthesis of n >= 2 samples stored as [low | high].
void synthesize(std::int32_t *samples, unsigned n, std::vector<std::int32_t> &out) {
  const unsigned low = n - n / 2, high = n / 2;
  const auto *l = samples, *h = samples + low;
  for (unsigned k = 0; k < low; ++k)
    out[2 * k] =
        wrap(l[k] - ((std::int64_t(h[k ? k - 1 : 0]) + h[std::min(k, high - 1)] + 2) >> 2));
  for (unsigned k = 0; k < high; ++k)
    out[2 * k + 1] =
        wrap(h[k] + ((std::int64_t(out[2 * k]) + out[2 * k + 2 < n ? 2 * k + 2 : 2 * k]) >> 1));
  std::copy_n(out.begin(), n, samples);
}
void wavelet(std::vector<std::int32_t> &plane, unsigned width, unsigned height,
             const Check &check) {
  const auto low = lows(width);
  std::vector<std::int32_t> out(width);
  for (unsigned y = 0; y < height; ++y) {
    if (check && y % 256 == 0)
      check();
    auto *row = plane.data() + std::size_t(y) * width;
    if (y % 2)
      synthesize(row, width, out);
    else
      for (unsigned level = 5; level >= 1; --level)
        synthesize(row, low[level - 1], out);
  }
  // One vertical level across the whole frame; slices do not restart it.
  const auto row = [&](unsigned y) { return plane.data() + std::size_t(y) * width; };
  const unsigned precincts = height / 2;
  for (unsigned p = 0; p < precincts; ++p) {
    auto *even = row(2 * p);
    const auto *above = row(2 * (p ? p - 1 : 0) + 1), *below = row(2 * p + 1);
    for (unsigned x = 0; x < width; ++x)
      even[x] = wrap(even[x] - ((std::int64_t(above[x]) + below[x] + 2) >> 2));
  }
  for (unsigned p = 0; p < precincts; ++p) {
    auto *odd = row(2 * p + 1);
    const auto *above = row(2 * p), *below = row(p + 1 < precincts ? 2 * p + 2 : 2 * p);
    for (unsigned x = 0; x < width; ++x)
      odd[x] = wrap(odd[x] + ((std::int64_t(above[x]) + below[x]) >> 1));
  }
}

// Inverse Star-Tetrix with Nikon's quad geometry: R G / G B holds components
// Cr Y / Delta Cb. Neighbours outside the mosaic reflect, which clamps plane indices.
void star_tetrix(Planes &planes, unsigned width, unsigned height, const Check &check) {
  auto &y_plane = planes[0], &r_plane = planes[1], &d_plane = planes[2], &b_plane = planes[3];
  const auto step = [&](std::vector<std::int32_t> &target, int shift, int sign, auto &&neighbours) {
    if (check)
      check();
    for (unsigned y = 0; y < height; ++y)
      for (unsigned x = 0; x < width; ++x) {
        const unsigned ym = y ? y - 1 : 0, yp = std::min(y + 1, height - 1), xm = x ? x - 1 : 0,
                       xp = std::min(x + 1, width - 1);
        auto &value = target[std::size_t(y) * width + x];
        value = wrap(value + sign * (neighbours(ym, y, yp, xm, x, xp) >> shift));
      }
  };
  const auto at = [width](const std::vector<std::int32_t> &plane, unsigned y, unsigned x) {
    return std::int64_t(plane[std::size_t(y) * width + x]);
  };
  step(y_plane, 3, -1, [&](unsigned ym, unsigned y, unsigned, unsigned, unsigned x, unsigned xp) {
    return at(d_plane, ym, x) + at(d_plane, ym, xp) + at(d_plane, y, x) + at(d_plane, y, xp);
  });
  step(d_plane, 2, 1, [&](unsigned, unsigned y, unsigned yp, unsigned xm, unsigned x, unsigned) {
    return at(y_plane, y, xm) + at(y_plane, y, x) + at(y_plane, yp, xm) + at(y_plane, yp, x);
  });
  step(y_plane, 3, -1, [&](unsigned ym, unsigned y, unsigned, unsigned, unsigned x, unsigned xp) {
    return at(r_plane, y, x) + at(r_plane, y, xp) + at(b_plane, ym, x) + at(b_plane, y, x);
  });
  step(d_plane, 3, -1, [&](unsigned, unsigned y, unsigned yp, unsigned xm, unsigned x, unsigned) {
    return at(b_plane, y, xm) + at(b_plane, y, x) + at(r_plane, y, x) + at(r_plane, yp, x);
  });
  step(r_plane, 2, 1, [&](unsigned ym, unsigned y, unsigned, unsigned xm, unsigned x, unsigned) {
    return at(y_plane, y, xm) + at(y_plane, y, x) + at(d_plane, ym, x) + at(d_plane, y, x);
  });
  step(b_plane, 2, 1, [&](unsigned, unsigned y, unsigned yp, unsigned, unsigned x, unsigned xp) {
    return at(d_plane, y, x) + at(d_plane, y, xp) + at(y_plane, y, x) + at(y_plane, yp, x);
  });
}

// Vendor quadratic output curve from the picture-header extension.
std::vector<Sample> curve(const Header &header) {
  std::vector<Sample> table(65536);
  for (std::int64_t i = 0; i < 65536; ++i) {
    const auto d = i - header.knee;
    const auto scaled = (d * header.gain + 32768) >> 16;
    const auto q = d < 0 ? -(d * d) : scaled * scaled;
    table[i] = Sample(std::clamp<std::int64_t>(header.black + ((q + 163840) >> 18), 0, 16383));
  }
  return table;
}
} // namespace

Plane decode_ticoraw(std::span<const std::uint8_t> codestream, std::uint32_t width,
                     std::uint32_t height, const Check &check) {
  const auto parsed = header(codestream, width, height);
  const unsigned w = width / 2, h = height / 2;
  Planes planes;
  for (auto &plane : planes)
    plane.assign(sample_count(w, h, 1), 0);
  entropy(codestream, parsed, layout(w), w, h, planes, check);
  for (unsigned c : {0u, 1u, 3u})
    wavelet(planes[c], w, h, check);
  star_tetrix(planes, w, h, check);
  const auto table = curve(parsed);
  const auto map = [&](std::int32_t v) {
    return table[std::clamp<std::int64_t>(std::int64_t(v) + 32768, 0, 65535)];
  };
  Plane result{width, height, std::vector<Sample>(sample_count(width, height, 1))};
  for (unsigned y = 0; y < h; ++y) {
    if (check && y % 256 == 0)
      check();
    auto *top = result.pixels.data() + std::size_t(2 * y) * width, *bottom = top + width;
    for (unsigned x = 0; x < w; ++x) {
      const auto i = std::size_t(y) * w + x;
      top[2 * x] = map(planes[1][i]);
      top[2 * x + 1] = map(planes[0][i]);
      bottom[2 * x] = map(planes[2][i]);
      bottom[2 * x + 1] = map(planes[3][i]);
    }
  }
  return result;
}
} // namespace cnefx::detail
