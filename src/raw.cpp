#include "raw.hpp"

#include <libraw/libraw.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>

namespace cnefx::detail {
namespace {
// Decode a written output through its owned descriptor. LibRaw never reopens a
// pathname that could have been replaced after staging.
class Stream final : public LibRaw_abstract_datastream {
  std::FILE *file_;
  INT64 size_;

public:
  explicit Stream(File &file) : file_(file.stream()), size_(file.size()) { file.seek(0); }
  int valid() override { return 1; }
  int read(void *p, size_t size, size_t n) override {
    if (n > INT_MAX || (size && n > SIZE_MAX / size))
      return 0;
    return static_cast<int>(std::fread(p, size, n, file_));
  }
  int seek(INT64 offset, int whence) override {
#ifdef _WIN32
    return _fseeki64(file_, offset, whence);
#else
    return fseeko(file_, offset, whence);
#endif
  }
  INT64 tell() override {
#ifdef _WIN32
    return _ftelli64(file_);
#else
    return ftello(file_);
#endif
  }
  INT64 size() override { return size_; }
  int get_char() override { return std::fgetc(file_); }
  char *gets(char *p, int n) override { return std::fgets(p, n, file_); }
  int scanf_one(const char *format, void *value) override {
    return std::fscanf(file_, format, value);
  }
  int eof() override { return std::feof(file_); }
};
struct Control {
  const Check &check;
  std::exception_ptr failure;
  bool corrupt = false;
  static int progress(void *opaque, LibRaw_progress, int, int) noexcept {
    auto &c = *static_cast<Control *>(opaque);
    try {
      if (c.check)
        c.check();
      return 0;
    } catch (...) {
      c.failure = std::current_exception();
      return 1;
    }
  }
  static void error(void *opaque, const char *, INT64) {
    static_cast<Control *>(opaque)->corrupt = true;
  }
  void result(int code) {
    if (failure)
      std::rethrow_exception(failure);
    if (code != LIBRAW_SUCCESS)
      throw std::runtime_error(std::string("LibRaw: ") + libraw_strerror(code));
    if (corrupt)
      throw std::invalid_argument("LibRaw reported corrupt RAW data.");
  }
};
template <class Open> void decode(LibRaw &raw, Control &control, Open open) {
  raw.set_progress_handler(Control::progress, &control);
  raw.set_dataerror_handler(Control::error, &control);
  control.result(open());
  control.result(raw.unpack());
}
Calibration calibration(const LibRaw &raw, const std::string &model) {
  const auto &color = raw.imgdata.rawdata.color;
  Calibration result;
  result.white = color.maximum;
  const auto rows = color.cblack[4], cols = color.cblack[5];
  if (rows > 2 || cols > 2)
    throw std::invalid_argument("Unsupported spatial black levels.");
  constexpr unsigned channels[] = {0, 1, 3, 2};
  for (unsigned i = 0; i < 4; ++i) {
    const unsigned c = channels[i];
    const std::uint64_t value =
        std::uint64_t(color.black) + color.cblack[c] +
        ((rows && cols) ? color.cblack[6 + (i / 2 % rows) * cols + i % 2 % cols] : 0);
    if (value >= 16383)
      throw std::invalid_argument("Unsupported black level.");
    result.black[c] = value;
    result.gains[c] = color.cam_mul[c];
  }
  if (result.white != 16383 || !std::all_of(result.black.begin(), result.black.end(),
                                            [&](auto v) { return v == result.black[0]; }))
    throw std::invalid_argument("Unequal black levels or unsupported 14-bit sample range.");
  if (!std::all_of(result.gains.begin(), result.gains.end(),
                   [](auto v) { return std::isfinite(v) && v > 0; }) ||
      result.gains[1] != result.gains[3])
    throw std::invalid_argument("Missing white balance or unequal green gains are unsupported.");
  for (unsigned y = 0; y < 3; ++y)
    for (unsigned x = 0; x < 3; ++x)
      result.matrix[y * 3 + x] = raw.imgdata.color.cam_xyz[y][x];
  const auto &m = result.matrix;
  result.missing_matrix =
      model == "NIKON Z5_2" && std::all_of(m.begin(), m.end(), [](auto v) { return v == 0; });
  const auto determinant = m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6]) +
                           m[2] * (m[3] * m[7] - m[4] * m[6]);
  if (!result.missing_matrix && (!std::all_of(m.begin(), m.end(), [](auto v) {
        return std::isfinite(v);
      }) || std::abs(determinant) < 1e-6))
    throw std::invalid_argument("Missing or singular XYZ-to-camera matrix.");
  return result;
}
} // namespace
Source::Source(const fs::path &path) {
  const auto resolved = fs::canonical(path);
  initial = file_state(resolved);
  if (!initial.regular)
    throw std::invalid_argument("Input must be a regular file.");
  file = std::make_shared<File>(resolved);
  check();
}
void Source::check() const {
  if (file->state() != initial || file_state(file->path()) != initial)
    throw std::invalid_argument("Input changed during synthesis.");
}
std::shared_ptr<Memory> Source::read(const Check &check_cancel) {
  check();
  auto contents = read_contents(*file, check_cancel);
  check();
  if (!hash.empty() && contents.sha256 != hash)
    throw std::invalid_argument("Input hash changed during synthesis.");
  hash = std::move(contents.sha256);
  return std::move(contents.memory);
}
void Source::verify(const Check &check_cancel) const {
  check();
  const auto actual = sha256(*file, check_cancel);
  check();
  if (hash.empty() || actual != hash)
    throw std::invalid_argument("Input hash changed during synthesis.");
}
std::pair<Plane, Metadata> read_frame(Source &source, const Check &check) {
  if (check)
    check();
  // Metadata and LibRaw both read the hashed bytes, never the file again.
  const auto memory = source.read(check);
  auto metadata = read_metadata(read_tiff(memory));
  auto &m = metadata.sequence;
  Control control{check, {}};
  // LibRaw owns a large decoder state. Keep it off the caller's thread stack;
  // the source bytes and callback control must outlive its destructor.
  auto decoder = std::make_unique<LibRaw>();
  auto &raw = *decoder;
  decode(raw, control, [&] {
    const auto bytes = memory->bytes();
    return raw.open_buffer(bytes.data(), bytes.size());
  });
  const auto &sizes = raw.imgdata.sizes;
  if (!raw.imgdata.rawdata.raw_image || raw.imgdata.idata.colors != 3 ||
      std::strcmp(raw.imgdata.idata.cdesc, "RGBG") || raw.COLOR(0, 0) != 0 ||
      raw.COLOR(0, 1) != 1 || raw.COLOR(1, 0) != 3 || raw.COLOR(1, 1) != 2)
    throw std::invalid_argument("Decoded CFA layout is incompatible with the recovered mapping.");
  const auto &crop = sizes.raw_inset_crops[0];
  if (sizes.raw_width != m.width || sizes.raw_height != m.height || sizes.top_margin ||
      sizes.left_margin || sizes.pixel_aspect != 1 ||
      Crop{crop.cleft, crop.ctop, crop.cwidth, crop.cheight} != m.crop ||
      sizes.raw_pitch < m.width * sizeof(Sample) || sizes.raw_pitch % 2)
    throw std::invalid_argument("Decoded crop or pixel geometry disagrees with Nikon metadata.");
  m.calibration = calibration(raw, m.model);
  Plane plane{m.width, m.height, std::vector<Sample>(sample_count(m.width, m.height, 1))};
  for (unsigned y = 0; y < m.height; ++y) {
    if (check && y % 128 == 0)
      check();
    const auto *row = raw.imgdata.rawdata.raw_image + std::size_t(y) * sizes.raw_pitch / 2;
    std::copy_n(row, m.width, plane.pixels.begin() + std::size_t(y) * m.width);
  }
  if (*std::max_element(plane.pixels.begin(), plane.pixels.end()) > 16383)
    throw std::invalid_argument("RAW samples exceed 14 bits.");
  source.check();
  return {std::move(plane), std::move(metadata)};
}
void verify_rgb(const fs::path &path, const Rgb &expected, const Check &check) {
  File file(path);
  Stream stream(file);
  Control control{check, {}};
  auto decoder = std::make_unique<LibRaw>();
  auto &raw = *decoder;
  decode(raw, control, [&] { return raw.open_datastream(&stream); });
  const auto &sizes = raw.imgdata.sizes;
  const auto &data = raw.imgdata.rawdata;
  if (sizes.raw_width != expected.width || sizes.raw_height != expected.height ||
      (!data.color3_image && !data.color4_image) ||
      expected.pixels.size() != sample_count(expected.width, expected.height, 3))
    throw std::runtime_error("Written NEFX failed decoded geometry verification.");
  for (unsigned y = 0; y < expected.height; ++y) {
    if (check && y % 128 == 0)
      check();
    for (unsigned x = 0; x < expected.width; ++x) {
      const auto i = std::size_t(y) * expected.width + x;
      for (unsigned c = 0; c < 3; ++c) {
        const auto actual = data.color3_image ? data.color3_image[i][c] : data.color4_image[i][c];
        if (actual != expected.pixels[i * 3 + c])
          throw std::runtime_error("Written NEFX failed exact pixel verification.");
      }
    }
  }
}
const char *decoder_version() { return LibRaw::version(); }
} // namespace cnefx::detail
