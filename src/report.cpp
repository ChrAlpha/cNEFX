#include "cnefx/merge.hpp"

#include <sstream>

namespace cnefx {
namespace {
void unicode(std::string &out, unsigned value) {
  constexpr char hex[] = "0123456789abcdef";
  out += "\\u";
  for (int shift = 12; shift >= 0; shift -= 4)
    out += hex[(value >> shift) & 15];
}
std::string quote(const std::string &text) {
  std::string out = "\"";
  for (std::size_t i = 0; i < text.size();) {
    const auto c = static_cast<unsigned char>(text[i]);
    if (c < 128) {
      if (c == '"' || c == '\\') {
        out += '\\';
        out += char(c);
      } else if (c < 32)
        unicode(out, c);
      else
        out += char(c);
      ++i;
      continue;
    }
    unsigned n = c >= 0xc2 && c <= 0xdf   ? 2
                 : c >= 0xe0 && c <= 0xef ? 3
                 : c >= 0xf0 && c <= 0xf4 ? 4
                                          : 0;
    unsigned code = n ? c & ((1u << (7 - n)) - 1) : 0;
    bool valid = n && n <= text.size() - i;
    for (unsigned j = 1; valid && j < n; ++j) {
      const auto b = static_cast<unsigned char>(text[i + j]);
      valid = (b & 0xc0) == 0x80;
      code = (code << 6) | (b & 63);
    }
    valid = valid &&
            code >= (n == 2   ? 0x80u
                     : n == 3 ? 0x800u
                              : 0x10000u) &&
            code <= 0x10ffff && !(code >= 0xd800 && code <= 0xdfff);
    if (!valid) {
      unicode(out, 0xdc00 + c);
      ++i;
      continue;
    }
    if (code < 65536)
      unicode(out, code);
    else {
      code -= 0x10000;
      unicode(out, 0xd800 + (code >> 10));
      unicode(out, 0xdc00 + (code & 1023));
    }
    i += n;
  }
  return out + '"';
}
std::string strings(const std::vector<std::string> &values) {
  std::string out = "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i)
      out += ',';
    out += quote(values[i]);
  }
  return out + "]";
}
std::string output(const OutputReport &o, bool indices) {
  std::string out = "\"output\":" + quote(path_utf8(o.path)) + ",\"sha256\":" + quote(o.sha256) +
                    ",\"shape\":[" + std::to_string(o.height) + "," + std::to_string(o.width) +
                    ",3],\"samples_verified\":" + std::to_string(o.samples_verified);
  if (indices) {
    out += ",\"input_indices\":[";
    for (std::size_t i = 0; i < o.input_indices.size(); ++i) {
      if (i)
        out += ',';
      out += std::to_string(o.input_indices[i]);
    }
    out += ']';
  }
  return out;
}
} // namespace
std::string path_utf8(const std::filesystem::path &path) {
  const auto bytes = path.u8string();
  return std::string(bytes.begin(), bytes.end());
}
std::string report_json(const MergeReport &r) {
  std::string out = "{\"schema_version\":1,\"status\":\"ok\",\"backend\":\"native\",\"output_"
                    "format\":\"NEFX\",\"version\":" +
                    quote(version()) + ",\"libraw_version\":" + quote(r.decoder_version) +
                    ",\"camera\":{\"make\":" + quote(r.make) + ",\"model\":" + quote(r.model) +
                    "},\"sequence_validation\":\"complete_consecutive_metadata_consistent\","
                    "\"chromatic_aberration_correction\":" +
                    (r.options.chromatic_aberration_correction ? "true" : "false") +
                    ",\"inputs\":[";
  for (std::size_t i = 0; i < r.inputs.size(); ++i) {
    if (i)
      out += ',';
    const auto &input = r.inputs[i];
    out += "{\"path\":" + quote(path_utf8(input.path)) + ",\"sha256\":" + quote(input.sha256) +
           ",\"shot_number\":" + std::to_string(input.shot_number) + '}';
  }
  out +=
      "],\"limitations\":" + strings(r.limitations) + ",\"warnings\":" + strings(r.warnings) + ",";
  if (r.options.directory) {
    out += "\"output_directory\":" + quote(path_utf8(r.options.output)) +
           ",\"output_count\":" + std::to_string(r.outputs.size()) + ",\"outputs\":[";
    std::uint64_t total = 0;
    for (std::size_t i = 0; i < r.outputs.size(); ++i) {
      if (i)
        out += ',';
      out += '{' + output(r.outputs[i], true) + '}';
      total += r.outputs[i].samples_verified;
    }
    out += "],\"samples_verified\":" + std::to_string(total);
  } else {
    if (r.outputs.size() != 1)
      throw std::logic_error("A single-output report needs one output.");
    out += output(r.outputs.front(), false);
  }
  return out + '}';
}
std::string error_json(const std::string &code, const std::string &message) {
  return "{\"schema_version\":1,\"status\":\"error\",\"backend\":\"native\",\"error\":{\"code\":" +
         quote(code) + ",\"message\":" + quote(message) + "}}";
}
} // namespace cnefx
