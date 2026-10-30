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
std::string indices(const std::vector<unsigned> &values) {
  std::string out = "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i)
      out += ',';
    out += std::to_string(values[i]);
  }
  return out + ']';
}
} // namespace
std::string path_utf8(const std::filesystem::path &path) {
  const auto bytes = path.u8string();
  return std::string(bytes.begin(), bytes.end());
}
std::string report_json(const MergeReport &r) {
  const auto files = [](const std::vector<FileReport> &items) {
    std::string out = "[";
    for (std::size_t i = 0; i < items.size(); ++i)
      out += (i ? ",{\"path\":" : "{\"path\":") + quote(path_utf8(items[i].path)) +
             ",\"reason\":" + quote(items[i].reason) + '}';
    return out + ']';
  };
  constexpr const char *statuses[] = {"merged", "already_merged", "failed"};
  std::string out =
      "{\"schema_version\":2,\"status\":" + quote(r.complete() ? "ok" : "partial") +
      ",\"backend\":\"native\",\"output_format\":\"NEFX\",\"version\":" + quote(version()) +
      ",\"libraw_version\":" + quote(r.decoder_version) +
      ",\"sequence_validation\":\"burst_grouped_complete_consecutive_metadata_"
      "consistent\",\"chromatic_aberration_correction\":" +
      (r.options.chromatic_aberration_correction ? "true" : "false") + ",\"sequences\":[";
  for (std::size_t i = 0; i < r.sequences.size(); ++i) {
    const auto &sequence = r.sequences[i];
    out += (i ? ",{\"camera\":{\"make\":" : "{\"camera\":{\"make\":") + quote(sequence.make) +
           ",\"model\":" + quote(sequence.model) +
           "},\"first_capture\":" + quote(path_utf8(sequence.first_capture)) +
           ",\"captures\":" + std::to_string(sequence.captures) +
           ",\"status\":" + quote(statuses[static_cast<int>(sequence.status)]);
    if (!sequence.existing.empty()) {
      out += ",\"existing\":[";
      for (std::size_t j = 0; j < sequence.existing.size(); ++j)
        out += (j ? "," : "") + quote(path_utf8(sequence.existing[j]));
      out += ']';
    }
    if (sequence.status == SequenceStatus::failed)
      out += ",\"error\":" + quote(sequence.error);
    out += '}';
  }
  out += "],\"inputs\":[";
  for (std::size_t i = 0; i < r.inputs.size(); ++i) {
    const auto &input = r.inputs[i];
    out += (i ? ",{\"path\":" : "{\"path\":") + quote(path_utf8(input.path)) +
           ",\"sha256\":" + quote(input.sha256) +
           ",\"shot_number\":" + std::to_string(input.shot_number) +
           ",\"sequence_index\":" + std::to_string(input.sequence_index) + '}';
  }
  out += "],\"skipped\":" + files(r.skipped) + ",\"errors\":" + files(r.errors) +
         ",\"limitations\":" + strings(r.limitations) + ",\"warnings\":" + strings(r.warnings);
  if (r.options.directory)
    out += ",\"output_directory\":" + quote(path_utf8(r.options.output));
  out += ",\"output_count\":" + std::to_string(r.outputs.size()) + ",\"outputs\":[";
  std::uint64_t total = 0;
  for (std::size_t i = 0; i < r.outputs.size(); ++i) {
    const auto &o = r.outputs[i];
    out += (i ? ",{\"output\":" : "{\"output\":") + quote(path_utf8(o.path)) +
           ",\"sha256\":" + quote(o.sha256) + ",\"shape\":[" + std::to_string(o.height) + "," +
           std::to_string(o.width) +
           ",3],\"samples_verified\":" + std::to_string(o.samples_verified) +
           ",\"sequence_index\":" + std::to_string(o.sequence_index) +
           ",\"input_indices\":" + indices(o.input_indices) + '}';
    total += o.samples_verified;
  }
  return out + "],\"samples_verified\":" + std::to_string(total) + '}';
}
std::string error_json(const std::string &code, const std::string &message) {
  return "{\"schema_version\":2,\"status\":\"error\",\"backend\":\"native\",\"error\":{\"code\":" +
         quote(code) + ",\"message\":" + quote(message) + "}}";
}
} // namespace cnefx
