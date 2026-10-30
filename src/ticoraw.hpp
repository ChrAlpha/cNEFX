#pragma once

#include "io.hpp"

namespace cnefx::detail {
// Decodes the TicoRAW codestream of a Nikon High Efficiency (HE or HE*) RAW
// strip into the width x height RGGB mosaic of 14-bit samples. Only the
// observed camera profile is accepted; anything else is rejected rather than
// approximated.
Plane decode_ticoraw(std::span<const std::uint8_t> codestream, std::uint32_t width,
                     std::uint32_t height, const Check &check = {});
} // namespace cnefx::detail
