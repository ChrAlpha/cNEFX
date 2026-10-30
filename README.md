# cNEFX

Merge Nikon pixel-shift captures into NEFX files on Linux, macOS and Windows.
cNEFX accepts complete 4, 8, 16 or 32-frame sequences from the Z8, Zf, Z6III and
Z5II. It runs natively, with no Nikon software or Python runtime required.

**Experimental:** validation against authentic captures and matching NX Studio
references, and Lightroom compatibility testing, remain pending for all models.

## Install

Install with [mise](https://mise.jdx.dev/) (recommended):

```sh
mise use -g github:ChrAlpha/cNEFX@latest
cnefx merge --help
```

Or run without adding cNEFX to your global configuration:

```sh
mise x github:ChrAlpha/cNEFX@latest -- cnefx merge --help
```

For Node.js users, run with `pnx`:

```sh
pnx cnefx merge --help
```

Or install globally with `pnpm`:

```sh
pnpm add -g cnefx
```

Update with `mise upgrade github:ChrAlpha/cNEFX` or `pnpm update -g cnefx`.
You can also [build from source](#build-from-source) or download native archives
from [Releases](https://github.com/ChrAlpha/cNEFX/releases).
Builds cover Linux x86_64/ARM64, macOS Apple silicon and Windows x86_64.
Check the accompanying `.sha256` file, extract the archive, and add its `bin`
directory to `PATH`. Builds are unsigned and not notarized; older-system
compatibility is not established.

## Merge a sequence

Use **lossless-compressed, 14-bit FX NEF files** from one complete pixel-shift
sequence. Pass every file explicitly in capture order, with consecutive shot
numbers. Directory inputs and shell glob ordering are not a substitute for
capture order. HE/HE* is unsupported; HLG, other image areas and untested
firmware/settings remain unvalidated.

The destination must not exist, and its parent directory must already exist.
Four captures to one NEFX:

```sh
cnefx merge \
  shot-01.NEF shot-02.NEF shot-03.NEF shot-04.NEF \
  --output merged.nefx
```

Eight captures to two NEFX files in a new directory:

```sh
cnefx merge \
  shot-01.NEF shot-02.NEF shot-03.NEF shot-04.NEF \
  shot-05.NEF shot-06.NEF shot-07.NEF shot-08.NEF \
  --outputs 2 --output-directory merged-set
```

The directory contains the NEFX files and a `manifest.json` report. cNEFX checks
the output samples before publishing and never replaces an existing destination,
including an empty directory. A successful merge can still report cleanup or
sync warnings; keep the published output and read the warning before retrying.

### Output modes

`--outputs` defaults to `1`. Use `--output-directory` for multiple outputs.

| Captures | Allowed outputs | Dimensions per output, relative to the source crop |
| --- | --- | --- |
| 4 | 1 | Original width and height |
| 8 | 1, 2 | Original width and height |
| 16 | 1, 4 | 2x width and height for one output; original for four |
| 32 | 1, 2, 4, 8 | 2x width and height for one or two outputs; original for four or eight |

### Options

- `--jobs N`: set the worker limit (1-256; default: all cores). Use `--jobs 1`
  for lower memory use.
- `--chromatic-aberration-correction`: enable correction when supported by the
  capture metadata; disabled by default.
- `--json`: write a structured report to stdout for scripts.
- `cnefx merge --help`: show all options.

## Build from source

Requires a C++20 compiler and CMake 3.21+. The build downloads and statically
links a pinned LibRaw release.

```sh
git clone https://github.com/ChrAlpha/cNEFX.git
cd cNEFX
cmake --preset release
cmake --build --preset release
./build/release/cnefx merge --help
```

With Visual Studio, the executable is `build/release/Release/cnefx.exe`.
Use the executable's full path or add its directory to `PATH` for the merge
examples above.

## Development

See [AGENTS.md](https://github.com/ChrAlpha/cNEFX/blob/master/AGENTS.md) for the
source map, C++ API notes, build checks, packaging and contribution guidelines.
[Project scope](https://github.com/ChrAlpha/cNEFX/blob/master/research/project-scope.json)
tracks supported behavior and remaining validation gaps. The former Python
implementation is available at the
[`v0.0.1` tag](https://github.com/ChrAlpha/cNEFX/tree/v0.0.1).

## License

[MIT](https://github.com/ChrAlpha/cNEFX/blob/master/LICENSE). Distributions also
include LibRaw's third-party license files.

## Acknowledgements

- [Nikon NX Studio](https://www.nikonusa.com/content/nx-studio), for the reference
  NEFX outputs used in format research.
- [XinmiaoZhao's NikonFirmwarePatch / z8pixelshift](https://github.com/XinmiaoZhao/NikonFirmwarePatch/tree/v0.3.0/z8pixelshift),
  for its pixel-shift reconstruction research and arithmetic reference.
