#!/usr/bin/env node
import { spawn } from 'node:child_process'
import { constants } from 'node:os'
import { fileURLToPath } from 'node:url'

const targets: Record<string, string> = {
  'linux-x64': 'linux-x86_64',
  'linux-arm64': 'linux-arm64',
  'darwin-arm64': 'macos-arm64',
  'win32-x64': 'windows-x86_64',
}
const target = targets[`${process.platform}-${process.arch}`]
if (!target) {
  console.error(`cnefx: Unsupported platform ${process.platform}/${process.arch}; build cNEFX from source.`)
  process.exit(1)
}
const name = process.platform === 'win32' ? 'cnefx.exe' : 'cnefx'
const executable = fileURLToPath(new URL(`../native/${target}/bin/${name}`, import.meta.url))

const child = spawn(executable, process.argv.slice(2), { stdio: 'inherit' })
// Stay alive until the native process reports cancellation. On Windows the
// child receives Ctrl+C from the shared console, and kill() would terminate it.
for (const signal of ['SIGINT', 'SIGTERM'] as const) {
  process.on(signal, () => {
    if (process.platform !== 'win32')
      child.kill(signal)
  })
}
child.on('error', (error) => {
  console.error(`cnefx: Unable to start ${executable}: ${error.message}`)
  process.exit(1)
})
child.on('exit', (code, signal) => {
  process.exit(code ?? 128 + (signal ? constants.signals[signal] : 0))
})
