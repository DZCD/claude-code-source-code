import { spawnSync } from "node:child_process";
import { createRequire } from "node:module";
import { mkdirSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const require = createRequire(import.meta.url);
const JavaScriptObfuscator = require("javascript-obfuscator");

const packageDir = dirname(dirname(fileURLToPath(import.meta.url)));
const distDir = join(packageDir, "dist");
const bundlePath = join(distDir, "index.bundle.js");
const outputPath = join(distDir, "index.js");

function run(command, args) {
  const result = spawnSync(command, args, {
    cwd: packageDir,
    stdio: "inherit",
    shell: process.platform === "win32",
  });
  if (result.status !== 0) {
    throw new Error(`${command} ${args.join(" ")} failed with status ${result.status}`);
  }
}

function formatSize(bytes) {
  return `${(bytes / 1024).toFixed(2)} KiB`;
}

rmSync(distDir, { recursive: true, force: true });
mkdirSync(distDir, { recursive: true });

console.log("==> Bundling ESM entry");
run("bun", [
  "build",
  "./src/index.ts",
  "--outfile",
  "./dist/index.bundle.js",
  "--target",
  "node",
  "--format",
  "esm",
]);

console.log("==> Obfuscating published JavaScript");
const bundleCode = readFileSync(bundlePath, "utf8");
const obfuscationResult = JavaScriptObfuscator.obfuscate(bundleCode, {
  compact: true,
  controlFlowFlattening: false,
  deadCodeInjection: false,
  debugProtection: false,
  disableConsoleOutput: false,
  identifierNamesGenerator: "hexadecimal",
  renameGlobals: false,
  selfDefending: false,
  simplify: true,
  splitStrings: false,
  stringArray: true,
  stringArrayEncoding: ["none"],
  stringArrayThreshold: 0.75,
  transformObjectKeys: false,
  unicodeEscapeSequence: false,
});

writeFileSync(outputPath, `${obfuscationResult.getObfuscatedCode()}\n`);
rmSync(bundlePath, { force: true });

console.log("==> Emitting TypeScript declarations");
run("tsc", ["-p", "tsconfig.json"]);

const outputSize = statSync(outputPath).size;
console.log(`==> Built ${outputPath} (${formatSize(outputSize)})`);
