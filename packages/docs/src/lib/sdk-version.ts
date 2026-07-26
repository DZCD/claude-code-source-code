import sdkPackage from "../../../agent-sdk/package.json";

/**
 * The version these docs describe, read from the SDK package at build time so a
 * release can never leave the documentation claiming an older one.
 */
export const SDK_VERSION: string = sdkPackage.version;

/**
 * Prepended to the agent-facing entry points.
 *
 * The warning is not boilerplate: the SDK ignores unknown options rather than
 * rejecting them, so an agent that reads current docs and writes `autoCompact`
 * against an older install gets a successful run with the feature silently
 * absent. Stating the version is what lets it notice.
 */
export const VERSION_NOTE = `This documentation describes agent-lattice v${SDK_VERSION}.

If the package is already installed, check the installed version before writing
code against these pages, and upgrade if it is older. Unknown options are ignored
rather than rejected, so calling a newer API on an older install succeeds with the
feature silently doing nothing — there is no error to catch. Install or upgrade
with \`npm install agent-lattice@${SDK_VERSION}\`.`;
