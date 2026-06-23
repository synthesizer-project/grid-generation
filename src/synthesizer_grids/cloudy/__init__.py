"""The Cloudy subpackage for generating, running and collecting Cloudy grids.

Cloudy 25 shares the Cloudy 23 interface, so ``cloudy25`` is exposed here as a
single canonical alias of ``cloudy23``. Importing the version modules from this
package (rather than redefining the alias in each module) keeps the Cloudy
version handling consistent across the subpackage.
"""

from synthesizer.photoionisation import cloudy17, cloudy23

# Cloudy 25 uses the same interface as Cloudy 23; alias for semantic clarity.
cloudy25 = cloudy23

__all__ = ["cloudy17", "cloudy23", "cloudy25"]
