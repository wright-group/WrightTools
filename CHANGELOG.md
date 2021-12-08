# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

## [3.4.3]

### Added
- `artists` now has a wrapper for matplotlib's `imshow`.  Make sure to use uniform grids.

### Fixed
- `artists._parse_limits` now recognizes channel `null` for signed data limits.
- fixed bug where `Data.map_variable` did not interpolate on `yaqc-cmds` type Data

## [3.4.2]

### Added
- `data.from_Solis`: "kinetic series" acquisition type now supported.

## [3.4.1]

### Added
- `artists.create_figure`: kwarg `margin` allows unique margins on all sides

### Fixed
- ZeroDivisionError when converting units (e.g. wn to nm) now returns inf instead of raising
- PermissionError on windows when copying Data/Collection objects
- Fixed bug in `wt.artists.interact2D` where constants were not handled properly.

## [3.4.0]

### Added
- `from_databroker` method to import Data objects from databroker catalogs

### Changed
- complete units overhaul, now using pint library
- explicitly store axes as fixed string dtype
- validate units are in the unit registry on set

### Fixed
- Fixed bug in `from_COLORS` that misordered the relationship between variables and channels for 1D datasets.
- Avoid passing both `vmin/vmax` and `norm` to `pcolor*` methods
- Fixed matplotlib depreciation in artists._helpers

## [3.3.3]

## Added
- Collection.convert method for converting all units of a single kind of a collection's data to another unit (with test script)

### Fixed
- Absence of verbose kwarg propogation in data.split, silenced convert call.
- Further handling for readonly files
- Improved chopping with axes that span the kept axes removed
- Timezone offset for `wt.kit.TimeStamp.RFC3339`

## [3.3.2]

## Added
- `wt_for_numpy_users` Jupyter notebook illustrating differences between numpy and WrightTools workflow

### Fixed
- Return min and max when file is read only
- try/except the os.close call for tempfile (may already be gone in SWMR mode)
- data objects retain units of individual axes when copied
- h5py 3.0 string handling

## [3.3.1]

### Added
- pytest and pytest-cov to dev requirements

### Fixed
- representation of units with micro prefix
- Do not attempt to write attrs when file is read only

## [3.3.0]

### Added
- lineshapes in kit: `gaussian`, `lorentzian_complex`, `lorentzian_real`, `voight`

### Changed
- misc text effects within `artists.corner_text`
- deprecate `pcolor_helper`, remove from internal plotting functions

### Fixed
- none-type units improperly handled by split through save/load
- Copy colormaps to avoid editing default mpl colormaps
- Correct units in output of split to be the same as input for axes
- Conversion to/from inches was incorrect

## [3.2.7]

### Added
- matplotlib framework classifier

### Fixed
- Fix error when giving an explicit output array to `symmetric_sqrt`
- Remove deprecated pytest runner from setup.py

## [3.2.6]

### Added
- The turbo colormap has been added (temporarily, as it is staged to be added in matplotlib itself)

### Changed
- Silence matplotlib warnings (including in future matplotlib 3.2 release)
- wt.artists.savefig now accepts kwargs passed onto the underlying implementation
- Quick[1|2]D figures now have a white background
- numexpr is now used to evaluate unit conversions rather than eval

## [3.2.5]

### Fixed
- handling of resultant kwarg in data.Data.moment

## [3.2.4]

### Fixed
- allow contour plotting with colormaps

## [3.2.3]

### Added
- New file type supported: Horiba LabRAM Aramis binary files
- New mode for reading PyCMDS data: collapse=False will read without collapsing data, useful for reading large files in a memory safe way
- interact2D supports keyboard navigation

### Changed
- Colormaps dict will now default to look for Matplotlib colormaps

### Fixed
- Better handling of tolerances in `from_PyCMDS`

## [3.2.2]

### Added
- Additional fluence calculation in kit
- New wt.close method to close all open WT objects in a particular python session

### Changed
- Citation now points to our JOSS Publication
- Trim now works with no outliers, and allows you to ignore the point itself when averaging
- Now uses expressions rather than natural name for axes stored in the file
- Data.moment now allows you to pass resultant=shape such that it can work with multidimensional axes
- `Data.print_tree` will now print scalar values, rather than the shape

### Fixed
- Fixed unit conversion
- Fixed tolerances used for delays in `from_PyCMDS`
- Correction factor applied in `from_PyCMDS` for delay stages collected with errant constant
- bug where `wt-tree` modified accessed file

## [3.2.1]

### Added
- JOSS review
- Quick2D now respects adding contours
- Collections and Data objects can now be created using multi-part posix paths
- Collections and Data objects can now be used as context managers, which close upon exiting
- wt.units.convert added as an alias of wt.units.converter

### Fixed
- Fixed a bug regarding using strings as parameters to prune

## [3.2.0]

### Added
- new method: moment - Take the Nth moment of a channel along an axis
- new method: trim - Use statistical test in channels to remove outlier points
- Support for remote files in `from_` functions

### Changed
- dropped support for Python 3.5, now 3.6+ only
- Deprecated "integrate" method of collapse, as moment handles this
- Use `natural_name` setter to rename groups within the underlying hdf5 file

### Fixed
- bugs in chunkwise operations

## [3.1.6]

### Added
- `Data.prune` method to remove unused variables and channels easily

### Fixed
- `from_Cary` will now read data files correctly where multiple objects have the same name
- improvements to `set_fig_labels`

## [3.1.5]

### Added
- temperature units
- gradient method of data objects

### Fixed
- make interact2D more robust
- ensure that data objects with constants open properly
- fix units behavior when splitting and reading PyCMDS data files

## [3.1.4]

### Added
- Plotting functions like plot, pcolor, and contour now will plot channels e.g. 2D channels within a 3D data object
- Quick plotting methods now have lines and labels for constants
- π release

## [3.1.3]

### Added
- much improved joining of data sets
- constants
- new window functions in smooth1D

## [3.1.2]

### Added
- implementation of split

### Fixed
- improved import of PyCMDS data
- improved import of spcm data
- improved handling of complex dtype within collapse

## [3.1.1]

### Added
- `wt-tree`
- `from_Solis`
- `Downscale`
- methods in kit to calculate fluence
- interactive 2D plots
- `pcolormesh`

### Fixed
- additional bug fixes and conde maintainability improvements

## [3.1.0]

### Added
- `__citation__`
- support for `pathlib` objects

### Changed
- removed fit module
- removed unused artist classes

### Fixed
- recovered `Data.collapse`

## [3.0.4]

### Added
- manifest

## [3.0.3]

### Changed
- migrate requirements

## [3.0.2]

### Fixed
- PyCMDS
- Cary

## [3.0.1]

### Added
- allow for constant values in axis expressions

### Changed
- `from_PyCMDS` will now read incomplete data acquisitions

### Fixed
- windows bug prevented deletion of files
- bug fixes to match actual behavior

## [3.0.0]

## [2.13.9]

## [2.13.8]

## [2.13.7]

## [2.13.6]

### Fixed
- reqirements.txt distribution error

## [2.13.5]

### Added
- initial release

[Unreleased]: https://github.com/wright-group/WrightTools/-/compare/3.4.3...master
[3.4.3]: https://github.com/wright-group/WrightTools/compare/3.4.2...3.4.3
[3.4.2]: https://github.com/wright-group/WrightTools/compare/3.4.1...3.4.2
[3.4.1]: https://github.com/wright-group/WrightTools/compare/3.4.0...3.4.1
[3.4.0]: https://github.com/wright-group/WrightTools/compare/3.3.3...3.4.0
[3.3.3]: https://github.com/wright-group/WrightTools/compare/3.3.2...3.3.3
[3.3.2]: https://github.com/wright-group/WrightTools/compare/3.3.1...3.3.2
[3.3.1]: https://github.com/wright-group/WrightTools/compare/3.3.0...3.3.1
[3.3.0]: https://github.com/wright-group/WrightTools/compare/3.2.7...3.3.0
[3.2.7]: https://github.com/wright-group/WrightTools/compare/3.2.6...3.2.7
[3.2.6]: https://github.com/wright-group/WrightTools/compare/3.2.5...3.2.6
[3.2.5]: https://github.com/wright-group/WrightTools/compare/3.2.4...3.2.5
[3.2.4]: https://github.com/wright-group/WrightTools/compare/3.2.3...3.2.4
[3.2.3]: https://github.com/wright-group/WrightTools/compare/3.2.2...3.2.3
[3.2.2]: https://github.com/wright-group/WrightTools/compare/3.2.1...3.2.2
[3.2.1]: https://github.com/wright-group/WrightTools/compare/3.2.0...3.2.1
[3.2.0]: https://github.com/wright-group/WrightTools/compare/3.1.6...3.2.0
[3.1.6]: https://github.com/wright-group/WrightTools/compare/3.1.5...3.1.6
[3.1.5]: https://github.com/wright-group/WrightTools/compare/3.1.4...3.1.5
[3.1.4]: https://github.com/wright-group/WrightTools/compare/3.1.3...3.1.4
[3.1.3]: https://github.com/wright-group/WrightTools/compare/3.1.2...3.1.3
[3.1.2]: https://github.com/wright-group/WrightTools/compare/3.1.1...3.1.2
[3.1.1]: https://github.com/wright-group/WrightTools/compare/3.1.0...3.1.1
[3.1.0]: https://github.com/wright-group/WrightTools/compare/3.0.4...3.1.0
[3.0.4]: https://github.com/wright-group/WrightTools/compare/3.0.3...3.0.4
[3.0.3]: https://github.com/wright-group/WrightTools/compare/3.0.2...3.0.3
[3.0.2]: https://github.com/wright-group/WrightTools/compare/3.0.1...3.0.2
[3.0.1]: https://github.com/wright-group/WrightTools/compare/3.0.0...3.0.1
[3.0.0]: https://github.com/wright-group/WrightTools/compare/2.13.9...3.0.0
[2.13.9]: https://github.com/wright-group/WrightTools/compare/2.13.8...2.13.9
[2.13.8]: https://github.com/wright-group/WrightTools/compare/2.13.7...2.13.8
[2.13.7]: https://github.com/wright-group/WrightTools/compare/2.13.6...2.13.7
[2.13.6]: https://github.com/wright-group/WrightTools/compare/2.13.5...2.13.6
[2.13.5]: https://github.com/wright-group/WrightTools/releases/tag/2.13.5

