"""Channel class and associated."""


# --- import --------------------------------------------------------------------------------------


import numpy as np

import h5py

import warnings
import numbers

from .. import kit as wt_kit
from .._dataset import Dataset
from .. import exceptions as wt_exceptions

__all__ = ["Channel"]

# --- class ---------------------------------------------------------------------------------------


class Channel(Dataset):
    """Channel."""

    class_name = "Channel"

    def __init__(
        self,
        parent,
        id,
        *,
        units=None,
        null=None,
        signed=None,
        label=None,
        label_seed=None,
        **kwargs
    ):
        """Construct a channel object.

        Parameters
        ----------
        values : array-like
            Values.
        name : string
            Channel name.
        units : string (optional)
            Channel units. Default is None.
        null : number (optional)
            Channel null. Default is None (0).
        signed : booelan (optional)
            Channel signed flag. Default is None (guess).
        label : string.
            Label. Default is None.
        label_seed : list of strings
            Label seed. Default is None.
        **kwargs
            Additional keyword arguments are added to the attrs dictionary
            and to the natural namespace of the object (if possible).
        """
        self._parent = parent
        super().__init__(id)
        self.label = label
        self.label_seed = label_seed
        self.units = units
        self.dimensionality = len(self.shape)
        # attrs
        if self._parent.file.mode is not None and self._parent.file.mode != "r":
            try:
                self.attrs.update(kwargs)
                self.attrs["name"] = h5py.h5i.get_name(self.id).decode().split("/")[-1]
                self.attrs["class"] = "Channel"
                if signed is not None:
                    self.attrs["signed"] = signed
                if null is not None:
                    self.attrs["null"] = null
            except (RuntimeError, KeyError):
                # e.g. readonly file
                pass
        for key, value in self.attrs.items():
            identifier = wt_kit.string2identifier(key)
            if not hasattr(self, identifier):
                setattr(self, identifier, value)

    @property
    def major_extent(self) -> complex:
        """Maximum deviation from null."""
        return max((self.max() - self.null, self.null - self.min()))

    @property
    def minor_extent(self) -> complex:
        """Minimum deviation from null."""
        return min((self.max() - self.null, self.null - self.min()))

    @property
    def null(self) -> complex:
        if "null" not in self.attrs.keys():
            self.attrs["null"] = 0
        return self.attrs["null"]

    @null.setter
    def null(self, value):
        self.attrs["null"] = value

    @property
    def signed(self) -> bool:
        if "signed" not in self.attrs.keys():
            self.attrs["signed"] = False
        return self.attrs["signed"]

    @signed.setter
    def signed(self, value):
        self.attrs["signed"] = value

    def mag(self) -> complex:
        """Channel magnitude (maximum deviation from null)."""
        return self.major_extent

    def normalize(self, mag=1.0):
        """Normalize a Channel, set `null` to 0 and the mag to given value.

        Parameters
        ----------
        mag : float (optional)
            New value of mag. Default is 1.
        """

        def f(dataset, s, null, mag):
            dataset[s] -= null
            dataset[s] /= mag

        if self.signed:
            mag = self.mag() / mag
        else:
            mag = self.max() / mag
        self.chunkwise(f, null=self.null, mag=mag)
        self._null = 0

    def trim(self, neighborhood, method="ztest", factor=3, replace="nan", verbose=True):
        """Remove outliers from the dataset.

        Identifies outliers by comparing each point to its
        neighbors using a statistical test.

        Parameters
        ----------
        neighborhood : list of integers
            Size of the neighborhood in each dimension. Length of the list must
            be equal to the dimensionality of the channel.
        method : {'ztest'} (optional)
            Statistical test used to detect outliers. Default is ztest.

            ztest
                Compare point deviation from neighborhood mean to neighborhood
                standard deviation.

        factor : number (optional)
            Tolerance factor.  Default is 3.
        replace : {'nan', 'mean', 'exclusive_mean', number} (optional)
            Behavior of outlier replacement. Default is nan.

            nan
                Outliers are replaced by numpy nans.

            mean
                Outliers are replaced by the mean of its neighborhood, including itself.

            exclusive_mean
                Outilers are replaced by the mean of its neighborhood, not including itself.

            number
                Array becomes given number.

        Returns
        -------
        list of tuples
            Indicies of trimmed outliers.

        See Also
        --------
        clip
            Remove pixels outside of a certain range.
        """
        warnings.warn("trim", category=wt_exceptions.EntireDatasetInMemoryWarning)
        outliers = []
        means = []
        ex_means = []
        # find outliers
        for idx in np.ndindex(self.shape):
            slices = []
            for i, di, size in zip(idx, neighborhood, self.shape):
                start = max(0, i - di)
                stop = min(size, i + di + 1)
                slices.append(slice(start, stop, 1))
            neighbors = self[slices]
            mean = np.nanmean(neighbors)
            sum_ = np.nansum(neighbors)
            limit = np.nanstd(neighbors) * factor
            if np.abs(self[idx] - mean) > limit:
                outliers.append(idx)
                means.append(mean)
                # Note, "- 1" is to exclude the point itself, which is not nan, in order
                # to enter this if block, as `np.abs(nan - mean)` is nan, which would
                # evaluate to False
                ex_means.append((sum_ - self[idx]) / (np.sum(~np.isnan(neighbors)) - 1))

        # replace outliers
        i = tuple(zip(*outliers))

        if len(i) == 0:
            if verbose:
                print("No outliers found")
            return []

        replace = {"nan": np.nan, "mean": means, "exclusive_mean": ex_means}.get(replace, replace)

        # This may someday be available in h5py directly, but seems that day is not yet.
        # This is annoying because it is the only reason we hold the whole set in memory.
        # KFS 2019-03-21
        arr = self[:]
        arr[i] = replace
        self[:] = arr

        # finish
        if verbose:
            print("%i outliers removed" % len(outliers))
        return outliers
