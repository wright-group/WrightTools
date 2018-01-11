"""Channel class and associated."""


# --- import --------------------------------------------------------------------------------------


import h5py

from .. import kit as wt_kit
from .._dataset import Dataset


# --- class ---------------------------------------------------------------------------------------


class Channel(Dataset):
    """Channel."""

    class_name = 'Channel'

    def __init__(self, parent, id, *, units=None, null=None, signed=None, label=None,
                 label_seed=None, **kwargs):
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
        self.attrs.update(kwargs)
        self.attrs['name'] = h5py.h5i.get_name(self.id).decode().split('/')[-1]
        self.attrs['class'] = 'Channel'
        if signed is not None:
            self.attrs['signed'] = signed
        if null is not None:
            self.attrs['null'] = null
        for key, value in self.attrs.items():
            identifier = wt_kit.string2identifier(key)
            if not hasattr(self, identifier):
                setattr(self, identifier, value)

    @property
    def minor_extent(self):
        """Minimum deviation from null."""
        return min((self.max() - self.null, self.null - self.min()))

    @property
    def natural_name(self):
        """Natural name of the dataset. May be different from name."""
        try:
            assert self._natural_name is not None
        except (AssertionError, AttributeError):
            self._natural_name = self.attrs['name']
        finally:
            return self._natural_name

    @natural_name.setter
    def natural_name(self, value):
        index = wt_kit.get_index(self.parent.channel_names, self.natural_name)
        new = list(self.parent.channel_names)
        new[index] = value
        self.parent.channel_names = new
        self.attrs['name'] = value
        self._natural_name = None

    @property
    def null(self):
        if 'null' not in self.attrs.keys():
            self.attrs['null'] = 0
        return self.attrs['null']

    @property
    def info(self):
        """Return Channel info dictionary."""
        info = collections.OrderedDict()
        info['name'] = self.name
        info['min'] = self.min()
        info['max'] = self.max()
        info['null'] = self.null
        info['signed'] = self.signed
        return info

    @property
    def major_extent(self):
        """Maximum deviation from null."""
        return max((self.max() - self.null, self.null - self.min()))

    @property
    def signed(self):
        if 'signed' not in self.attrs.keys():
            self.attrs['signed'] = False
        return self.attrs['signed']

    def clip(self, min=None, max=None, replace='nan'):
        """Clip (limit) the values in a channel.

        Parameters
        ----------
        min : number (optional)
            New channel minimum. Default is None.
        max : number (optional)
            New channel maximum. Default is None.
        replace : {'val', 'nan'} (optional)
           Replace behavior. Default is nan.
        """
        # decide what min and max will actually be
        if max is None:
            max = self.max()
        if min is None:
            min = self.min()
        # replace values
        if replace == 'val':
            self[:].clip(min, max, out=self[:])
        elif replace == 'nan':
            self[self[:] < min] = np.nan
            self[self[:] > max] = np.nan
        else:
            print('replace not recognized in channel.clip')

    def invert(self):
        """Invert channel values."""
        self[:] *= -1

    def mag(self):
        """Channel magnitude (maximum deviation from null)."""
        return self.major_extent

    def normalize(self, axis=None):
        """Normalize a Channel, set `null` to 0 and the max to 1."""
        # process axis argument
        if axis is not None:
            if hasattr(axis, '__contains__'):  # list, tuple or similar
                axis = tuple((int(i) for i in axis))
            else:  # presumably a simple number
                axis = int(axis)
        # subtract off null
        self[:] -= self.null
        self._null = 0.
        # create dummy array
        dummy = self[:].copy()
        dummy[np.isnan(dummy)] = 0  # nans are propagated in np.amax
        if self.signed:
            dummy = np.absolute(dummy)
        # divide through by max
        self[:] /= np.amax(dummy, axis=axis, keepdims=True)
        # finish

    def trim(self, neighborhood, method='ztest', factor=3, replace='nan',
             verbose=True):
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
        replace : {'nan', 'mean', 'mask', number} (optional)
            Behavior of outlier replacement. Default is nan.

            nan
                Outliers are replaced by numpy nans.

            mean
                Outliers are replaced by the mean of its neighborhood.

            mask
                Array is masked at outliers.

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
        outliers = []
        means = []
        # find outliers
        for idx in np.ndindex(self.shape):
            slices = []
            for i, di, size in zip(idx, neighborhood, self.shape):
                start = max(0, i - di)
                stop = min(size, i + di + 1)
                slices.append(slice(start, stop, 1))
            neighbors = self[slices]
            mean = np.nanmean(neighbors)
            limit = np.nanstd(neighbors) * factor
            if np.abs(self[idx] - mean) > limit:
                outliers.append(idx)
                means.append(mean)
        # replace outliers
        i = tuple(zip(*outliers))
        if replace == 'nan':
            self[i] = np.nan
        elif replace == 'mean':
            self[i] = means
        elif replace == 'mask':
            self[:] = np.ma.array(self[:])
            self[i] = np.ma.masked
        elif type(replace) in [int, float]:
            self[i] = replace
        else:
            raise KeyError('replace must be one of {nan, mean, mask} or some number')
        # finish
        if verbose:
            print('%i outliers removed' % len(outliers))
        return outliers
