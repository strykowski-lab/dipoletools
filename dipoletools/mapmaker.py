"""MapMaker class: load catalogues, apply cuts, crossmatch, and generate HEALPix maps."""

import os
import numpy as np
import pandas as pd
import healpy as hp
from astropy.coordinates import SkyCoord, search_around_sky
from astropy import units as u
from pathlib import Path

from ._utils import convert_lonlat, lonlat_names, d2r, r2d, _CallableArray
from ._defaults import detect_catalogue, SHORTHAND_CATALOGUES, DATASTORE_PATH


class MapMaker:
    """Load source catalogues, apply cuts, crossmatch, and bin into HEALPix maps.

    Parameters
    ----------
    catalogue : str or tuple
        Filepath to the catalogue, (filepath, labels_dict), or a shorthand
        name (e.g. 'nvss', 'racs-low1'). See SHORTHAND_CATALOGUES in
        _defaults.py for all available shorthands.
    labels : dict, optional
        Column label mapping if not bundled with the filepath.
    """

    def __init__(self, catalogue=None, labels=None):
        self._nside = 64

        # Multi-catalogue storage: name -> data
        self._catalogues = {}       # name -> current DataFrame
        self._backups = {}          # name -> backup DataFrame
        self._labels = {}           # name -> labels dict (copy)
        self._catalogue_order = []  # names in insertion order

        self._map = None
        self._coord_system = 'C'
        self._map_coords = None  # target coord system for map output
        self._shorthand_config = None
        self._shorthand_name = None   # shorthand catalogue name if used
        self._default_cuts = None  # deferred cuts from shorthand config
        self._user_cut = False     # True if user called cut() manually
        self._map_loader = None    # callable for pre-computed map shorthands

        if catalogue is not None:
            self.catalogue(catalogue, labels=labels)

    # ------------------------------------------------------------------
    # nside property
    # ------------------------------------------------------------------
    @property
    def nside(self):
        return self._nside

    @nside.setter
    def nside(self, value):
        self._nside = int(value)

    # ------------------------------------------------------------------
    # catalogue
    # ------------------------------------------------------------------
    def catalogue(self, filepath=None, labels=None, name=None):
        """Load a source catalogue or return an existing one.

        Parameters
        ----------
        filepath : str or tuple, optional
            Path to catalogue file, (filepath, labels_dict), or a shorthand
            name (e.g. 'nvss', 'racs-low1').
            If None, returns an existing catalogue.
        labels : dict, optional
            Maps standard names to column names.
            First catalogue requires ('ra','dec') or ('l','b') and 'flux'.
            Additional catalogues only require coordinate columns.
        name : str, optional
            Name for this catalogue. First catalogue defaults to 'original'.
            When filepath is None, retrieves the catalogue with this name
            (defaults to 'original').
        """
        # --- Retrieve mode ---
        if filepath is None:
            retrieve_name = name if name is not None else (
                self._catalogue_order[0] if self._catalogue_order else None
            )
            if retrieve_name is None:
                raise ValueError("No catalogue has been loaded yet.")
            if retrieve_name not in self._catalogues:
                raise ValueError(f"No catalogue named '{retrieve_name}'.")
            return self._catalogues[retrieve_name].copy()

        # --- Load mode ---
        if isinstance(filepath, tuple):
            filepath, labels = filepath

        # Accept DataFrame directly
        if isinstance(filepath, pd.DataFrame):
            df = filepath.copy()
        elif isinstance(filepath, str) and filepath in SHORTHAND_CATALOGUES:
            # Shorthand catalogue name
            self._shorthand_name = filepath
            config = SHORTHAND_CATALOGUES[filepath]
            if config.get('type') == 'map':
                # Pre-computed map shorthand (e.g. Planck) — no catalogue
                self._map_loader = config['loader']
                return None
            filepath = os.path.join(DATASTORE_PATH, config['file'])
            if labels is None:
                labels = config['labels']
            df = self._load_file(filepath)
            self._shorthand_config = config
        else:
            filepath = str(filepath)
            if labels is None:
                config = detect_catalogue(filepath)
                if config is not None:
                    labels = config['labels']
            df = self._load_file(filepath)

        if labels is None:
            raise ValueError(
                "Could not auto-detect catalogue format. "
                "Please provide a labels dict."
            )

        # Validate coordinate labels
        has_eq = 'ra' in labels and 'dec' in labels
        has_gal = 'l' in labels and 'b' in labels
        if not has_eq and not has_gal:
            raise ValueError("Labels must include ('ra','dec') or ('l','b').")

        # First catalogue requires 'flux'
        is_first = len(self._catalogue_order) == 0
        if is_first and 'flux' not in labels:
            raise ValueError("First catalogue labels must include 'flux'.")

        # Determine name
        if name is None:
            if is_first:
                name = 'original'
            else:
                raise ValueError(
                    "Name is required for additional catalogues. "
                    "Pass name='...' to catalogue()."
                )

        if name in self._catalogues and name != 'original':
            raise ValueError(f"Catalogue '{name}' already exists.")

        # Store
        self._labels[name] = dict(labels)  # always copy
        self._catalogues[name] = df
        self._backups[name] = df.copy()
        if name not in self._catalogue_order:
            self._catalogue_order.append(name)

        # Set coord system from first catalogue
        if is_first:
            if has_gal and not has_eq:
                self._coord_system = 'G'
            else:
                self._coord_system = 'C'

        # Store shorthand defaults for deferred application in map()
        if self._shorthand_config is not None:
            config = self._shorthand_config
            if 'cuts' in config:
                self._default_cuts = config['cuts']
            if 'map_coords' in config:
                self._map_coords = config['map_coords']
            self._shorthand_config = None

        return self._catalogues[name].copy()

    # ------------------------------------------------------------------
    # cut
    # ------------------------------------------------------------------
    def cut(self, label, min=None, max=None, catalogue=None, strict=False):
        """Apply a cut to a catalogue. Cuts are cumulative.

        Parameters
        ----------
        label : str or list of str
            Column name(s) to cut on. Resolved via labels dict.
        min : float or list, optional
            Minimum value(s). None means no lower bound.
        max : float or list, optional
            Maximum value(s). None means no upper bound.
        catalogue : str, optional
            Name of catalogue to cut. Defaults to first catalogue.
        strict : bool, optional
            If True, use strict inequalities (> for min, < for max).
            Default False uses >= for min, <= for max.
        """
        cat_name = catalogue if catalogue is not None else self._catalogue_order[0]
        if cat_name not in self._catalogues:
            raise ValueError(f"No catalogue named '{cat_name}'.")

        self._user_cut = True

        df = self._catalogues[cat_name]
        cat_labels = self._labels[cat_name]

        # Normalize to lists
        if isinstance(label, str):
            label = [label]
            min = [min]
            max = [max]
        else:
            if min is None:
                min = [None] * len(label)
            if max is None:
                max = [None] * len(label)

        for lbl, lo, hi in zip(label, min, max):
            col = cat_labels.get(lbl, lbl)
            df[col] = df[col].astype(np.float64)
            if lo is not None:
                df = df[df[col] > lo] if strict else df[df[col] >= lo]
            if hi is not None:
                df = df[df[col] < hi] if strict else df[df[col] <= hi]

        self._catalogues[cat_name] = df

    # ------------------------------------------------------------------
    # restore
    # ------------------------------------------------------------------
    def restore(self, catalogue=None):
        """Restore a catalogue to its original state (undo all cuts).

        Parameters
        ----------
        catalogue : str, optional
            Name of catalogue to restore. Defaults to first catalogue.
        """
        cat_name = catalogue if catalogue is not None else self._catalogue_order[0]
        if cat_name not in self._backups:
            raise ValueError(f"No catalogue named '{cat_name}'.")
        self._catalogues[cat_name] = self._backups[cat_name].copy()

    # ------------------------------------------------------------------
    # crossmatch
    # ------------------------------------------------------------------
    def crossmatch(self, *args, radius=5):
        """Crossmatch two catalogues and remove matched sources.

        Replicates the full crossmatch algorithm with duplicate resolution:
        for each source in catalogue A, finds all sources in catalogue B
        within the radius. Assigns each A source its closest B match,
        then iteratively resolves conflicts where multiple A sources
        claim the same B source (closest wins, losers get reassigned to
        their next-closest backup match).

        The first argument is the catalogue to be filtered (B in the old
        notation). The second argument is the reference catalogue (A).
        Matched sources are removed from the first catalogue.

        Parameters
        ----------
        *args : str
            Zero or two catalogue name strings.
            crossmatch('survey', 'reference') removes matched sources
            from 'survey' using 'reference' as the matching catalogue.
            crossmatch() defaults to crossmatch('original', first_additional).
        radius : float
            Crossmatch radius in arcminutes. Default 5.
        """
        if len(args) == 0:
            if len(self._catalogue_order) < 2:
                raise ValueError(
                    "Need at least two catalogues for default crossmatch()."
                )
            b_name = self._catalogue_order[0]
            a_name = self._catalogue_order[1]
        elif len(args) == 2:
            b_name, a_name = args
        else:
            raise ValueError("Pass zero or two catalogue name strings.")

        if b_name not in self._catalogues:
            raise ValueError(f"No catalogue named '{b_name}'.")
        if a_name not in self._catalogues:
            raise ValueError(f"No catalogue named '{a_name}'.")

        A_sources = self._catalogues[a_name]
        B_sources = self._catalogues[b_name]
        A_labels = self._labels[a_name]
        B_labels = self._labels[b_name]

        # Get coordinate columns
        a_ra_col = A_labels.get('ra', A_labels.get('l', 'ra'))
        a_dec_col = A_labels.get('dec', A_labels.get('b', 'dec'))
        b_ra_col = B_labels.get('ra', B_labels.get('l', 'ra'))
        b_dec_col = B_labels.get('dec', B_labels.get('b', 'dec'))

        # Get B id column (use index if no id label)
        b_id_col = B_labels.get('id', None)
        if b_id_col is None or b_id_col not in B_sources.columns:
            B_sources = B_sources.copy()
            B_sources['_xmatch_idx'] = np.arange(len(B_sources))
            b_id_col = '_xmatch_idx'

        A_coords = SkyCoord(
            ra=A_sources[a_ra_col].astype(float).values * u.deg,
            dec=A_sources[a_dec_col].astype(float).values * u.deg,
        )
        B_coords = SkyCoord(
            ra=B_sources[b_ra_col].astype(float).values * u.deg,
            dec=B_sources[b_dec_col].astype(float).values * u.deg,
        )

        # Find all pairs within radius
        idxA, idxB, sep, _ = search_around_sky(
            A_coords, B_coords, radius * u.arcmin
        )

        if len(idxA) == 0:
            return  # No matches

        sep_arcmin = sep.arcmin

        # For each A source with matches, build sorted list of (b_id, separation)
        from collections import defaultdict
        a_matches = defaultdict(list)
        for i in range(len(idxA)):
            a_matches[int(idxA[i])].append(
                (B_sources.iloc[int(idxB[i])][b_id_col], sep_arcmin[i])
            )

        # Sort each A source's matches by separation (closest first)
        for a_idx in a_matches:
            a_matches[a_idx].sort(key=lambda x: x[1])

        # Build initial match table: each A source gets its closest B match
        # with remaining matches as backups
        records = []
        for a_idx, b_list in a_matches.items():
            closest_b_id, closest_sep = b_list[0]
            if len(b_list) > 1:
                backup_ids = np.array([x[0] for x in b_list[1:]])
                backup_dists = np.array([x[1] for x in b_list[1:]])
            else:
                backup_ids = None
                backup_dists = None
            records.append({
                'a_idx': a_idx,
                'b_id': closest_b_id,
                'separation': closest_sep,
                'duplicates': backup_ids,
                'distances': backup_dists,
            })

        match_df = pd.DataFrame(records)

        # Iteratively resolve duplicate B matches (fixmatches algorithm)
        match_df, max_iters = self._fixmatches(match_df, first=True)
        for _ in range(max_iters):
            match_df = self._fixmatches(match_df)

        # Verify no duplicates remain
        if not match_df[match_df.duplicated(subset=['b_id'], keep=False)].empty:
            for attempt in range(10):
                match_df = self._fixmatches(match_df)
                if match_df[
                    match_df.duplicated(subset=['b_id'], keep=False)
                ].empty:
                    break
            else:
                raise ValueError("Could not resolve all duplicate matches.")

        # Remove matched B sources from the catalogue
        matched_b_ids = match_df['b_id'].values
        original_b = self._catalogues[b_name]

        if b_id_col == '_xmatch_idx':
            # We used synthetic indices; map back to original df indices
            keep_mask = ~original_b.index.isin(
                B_sources.loc[
                    B_sources['_xmatch_idx'].isin(matched_b_ids)
                ].index
            )
        else:
            keep_mask = ~original_b[b_id_col].isin(matched_b_ids)

        filtered = original_b[keep_mask].reset_index(drop=True)
        self._catalogues[b_name] = filtered

    @staticmethod
    def _fixmatches(match_df, first=False):
        """Resolve duplicate B matches: closest A wins, losers reassigned.

        Replicates the fixmatches algorithm from crossmatching_context.txt.
        """
        duplicates = match_df[
            match_df.duplicated(subset=['b_id'], keep=False)
        ]
        non_duplicates = match_df[
            ~match_df.duplicated(subset=['b_id'], keep=False)
        ]

        if duplicates.empty:
            if first:
                return match_df, 0
            return match_df

        # For each duplicated b_id, keep the A source with smallest separation
        unique_b_ids = duplicates['b_id'].unique()
        closest_indices = []
        for bid in unique_b_ids:
            subset = duplicates[duplicates['b_id'] == bid]
            closest_indices.append(subset['separation'].idxmin())

        closest_matches = duplicates.loc[closest_indices]
        farthest_matches = duplicates[~duplicates.index.isin(closest_indices)]

        # For losers with backup matches, reassign to their next backup
        def _has_backups(x):
            if x is None:
                return False
            if isinstance(x, np.ndarray):
                return x.size > 0
            return True

        farthest_with_backups = farthest_matches[
            farthest_matches['duplicates'].apply(_has_backups)
        ].copy()

        max_dupes = 0
        if not farthest_with_backups.empty:
            for idx in farthest_with_backups.index:
                backup_ids = farthest_with_backups.at[idx, 'duplicates']
                backup_dists = farthest_with_backups.at[idx, 'distances']

                backup_ids = np.atleast_1d(backup_ids)
                backup_dists = np.atleast_1d(backup_dists)

                farthest_with_backups.at[idx, 'b_id'] = backup_ids[0]
                farthest_with_backups.at[idx, 'separation'] = backup_dists[0]
                if len(backup_ids) > 1:
                    farthest_with_backups.at[idx, 'duplicates'] = (
                        backup_ids[1:]
                    )
                    farthest_with_backups.at[idx, 'distances'] = (
                        backup_dists[1:]
                    )
                    max_dupes = max(max_dupes, len(backup_ids))
                else:
                    farthest_with_backups.at[idx, 'duplicates'] = None
                    farthest_with_backups.at[idx, 'distances'] = None
                    max_dupes = max(max_dupes, 1)

        combined = pd.concat(
            [non_duplicates, closest_matches, farthest_with_backups]
        )
        combined.reset_index(drop=True, inplace=True)

        if first:
            return combined, max_dupes
        return combined

    # ------------------------------------------------------------------
    # map (property + callable)
    # ------------------------------------------------------------------
    @property
    def map(self):
        """Return the HEALPix count map as a callable array.

        Can be used as ``mm.map`` (computes with defaults) or ``mm.map()``
        (same), or ``mm.map(nside=128)`` (with arguments).
        """
        result = self._compute_map()
        return _CallableArray(result, self._compute_map)

    def _compute_map(self, nside=None, catalogue=None):
        """Generate a HEALPix count map from a catalogue.

        Parameters
        ----------
        nside : int, optional
            HEALPix nside. Uses self.nside if not provided.
        catalogue : str, optional
            Name of catalogue to use. Defaults to first catalogue.

        Returns
        -------
        numpy.ndarray
            Full HEALPix map of source counts.
        """
        if nside is not None:
            self._nside = nside
        nside = self._nside

        # Pre-computed map shorthand (e.g. Planck)
        if self._map_loader is not None:
            self._map = self._map_loader(nside=nside)
            return self._map

        cat_name = catalogue if catalogue is not None else self._catalogue_order[0]
        if cat_name not in self._catalogues:
            raise ValueError(f"No catalogue named '{cat_name}'.")

        # Apply default cuts from shorthand config if user hasn't cut manually
        if self._default_cuts and not self._user_cut:
            for c in self._default_cuts:
                self.cut(c['col'], min=c.get('min'), max=c.get('max'),
                         catalogue=cat_name, strict=c.get('strict', False))
            self._user_cut = False  # reset — these were auto-applied, not user

        cat = self._catalogues[cat_name]
        labels = self._labels[cat_name]

        # Get coordinate columns
        lon_name, lat_name = lonlat_names(self._coord_system)
        lon_col = labels.get(lon_name, lon_name)
        lat_col = labels.get(lat_name, lat_name)

        # Fallback: try to find coordinate columns
        if lon_col not in cat.columns:
            if 'ra' in labels and labels['ra'] in cat.columns:
                lon_col = labels['ra']
                lat_col = labels['dec']
            elif 'l' in labels and labels['l'] in cat.columns:
                lon_col = labels['l']
                lat_col = labels['b']
            else:
                for try_lon, try_lat in [('ra', 'dec'), ('l', 'b')]:
                    if try_lon in cat.columns:
                        lon_col, lat_col = try_lon, try_lat
                        break

        if isinstance(cat, pd.DataFrame):
            lon = cat[lon_col].astype(float).values
            lat = cat[lat_col].astype(float).values
        else:
            lon = np.asarray(cat[lon_col], dtype=float)
            lat = np.asarray(cat[lat_col], dtype=float)

        # Convert coordinates if map_coords differs from catalogue coords
        map_sys = self._map_coords or self._coord_system
        if map_sys != self._coord_system:
            new_lon, new_lat = convert_lonlat(lon, lat,
                                              self._coord_system, map_sys)
            lon, lat = new_lon, new_lat

        # Bin into HEALPix pixels
        npix = hp.nside2npix(nside)
        pix = hp.ang2pix(nside, lon, lat, lonlat=True)
        counts = np.bincount(pix, minlength=npix).astype(float)

        self._map = counts
        return counts

    # ------------------------------------------------------------------
    # show
    # ------------------------------------------------------------------
    def show(self, **kwargs):
        """Display the count map with healpy.projview.

        Default colormap: plasma. Default title: none.
        Any keyword arguments are forwarded to hp.projview.
        """
        import matplotlib.pyplot as plt
        m = self._map if self._map is not None else self._compute_map()
        kwargs.setdefault('cmap', 'plasma')
        kwargs.setdefault('title', '')
        hp.projview(m, **kwargs)
        plt.show()

    # ------------------------------------------------------------------
    # coords
    # ------------------------------------------------------------------
    def coords(self, *args, target=None):
        """Query or set/convert the coordinate system.

        Parameters
        ----------
        No args: returns current coordinate system string.
        One arg (str): labels the current coordinate system.
        Two args or list of two: converts from first system to second.
        target : str, optional
            Catalogue name to convert only that catalogue. If None,
            converts all catalogues.
        """
        if len(args) == 0:
            return self._coord_system

        if len(args) == 1:
            arg = args[0]
            if isinstance(arg, (list, tuple)):
                if len(arg) == 1:
                    self._coord_system = arg[0]
                    return
                elif len(arg) == 2:
                    from_sys, to_sys = arg
                else:
                    raise ValueError("Pass one or two coordinate system names.")
            else:
                self._coord_system = arg
                return
        elif len(args) == 2:
            from_sys, to_sys = args
        else:
            raise ValueError("Pass one or two coordinate system names.")

        # Perform conversion
        if target is not None:
            # Convert only the named catalogue
            if target in self._catalogues:
                self._convert_df(
                    self._catalogues[target],
                    self._labels[target], from_sys, to_sys
                )
                if target in self._backups:
                    backup_labels = dict(self._labels[target])
                    # Need fresh labels for backup since _convert_df mutates
                    # Actually, labels are already updated by the first call,
                    # so backup conversion uses updated label keys.
                    # We need to convert backup from original system.
                    # Re-derive backup labels from the pre-conversion state.
                    pass
                    # Backup conversion: use the same column names as current
                    self._convert_df(
                        self._backups[target],
                        dict(self._labels[target]), from_sys, to_sys
                    )
        else:
            # Convert all catalogues
            for cat_name in self._catalogue_order:
                # Copy labels so each conversion gets the right state
                labels_copy = dict(self._labels[cat_name])
                self._convert_df(
                    self._catalogues[cat_name], labels_copy,
                    from_sys, to_sys
                )
                # Convert backup with a fresh labels copy
                backup_labels = dict(labels_copy)  # already updated
                self._convert_df(
                    self._backups[cat_name], dict(self._labels[cat_name]),
                    from_sys, to_sys
                )
                # Update stored labels to the converted version
                self._labels[cat_name] = labels_copy

        self._coord_system = to_sys

    def _convert_df(self, df, labels, from_sys, to_sys):
        """Convert coordinates in a DataFrame in-place."""
        from_lon_name, from_lat_name = lonlat_names(from_sys)
        from_lon_col = labels.get(from_lon_name, labels.get('ra', labels.get('l', None)))
        from_lat_col = labels.get(from_lat_name, labels.get('dec', labels.get('b', None)))

        if from_lon_col not in df.columns:
            for try_lon, try_lat in [('ra', 'dec'), ('l', 'b')]:
                if try_lon in df.columns:
                    from_lon_col, from_lat_col = try_lon, try_lat
                    break

        if from_lon_col not in df.columns:
            return

        lon = df[from_lon_col].astype(float).values
        lat = df[from_lat_col].astype(float).values

        new_lon, new_lat = convert_lonlat(lon, lat, from_sys, to_sys)

        to_lon_name, to_lat_name = lonlat_names(to_sys)

        df.rename(columns={from_lon_col: to_lon_name, from_lat_col: to_lat_name},
                  inplace=True)
        df[to_lon_name] = new_lon
        df[to_lat_name] = new_lat

        # Update labels dict
        old_lon_key = None
        old_lat_key = None
        for k, v in labels.items():
            if v == from_lon_col:
                old_lon_key = k
            if v == from_lat_col:
                old_lat_key = k
        if old_lon_key:
            del labels[old_lon_key]
        if old_lat_key:
            del labels[old_lat_key]
        labels[to_lon_name] = to_lon_name
        labels[to_lat_name] = to_lat_name

    # ------------------------------------------------------------------
    # File loading
    # ------------------------------------------------------------------
    @staticmethod
    def _load_file(filepath):
        """Load a catalogue file into a pandas DataFrame."""
        filepath = str(filepath)
        ext = Path(filepath).suffix.lower()

        if ext == '.csv':
            return pd.read_csv(filepath)
        elif ext == '.dat' or ext == '.txt' or ext == '.tsv':
            try:
                df = pd.read_csv(filepath, sep=r'\s+')
                if len(df.columns) <= 1:
                    df = pd.read_csv(filepath)
                return df
            except Exception:
                return pd.read_csv(filepath)
        elif ext == '.fits' or ext == '.fit':
            from astropy.io import fits as astropy_fits
            from astropy.table import Table
            table = Table.read(filepath)
            return table.to_pandas()
        elif ext == '.pkl' or ext == '.pickle':
            import pickle
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            if isinstance(data, pd.DataFrame):
                return data
            else:
                raise ValueError(f"Pickle file does not contain a DataFrame: {filepath}")
        elif ext == '.npy':
            data = np.load(filepath, allow_pickle=True)
            if data.dtype.names:
                return pd.DataFrame(data)
            else:
                raise ValueError(f".npy file does not have named columns: {filepath}")
        else:
            return pd.read_csv(filepath)
