#!/usr/bin/env python
# -*- coding: utf-8 -*-

"Merge meteogram files"


import re
import glob
import functools
import itertools
from collections import OrderedDict, defaultdict, namedtuple

import netCDF4
import numpy as np
import pandas as pd


var_signature = namedtuple('var_signature', 'name dtype dimensions')
time_signature = var_signature('time', 'i4', ('time',))

fill_value = {'_FillValue': 9.96920996839e+36}

time_metadata = OrderedDict([
    ("standard_name", "time"),
    ("long_name", "time"),
    ("units", "seconds since"),
    ("calendar", "proleptic_gregorian"),
    ("axis", "T"),
])

group_mapping = OrderedDict([
    # group_key -> primary_key
    ('station_', 'station_name'),
    ('var_', 'var_name'),
    ('sfcvar_', 'sfcvar_name'),
])


class OrderedDefaultDict(OrderedDict, defaultdict):
    "default dict that keeps the order"
    def __init__(self, factory, *args, **kwargs):
        defaultdict.__init__(self, factory)
        OrderedDict.__init__(self, *args, **kwargs)


def memoize(func):
    cache = func.cache = {}

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        key = (args, frozenset(kwargs.items()))
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]
    return wrapper


def index2slice(indices, *more_indices):

    def inc(i): return i + 1

    def inc_deduct(values, counter=itertools.count()):
        c = next(counter)
        if isinstance(values, int):
            return values - c
        return [(val - c) for val in values]

    if more_indices:
        indices = zip(indices, *more_indices)

    slices = []
    for (_, g) in itertools.groupby(indices, inc_deduct):
        values = list(g)
        end = values.pop()
        if values:
            start = values.pop(0)
            if isinstance(start, tuple):
                slices.append(
                    [slice(s, e) for (s, e) in zip(start, map(inc, end))])
            else:
                slices.append(slice(start, inc(end)))
        else:
            slices.append(end)
    if more_indices:
        slices = zip(*slices)
    return slices


def empty(*items):
    if not items:
        return True
    result = []
    for item in items:
        if isinstance(item, (pd.Series, pd.DataFrame)):
            result.append(False if not item.empty else True)
        elif isinstance(item, (pd.Index, np.ndarray)):
            result.append(False if item.size else True)
        else:
            result.append(False if item else True)
    return all(result)


def merge_dimensions(ncids):
    ncids = sorted(ncids, key=lambda nc: len(nc.dimensions), reverse=True)
    dims = OrderedDefaultDict(list)
    for nc in ncids:
        for d in nc.dimensions.values():
            dims[d.name].append(None if d.isunlimited() else len(d))
    for name, val in dims.items():
        dims[name] = max(val)
    return OrderedDict(dims)


def name_dtype_dimension(ncids, varlist):
    ncids = sorted(ncids, key=lambda nc: len(nc.variables), reverse=True)
    varlist = OrderedDict.fromkeys(varlist)
    for name in varlist:
        for nc in ncids:
            if name in nc.variables:
                obj = nc.variables[name]
                varlist[name] = var_signature(
                    obj.name, obj.dtype, obj.dimensions)
                break
    varlist['time'] = time_signature
    for name, val in varlist.items():
        if val is None:
            varlist.popitem(name)
    return varlist


def _ensure_time_in_variable_list(varlist):
    if 'time' not in varlist:
        time_index = varlist.index('date') - 1
        varlist.insert(time_index, 'time')
    return varlist


def merge_variable_names(ncids):
    ncids = sorted(ncids, key=lambda nc: len(nc.variables), reverse=True)
    names = [name for nc in ncids for name in nc.variables]
    names = list(pd.Series(names, names).drop_duplicates())
    varlist = []
    for name in group_mapping:
        varlist.extend(filter(lambda x: x.startswith(name), names))
    rest = pd.Index(names).difference(pd.Index(varlist))
    varlist.extend(filter(lambda x: x in rest, names))
    varlist = _ensure_time_in_variable_list(varlist)
    varlist = name_dtype_dimension(ncids, varlist)
    return varlist


@memoize
def group_primary_ds(ncid, key):
    valid_keys = group_mapping.values()
    assert key in valid_keys, "valid keys: {}".format(valid_keys)
    if key not in ncid.variables:
        return pd.Series([])
    values = netCDF4.chartostring(ncid.variables[key][:])
    indices, values = zip(*[
        (ind, val) for (ind, val) in enumerate(values) if val])
    indices = np.array(indices)
    values = np.array(values)
    return pd.Series(indices, index=values, name=key)


def merged_group_primary_ds(ncids, key):
    valid_keys = group_mapping.values()
    assert key in valid_keys, "valid keys: {}".format(valid_keys)
    series = [group_primary_ds(nc, key) for nc in ncids]
    series.sort(key=len, reverse=True)
    combined = pd.concat(series).drop_duplicates().index
    return pd.Series(np.arange(len(combined)), index=combined, name=key)


def ncattrs(ncids):
    ncids = sorted(ncids, key=lambda nc: len(nc.variables), reverse=True)
    attrs = OrderedDefaultDict(OrderedDict)
    for nc in ncids:
        for name in nc.variables:
            attrs[name].update(nc[name].__dict__)
    attrs['time'].update(time_metadata)
    if 'values' in attrs:
        attrs['values'].update(fill_value)
    if 'sfcvalues' in attrs:
        attrs['values'].update(fill_value)
    return attrs


@memoize
def parse_date(ncid):
    "Fix typo in date if applicable."

    dates = netCDF4.chartostring(ncid['date'][:])
    _dates = pd.to_datetime(dates).to_series()
    dif = _dates.diff()
    mask = dif < pd.Timedelta(-1, 'ns')
    if not mask.any():
        # return pd.to_datetime(dates)
        return pd.Series(np.arange(len(_dates)), index=_dates, name='date')
    tofix = dif[mask].abs()
    print("date typos", list(_dates[mask]))
    freq = dif.dropna().value_counts().argmax()
    correction = _dates[mask] + freq + tofix
    print("corrections", list(correction))
    _dates[mask] = correction
    # return pd.to_datetime(_dates.values)
    return pd.Series(np.arange(len(_dates)), index=_dates, name='date')


def dates_as_array_of_strings(dates, ncids):
    ncid = ncids
    if isinstance(ncids, (list, tuple, set, dict)):
        ncid = ncids[0]
    ndates, numchars = ncid.variables['date'].shape
    stringtoarr = netCDF4.stringtoarr
    if getattr(dates, 'index', None) is None:
        dates_obj = dates
    else:
        dates_obj = dates.index
    dates_formatted = dates_obj.strftime("%Y%m%dT%H%M%SZ")
    dates_str = np.vstack(
        [stringtoarr(d, NUMCHARS=numchars) for d in dates_formatted])
    return dates_str


def merge_parse_date(ncids, fullday_timesteps=True):
    """
    This does not literally merge the dates from files as is.
    It looks at the total span of the combined dates and produces
    a time series with a uniform frequency. If the datasets have
    varying frequency, it opts for maximum occuring frequency.
    """
    dates = [parse_date(nc) for nc in ncids]
    freq = max([d.index.to_series().diff().dropna().value_counts().argmax()
                for d in dates])
    dates_min = min([d.index.min() for d in dates])
    dates_max = max([d.index.max() for d in dates])
    if fullday_timesteps:
        if dates_min.time():
            dates_min = pd.Timestamp(dates_min.date())
        if dates_max.time():
            dates_max = dates_max.date() + pd.Timedelta('1 day')
    dates = pd.date_range(dates_min, dates_max, freq=freq)
    return pd.Series(np.arange(len(dates)), index=dates, name='date')


def create_time_step(ntimesteps, dtype='i4'):
    return np.arange(ntimesteps, dtype=dtype) * 3


def create_time(dates, metadata=time_metadata):
    start_date = str(dates[0])
    units = metadata['units']
    calendar = metadata['calendar']
    units = " ".join([units.strip(), start_date])
    metadata['units'] = units
    tis = netCDF4.date2num(dates.to_pydatetime(), units, calendar)
    dtype = time_signature.dtype
    return tis.astype(dtype)


def domain_in_filename(files):
    domain_re = re.compile("(DOM\d+)").search
    if isinstance(files, (basestring, str)):
        dom = domain_re(files)
        if dom:
            return dom.group(0)
    objs = [domain_re(f) for f in files]
    doms = [obj.group(0) for obj in objs if obj]
    doms = set(doms)
    if len(doms) > 1:
        raise ValueError('Mutiple domains detected: {}'.format(doms))
    return doms.pop()


def get_indices(merged_ds, var_ds, varnames=None):
    """
    merge_ds: pd.Series
    var_ds: pd.Series
    varnames: (pd.Index or list or array or None)
    """
    required = merged_ds.index.intersection(var_ds.index)
    if varnames is not None:
        required = required.intersection(varnames)
        varnames = varnames.drop(required, errors='ignore')
    merged_index = merged_ds[required].values
    var_index = var_ds[required].values
    if varnames is not None:
        return merged_index, var_index, varnames
    return merged_index, var_index


def get_time_indices(merged_ds, var_ds, visited_dates):
    """
    merge_ds: pd.Series
    var_ds: pd.Series
    visited_dates: (pd.Index or list or np.array)
    """
    if not isinstance(visited_dates, (pd.Index, list, np.array)):
        msg = ("'visited_dates' must be one of these types "
               "(pd.Index list np.array)")
        raise ValueError(msg)
    visited_dates = pd.Index(visited_dates)
    notseen = var_ds.index.drop(visited_dates, errors='ignore')
    visited_dates = visited_dates.append(notseen).sort_values()
    return merged_ds[notseen].values, var_ds[notseen].values, visited_dates


def copy_group_data(ncids, outnc, group_key, varlist):
    valid_keys = group_mapping.keys()
    assert group_key in valid_keys, "valid keys: {}".format(valid_keys)

    primary_key = group_mapping[group_key]
    merged_ds = merged_group_primary_ds(ncids, key=primary_key)

    grp_names = filter(lambda x: x.startswith(group_key), varlist)
    for varname in grp_names:
        oobj = outnc.variables[varname]
        ndims = oobj.ndim
        pending = merged_ds.index[:]
        for nc in ncids:
            if varname not in nc.variables:
                continue
            var_ds = group_primary_ds(nc, primary_key)
            merged_index, var_index, pending = get_indices(
                merged_ds, var_ds, pending)
            if empty(merged_index, var_index):
                continue
            data = nc.variables[varname][var_index]
            if ndims == 1:
                oobj[merged_index] = data
            elif ndims == 2:
                oobj[merged_index, :] = data
            elif ndims == 3:
                oobj[merged_index, :, :] = data
            else:
                raise NotImplemented("only upto 3 dimensions implemented.")
            if empty(pending):
                break
    return


def merge_datasets(files, reference_file=None, domain=None, outfile=None,
                   fullday_timesteps=True):
    if domain is None:
        domain = domain_in_filename(files)
    if domain is None:
        domain = "DOMxx"

    ncids = allncids = [netCDF4.Dataset(f) for f in files]
    if reference_file is not None:
        refnc = netCDF4.Dataset(reference_file)
        allncids = [refnc] + ncids

    dates_ds = merge_parse_date(ncids, fullday_timesteps=fullday_timesteps)
    time_step = create_time_step(len(dates_ds))
    time_data = create_time(dates_ds.index)

    if outfile is None:
        start_ts = dates_ds.index[0].strftime("%Y%m%dT%H%M%SZ")
        end_ts = dates_ds.index[-1].strftime("%Y%m%dT%H%M%SZ")
        outfile = "1d_vars_{}_{}_{}.nc".format(domain, start_ts, end_ts)

    outnc = netCDF4.Dataset(outfile, "w")

    dimensions = merge_dimensions(allncids)
    for dname, dsize in dimensions.items():
        outnc.createDimension(dname, dsize)

    varlist = merge_variable_names(allncids)
    attrs = ncattrs(allncids)

    for vname, vsig in varlist.items():
        obj = outnc.createVariable(*vsig)
        obj.setncatts(attrs[vname])

    station_ds = merged_group_primary_ds(allncids, key='station_name')
    profile_ds = merged_group_primary_ds(allncids, key='var_name')
    surface_ds = merged_group_primary_ds(allncids, key='sfcvar_name')

    for group_key in group_mapping:
        copy_group_data(allncids, outnc, group_key, varlist)

    ts_obj = outnc.variables['time_step']
    ts_obj[:] = time_step[:]

    time_obj = outnc.variables['time']
    time_obj[:] = time_data[:]

    date_obj = outnc.variables['date']
    date_str = dates_as_array_of_strings(dates_ds.index, ncids)
    date_obj[:, :] = date_str

    # populating heights data
    print('populating heights')
    heights_obj = outnc.variables['heights']
    req_stations = station_ds.index[:]
    req_profile = profile_ds.index[:]
    for nc in ncids:
        if 'heights' not in nc.variables:
            continue
        if 'var_name' not in nc.variables:
            continue
        if 'station_name' not in nc.variables:
            continue
        nc_station_ds = group_primary_ds(nc, 'station_name')
        nc_profile_ds = group_primary_ds(nc, 'var_name')

        if not empty(req_stations):
            station_index, nc_station_index, req_stations = get_indices(
                station_ds, nc_station_ds, req_stations)
        else:
            station_index, nc_station_index = get_indices(
                station_ds, nc_station_ds)
        if not empty(req_profile):
            profile_index, nc_profile_index, req_profile = get_indices(
                profile_ds, nc_profile_ds, req_profile)
        else:
            profile_index, nc_profile_index = get_indices(
                profile_ds, nc_profile_ds)

        print('station_index', station_index)
        print('nc_station_index', nc_station_index)
        print('req_stations', req_stations)
        print('profile_index', profile_index)
        print('nc_profile_index', nc_profile_index)
        print('req_profiles', req_profile)

        if empty(station_index):
            continue
        if empty(profile_index):
            continue
        
        station_slices, nc_station_slices = index2slice(
            station_index, nc_station_index)
        profile_slices, nc_profile_slices = index2slice(
            profile_index, nc_profile_index)
        for ps, nc_ps in zip(profile_slices, nc_profile_slices):
            for sta, nc_sta in zip(station_slices, nc_station_slices):
                data = nc.variables['heights'][:, nc_ps, nc_sta]
                heights_obj[:, ps, sta] = data
        # if station_index.size and profile_index.size:
        #     data = nc.variables['heights'][:]
        #     for pind, pind_m in zip(nc_profile_index, profile_index):
        #         for sind, sind_m in zip(nc_station_index, station_index):
        #             heights_obj[:, pind_m, sind_m] = data[:, pind, sind]

        # if req_stations.empty and req_profile.empty:
        if empty(req_stations, req_profile):
            break

    # populating profile data
    profile_obj = outnc.variables['values']
    seen_dates = pd.Index([])
    for nc in ncids:
        if 'values' not in nc.variables:
            continue
        if 'var_name' not in nc.variables:
            continue
        nc_dates_ds = parse_date(nc)
        nc_station_ds = group_primary_ds(nc, 'station_name')
        nc_profile_ds = group_primary_ds(nc, 'var_name')

        merged_date_index, nc_date_index, seen_dates = get_time_indices(
            dates_ds, nc_dates_ds, seen_dates)
        station_index, nc_station_index = get_indices(
            station_ds, nc_station_ds)
        profile_index, nc_profile_index = get_indices(
            profile_ds, nc_profile_ds)

        if empty(merged_date_index):
            continue
        if empty(station_index):
            continue
        if empty(profile_index):
            continue

        merged_date_slices, nc_date_slices = index2slice(
            merged_date_index, nc_date_index)
        station_slices, nc_station_slices = index2slice(
            station_index, nc_station_index)
        profile_slices, nc_profile_slices = index2slice(
            profile_index, nc_profile_index)

        iprofile_obj = nc.variables['values']

        for date_s, nc_date_s in zip(merged_date_slices, nc_date_slices):
            for ps, nc_ps in zip(profile_slices, nc_profile_slices):
                for sta, nc_sta in zip(station_slices, nc_station_slices):
                    data = iprofile_obj[nc_date_s, :, nc_ps, nc_sta]
                    profile_obj[date_s, :, ps, sta] = data

        # for (tindex, tindex_m) in zip(nc_date_index, merged_date_index):
        #     for pind, pind_m in zip(nc_profile_index, profile_index):
        #         for (sind, sind_m) in zip(nc_station_index, station_index):
        #             data = iprofile_obj[tindex, :, pind, sind]
        #             profile_obj[tindex_m, :, pind_m, sind_m] = data
        # data = iprofile_obj[:]
        # for (tindex, tindex_m) in zip(nc_date_index, merged_date_index):
        #     for pind, pind_m in zip(nc_profile_index, profile_index):
        #         for (sind, sind_m) in zip(nc_station_index, station_index):
        #             # data = iprofile_obj[nc_station_index, :, pind, sind]
        #             profile_obj[tindex_m, :, pind_m, sind_m] = data[tindex, :, pind, sind]

    # populating surface data
    surface_obj = outnc.variables['sfcvalues']
    seen_dates = pd.Index([])
    for nc in ncids:
        if 'sfcvalues' not in nc.variables:
            continue
        if 'sfcvar_name' not in nc.variables:
            continue
        nc_dates_ds = parse_date(nc)
        nc_station_ds = group_primary_ds(nc, 'station_name')
        nc_surface_ds = group_primary_ds(nc, 'sfcvar_name')

        merged_date_index, nc_date_index, seen_dates = get_time_indices(
            dates_ds, nc_dates_ds, seen_dates)
        station_index, nc_station_index = get_indices(
            station_ds, nc_station_ds)
        surface_index, nc_surface_index = get_indices(
            surface_ds, nc_surface_ds)

        if empty(merged_date_index):
            continue
        if empty(station_index):
            continue
        if empty(surface_index):
            continue

        merged_date_slices, nc_date_slices = index2slice(
            merged_date_index, nc_date_index)
        station_slices, nc_station_slices = index2slice(
            station_index, nc_station_index)
        surface_slices, nc_surface_slices = index2slice(
            surface_index, nc_surface_index)

        isurface_obj = nc.variables['sfcvalues']

        for date_s, nc_date_s in zip(merged_date_slices, nc_date_slices):
            for ss, nc_ss in zip(surface_slices, nc_surface_slices):
                for sta, nc_sta in zip(station_slices, nc_station_slices):
                    data = isurface_obj[nc_date_s, nc_ss, nc_sta]
                    surface_obj[date_s, ss, sta] = data

        # data = isurface_obj[:]
        # for (tindex, tindex_m) in zip(nc_date_index, merged_date_index):
        #     for surf_ind, surf_ind_m in zip(nc_surface_index, surface_index):
        #         # data = isurface_obj[nc_date_index, surf_ind, nc_station_index]
        #         surface_obj[tindex_m, surf_ind_m, station_index] = data[tindex, surf_ind, nc_station_index]

    global_attrs = OrderedDict()
    for nc in ncids:
        global_attrs.update(nc.__dict__)

    global_attrs['merged_files'] = ", ".join(files)
    outnc.setncatts(global_attrs)

    outnc.sync()
    outnc.close()
    for nc in allncids:
        nc.close()

    print("Created {}".format(outfile))
    return


if __name__ == '__main__':
    import click

    help_timesteps = """
"fullday-timesteps" flag ensures that all timestamps exists for the complete
day by filling out any missing timesteps. The other flag restricts this
extent to the ones found the files. In either of the cases, any missing
timesteps in between the files are filled to have a continuous time series.
(default: is to have fullday timesteps).
""".strip().replace('\n', ' ')

    help_outfile = """The name of the output file. If not provided, it creates
a name based on the time-steps in the files"""

    help_domain = """if not provided, tries to infer domain name the files
"""

    @click.command()
    @click.option('--fullday-timesteps/--no-fullday-timesteps',
                  default=True,
                  help=help_timesteps)
    @click.option('--reference-file', type=click.Path(),
                  help="reference file for meta-data")
    @click.option('--domain', '-d', help=help_domain)
    @click.option('--outfile', '-o', type=click.Path(), help=help_outfile)
    @click.argument('files', nargs=-1, type=click.Path())
    def cli(fullday_timesteps, reference_file, domain, outfile, files):
        """
        Merge mutiple meteogram files into a single file.

        The script is designed to handle missing data like stations, profile
        variables (a.k.a volume variables), surface variables and time. It
        means, the files involved in the merge process may vary in the
        number of stations and/or variables.

        With respect to processing time dimension, any duplicated timesteps are
        automatically droped and any missing timesteps are filled with a default
        fill value. This is done to achive a continuous time series for the entire
        dataset. Moreover, "fullday-timesteps" flag ensures that all timesteps
        from begining of a day untill the end of the day exists in the outfile.

        In few meteogram files, a bug (typo) in the timestamp for the last
        timestep was discovered. This scipt also accounts to fix that bug.

        To avoid inconsistencies in meteogram files (across simulations), provide
        a well known good meteogram which has complete set of station, variables
        (profile and surface) etc., as a "reference file". This file is then used
        for copying any missing meta-data.

        FILES: files to merge
        
        \b
        EXAMPLE
        -------
        python merge.py /work/bm0834/k203095/OUTPUT/20130420-default-redone_v1/DATA/1d_vars_DOM01_*.nc
        """
        click.echo("fullday-timesteps: {}".format(fullday_timesteps))
        click.echo("reference_file: {}".format(reference_file))
        click.echo("domain: {}".format(domain))
        click.echo("outfile: {}".format(outfile))
        click.echo("files: {}".format(files))
        
        merge_datasets(files, reference_file, domain, outfile, fullday_timesteps)
    
    cli()
