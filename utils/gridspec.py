#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import numpy as np
from geolocation import loc
from collections import OrderedDict, namedtuple


"""Generates description of the target grid for a given resolution and lat-lon
bounds. This description is one of the input parameters for the cdo operator
'remap'.
"""

__author__ = "Pavan Siligam <siligam@dkrz.de>"
__contributors__ = ("Rieke Heinze <rieke.heinze@mpimet.mpg.de>",)


_bounds = namedtuple("Bounds", "lon_min lon_max lat_min lat_max")

griddes = OrderedDict()
griddes["gridtype"] = "lonlat"
griddes["gridsize"] = ""
griddes["xname"] = "lon"
griddes["xlongname"] = "longitude"
griddes["xunits"] = "degrees_east"
griddes["yname"] = "lat"
griddes["ylongname"] = "latitude"
griddes["yunits"] = "degrees_north"
griddes["xsize"] = ""
griddes["ysize"] = ""
griddes["xfirst"] = ""
griddes["xinc"] = ""
griddes["yfirst"] = ""
griddes["yinc"] = ""


_resolution_factor = 0.67

icon_grids = {"DOM03": 0.156,
              "DOM02": 0.312,
              "DOM01": 0.625}


def get_resolution(grid_name_or_resolution_in_km, icon_grids=icon_grids):
    """Resolution of the target grid.

    :param grid_name_or_resolution_in_km: Only in case of named grid,
    the resolution is computed by taking the product of native
    resolution of the grid and the resolution factor. Otherwise
    the value is returned as is.
    """
    def handle_grid_names(name):
        possible_names = ("d{}", "d{:02d}", "dom{}",
                          "dom{:02d}", "domain{}", "domain{:02d}")
        req_names = sorted(icon_grids.keys())
        if name in req_names:
            return name
        for idx, rname in enumerate(req_names, 1):
            _name = " ".join([n.format(idx) for n in possible_names])
            if name in _name:
                return rname
        return name
    try:
        resolution = float(grid_name_or_resolution_in_km)
    except ValueError:
        grid_name = handle_grid_names(grid_name_or_resolution_in_km)
        resolution = icon_grids[grid_name] * _resolution_factor
    return resolution


def get_bounds(bounds_or_filename):
    try:
        # expecting a string with comma seperated values
        bounds = _bounds(*map(float, bounds_or_filename.split(",")))
    except AttributeError:
        # could be a tuple or a list
        bounds = _bounds(*map(float, bounds_or_filename))
    except ValueError:
        # could be a filename
        bounds = bounds_from_file(bounds_or_filename)
    except IOError:
        raise
    return bounds


def bounds_from_file(filename):
    import netCDF4
    with netCDF4.Dataset(filename) as nc:
        try:
            lats = np.rad2deg(nc.variables['clat'][:])
            lons = np.rad2deg(nc.variables['clon'][:])
        except:
            lats = nc.variables['lat'][:]
            lons = nc.variables['lon'][:]
        lat_min = lats.min()
        lat_max = lats.max()
        lon_min = lons.min()
        lon_max = lons.max()
    return _bounds(lon_min, lon_max, lat_min, lat_max)


def gridspec(bounds_or_filename, grid_name_or_resolution_in_km,
             filehandle=None):
    gs = griddes.copy()
    resolution = get_resolution(grid_name_or_resolution_in_km)
    bounds = get_bounds(bounds_or_filename)
    lon_min = bounds.lon_min
    lon_max = bounds.lon_max
    lat_min = bounds.lat_min
    lat_max = bounds.lat_max
    ll = loc(lat_min, lon_min)
    ur = loc(lat_max, lon_max)
    dist = int(ll.distance_to(ur))
    nparts = dist if dist > 0 else 11
    points_lats = np.linspace(lat_min, lat_max, nparts)
    points_lons = np.linspace(lon_min, lon_max, nparts)
    points = zip(points_lats, points_lons)
    inc_yx = [_increment(pnt, resolution) for pnt in points]
    inc_x, inc_y = zip(*inc_yx)
    xinc = round(np.average(inc_x), 4)
    yinc = round(np.average(inc_y), 4)

    gs['xfirst'] = round(lon_min, 4)
    gs['yfirst'] = round(lat_min, 4)
    gs['xinc'] = round(xinc, 4)
    gs['yinc'] = round(yinc, 4)
    gs['xsize'] = int(np.floor((lon_max - lon_min)/xinc))
    gs['ysize'] = int(np.floor((lat_max - lat_min)/yinc))
    gs['gridsize'] = gs['xsize'] * gs['ysize']
    gs_text = ["{:<10} = {}".format(k, v) for k, v in gs.items()]
    gs_text = "\n".join(gs_text)
    if filehandle is None:
        print(gs_text)
    else:
        print(gs_text, file=filehandle)
    return gs


def _increment(point, resolution, loc=loc):
    p = loc(*point)
    b = p.bounding_box(resolution)
    x = p.lon - b.ll.lon
    y = p.lat - b.ll.lat
    return x, y


def cli():
    import argparse

    class customFormatter(argparse.RawTextHelpFormatter):
        pass

    def handle_grid_names(name):
        possible_names = ("{:02d}", "d{}", "d{:02d}", "dom{}",
                          "dom{:02d}", "domain{}", "domain{:02d}")
        req_names = "DOM01", "DOM02", "DOM03"
        if name in req_names:
            return name
        for idx, rname in enumerate(req_names, 1):
            _name = " ".join([n.format(idx) for n in possible_names])
            if name in _name:
                return rname
        return name

    doc_strings = {}
    doc_strings['description'] = """
Generates description of the target grid for a given resolution and lat-lon
bounds. This description is one of the input parameters for the cdo operator
'remap'.
""".strip()
    doc_strings['resolution'] = """
Resolution of the target grid in Kilometers.
""".strip()
    doc_strings['domain'] = """
Domain to be used for calculating resolution. In this case, the resolution of
the target grid is most similar to the native resolution of the domain
selected. The triangle edge lengths in the ICON grid are 625m, 312m, 156m for
DOM01, DOM02, DOM03, respectively.
""".strip()
    doc_strings['outfile'] = """
File to which output must be written. (default: stdout)
""".strip()
    doc_strings['bounds'] = """
Comma separated corner tics of region of interest given as
'lon1,lon2,lat1,lat2'. These corner tics refer to the lower left and upper
right corners.
""".strip()
    doc_strings['ncfile'] = """
Grid file from which the corner tics are to be read. Use this option if
the full extent of grid is considered to be the region of interest.
""".strip()
    doc_strings['epilog'] = """

options '--resolution' and '--domain' are mutually exclusive.
Provide either one of these.
In case of option '--domain', the resolution obtained from the specified domain
(DOM01,DOM02,DOM03) is obtained by multiplying the native triangle edge length
with a factor of 0.67 which is equivalent to defining the resolution as square
root of the area of the triangle.
For details about this factor, please refer to:
(http://onlinelibrary.wiley.com/doi/10.1002/2015MS000431/full#jame20180-app-0003)

"A. Dipankar, B. Stevens, R. Heinze, C. Moseley, G. Zängl, M. Giorgetta, and
S. Brdar (2015),Large eddy simulation using the general circulation model ICON,
J. Adv. Model. Earth Syst., 7, 963–986, doi:10.1002/2015MS000431."

options '--bounds' and '--ncfile' are also mutually exclusive.
Provide either one of these.

Example:
--------
1. To generate a grid description for a 10x10 grid around joyce for DOM03

 python {0} --outfile joyce_10x10_griddes --domain DOM03 --bounds 6.2715,6.5564,50.8191,50.9988

2. To generate a grid description for entire domain for DOM03

 python {0} -d DOM03 -f /work/bm0834/k203095/OUTPUT/GRIDS/GRID_3d_fine_DOM03_ML.nc

3. To generate a grid description for a custom resolution of 400m

 python {0} --resolution=0.4 --bounds 6.2715,6.5564,50.8191,50.9988

  TIP: Leave no whitespaces between the option and its value.

 python {0} -ogriddes -dDOM02 -b6.27,6.55,50.81,50.99
 python {0} --outfile=griddes --domain=DOM02 --bounds=6.27,6.55,50.81,50.99

""".format(os.path.basename(__file__))

    parser = argparse.ArgumentParser(description=doc_strings['description'],
                                     epilog=doc_strings['epilog'],
                                     formatter_class=customFormatter)
    parser.add_argument('-o', '--outfile',
                        help=doc_strings['outfile'],
                        type=argparse.FileType('wb', 0), default='-')
    res_group = parser.add_mutually_exclusive_group(required=True)
    res_group.add_argument('-r', '--resolution',
                           help=doc_strings['resolution'])
    res_group.add_argument('-d', '--domain',
                           type=handle_grid_names,
                           choices=("DOM01", "DOM02", "DOM03"),
                           help=doc_strings['domain'])
    tics_group = parser.add_mutually_exclusive_group(required=True)
    tics_group.add_argument('-b', '--bounds',
                            help=doc_strings['bounds'])
    tics_group.add_argument('-f', '--ncfile',
                            help=doc_strings['ncfile'])

    args = parser.parse_args()
    grid_name_or_resolution_in_km = args.resolution or args.domain
    bounds_or_filename = args.bounds or args.ncfile
    gridspec(bounds_or_filename, grid_name_or_resolution_in_km, args.outfile)


if __name__ == '__main__':
    cli()
