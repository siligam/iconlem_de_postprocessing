# -*- coding: utf-8 -*-

from __future__ import print_function

import math
import collections

"""Geolocation (lat,lon) utility functions for calculating bounding box and
distances.

Inspiration from the following sources:
# https://github.com/jfein/PyGeoTools/blob/master/geolocation.py
# http://janmatuschek.de/LatitudeLongitudeBoundingCoordinates
"""
__author__ = "Pavan Siligam <siligam@dkrz.de>"


_geoboundingbox = collections.namedtuple('geoboundingbox', 'll ur')


class geoboundingbox(_geoboundingbox):

    def __new__(cls, lowerleft, upperright):
        FLIP = False
        OPPOSITE = False
        ll_lat, ll_lon = lowerleft
        ur_lat, ur_lon = upperright
        # check if it is indeed lowerleft and upperright corners
        if sum([ll_lat > ur_lat, ll_lon > ur_lon]) == 1:
            print('Recieved opposite corners... correcting corners')
            OPPOSITE = True
            lowerleft = ur_lat, ll_lon
            upperright = ll_lat, ur_lon
        if lowerleft[0] > upperright[0]:
            FLIP = True
            print('Flipping corners...')
            lowerleft, upperright = upperright, lowerleft
        if OPPOSITE:
            if FLIP:
                direction = 'NorthWest'
            else:
                direction = 'SouthEast'
        else:
            if FLIP:
                direction = 'SouthWest'
            else:
                direction = 'NorthEast'
        ll = loc(*lowerleft)
        ur = loc(*upperright)
        box = super(geoboundingbox, cls).__new__(cls, ll, ur)
        box.direction = direction
        return box

    @property
    def ul(self):
        return loc(self.ur.lat, self.ll.lon)

    @property
    def lr(self):
        return loc(self.ll.lat, self.ur.lon)

    @property
    def tics(self):
        return (self.ul, self.ur, self.lr, self.ll)

    corners = tics

    @property
    def width(self):
        return self.lr.distance_to(self.ll)

    @property
    def height(self):
        return self.ul.distance_to(self.ll)

    @property
    def area(self):
        return self.width * self.height
    
    @property
    def center(self):
        return loc(
            (self.ll.lat + self.ur.lat)/2,
            (self.ll.lon + self.lr.lon)/2)

    @property
    def left(self):
        return loc(
            (self.ll.lat + self.ul.lat)/2,
            (self.ll.lon + self.ul.lon)/2,
            )

    @property
    def top(self):
        return loc(
            (self.ul.lat + self.ur.lat)/2,
            (self.ul.lon + self.ur.lon)/2,
            )

    @property
    def right(self):
        return loc(
            (self.ur.lat + self.lr.lat)/2,
            (self.ur.lon + self.lr.lon)/2,
        )

    @property
    def bottom(self):
        return loc(
            (self.lr.lat + self.ll.lat)/2,
            (self.lr.lon + self.ll.lon)/2,
        )

    @property
    def edges(self):
        return (self.left, self.top, self.right, self.bottom)

    def __contains__(self, point):
        return (
            self.ll.lat <= point.lat <= self.ul.lat and
            self.ll.lon <= point.lon <= self.lr.lon)

    def enlarge(self, buf_size):
        return geoboundingbox(
            self.ll.bounding_box(buf_size).ll,
            self.ur.bounding_box(buf_size).ur,
        )

    def shrink(self, buf_size):
        return geoboundingbox(
            self.ll.bounding_box(buf_size).ur,
            self.ur.bounding_box(buf_size).ll,
        )


_loc = collections.namedtuple('loc', 'lat lon')


class loc(_loc):

    RADIUS = 6378.1  # radius of Earth in km
    LON_BOUNDS = -180, 180
    LAT_BOUNDS = -90, 90

    def __new__(cls, lat, lon):
        errors = []
        lat_min, lat_max = cls.LAT_BOUNDS
        if not (lat_min <= lat <= lat_max):
            msg = "lat({}) outside valid bounds {!r}"
            errors.append(msg.format(lat, cls.LAT_BOUNDS))
        lon_min, lon_max = cls.LON_BOUNDS
        if not (lon_min <= lon <= lon_max):
            msg = "lon({}) outside valid bounds {!r}"
            errors.append(msg.format(lon, cls.LON_BOUNDS))
        if errors:
            raise ValueError(", ".join(errors))
        return super(loc, cls).__new__(cls, lat, lon)

    @classmethod
    def from_radians(cls, rlat, rlon):
        lat = math.degrees(rlat)
        lon = math.degrees(rlon)
        return cls(lat, lon)

    @property
    def rlat(self):
        return math.radians(self.lat)

    @property
    def rlon(self):
        return math.radians(self.lon)

    def distance_to(self, other):
        '''
        Computes the great circle distance between this Loc instance
        and the other.
        '''
        if not isinstance(other, self.__class__):
            other = self.__class__(*other)
        acos = math.acos
        sin = math.sin
        cos = math.cos
        rlat = self.rlat
        rlon = self.rlon
        other_rlat = other.rlat
        other_rlon = other.rlon
        return self.RADIUS * acos(
            sin(rlat) * sin(other_rlat) +
            cos(rlat) * cos(other_rlat) * cos(rlon - other_rlon))

    def bounding_box(self, distance):
        "distance in KM"
        asin = math.asin
        sin = math.sin
        cos = math.cos
        radians = math.radians
        degrees = math.degrees
        rlat = self.rlat
        rlon = self.rlon

        lat_bound_min, lat_bound_max = map(radians, self.LAT_BOUNDS)
        lon_bound_min, lon_bound_max = map(radians, self.LON_BOUNDS)

        if distance < 0:
            raise ValueError("distance must to positive")

        rdist = 1.0 * distance / self.RADIUS
        lat_min = rlat - rdist
        lat_max = rlat + rdist
        if (lat_min > lat_bound_min) and (lat_max < lat_bound_max):
            dlon = asin(sin(rdist) / cos(rlat))
            lon_min = rlon - dlon
            if lon_min < lon_bound_min:
                lon_min += 2 * math.pi
            lon_max = rlon + dlon
            if lon_max > lon_bound_max:
                lon_max -= 2 * math.pi
        else:
            lat_min = max(lat_min, lat_bound_min)
            lat_max = min(lat_max, lat_bound_max)
            lon_min = lon_bound_min
            lon_max = lon_bound_max
        lowerleft = tuple(map(degrees, (lat_min, lon_min)))
        upperright = tuple(map(degrees, (lat_max, lon_max)))
        return geoboundingbox(lowerleft, upperright)


if __name__ == "__main__":
    # Test degree to radian conversion
    loc1 = loc(26.062951, -80.238853)
    loc2 = loc.from_radians(loc1.rlat, loc1.rlon)
    assert (loc1.rlat == loc2.rlat and loc1.rlon == loc2.rlon and
            loc1.lat == loc2.lat and loc1.lon == loc2.lon)

    # Test distance between two locations
    loc1 = loc(26.062951, -80.238853)
    loc2 = loc(26.060484, -80.207268)
    assert loc1.distance_to(loc2) == loc2.distance_to(loc1)

    # Test bounding box
    loc1 = loc(26.062951, -80.238853)
    distance = 1  # 1 kilometer
    SW_loc, NE_loc = loc1.bounding_box(distance)
    print(loc1.distance_to(SW_loc))
    print(loc1.distance_to(NE_loc))
