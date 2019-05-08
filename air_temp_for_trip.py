import argparse
import datetime
import functools
import math
import os
import re

import netCDF4
import scipy.interpolate

SELF_PATH = os.path.dirname(os.path.abspath(__file__))

TRIP_FILE_LINE = re.compile(
    # For example: 2018-11-16 00:00:00+00    42.24    -8.73
    r"(?P<date>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\S*\s+"
    r"(?P<lat>\S+)\s+"
    r"(?P<lon>\S+)\s?.*"
)

NC_LON = tuple(0.0 + 2.5 * n for n in range(144))  # 0.0 -> 357.5
NC_LAT = tuple(90.0 - 2.5 * n for n in range(73))  # 90.0 -> -90.0
NC_LEVELS_COUNT = 17  # The actual count is NC_LEVELS_COUNT + 1 for level_1013

# First 5 are date, lat_orig, lon_orig, lat_nc, lon_nc
# Next 18 are levels
OUT_FILE_LINE = "    ".join("{" + str(i) + "}" for i in range(5 + 18)) + "\n"


@functools.lru_cache(maxsize=512, typed=False)
def approximate_position(nc_positions, lat=None, lon=None):
    if lat is not None:
        current_position = lat
    else:  # Convert longitude from [-180, 180] to [0, 360] format
        current_position = lon if lon >= 0 else 360 + lon
    pos = min(nc_positions, key=lambda p: math.fabs(p - current_position))
    return nc_positions.index(pos)


def six_hour_aligned_dates(start, end):
    start_from_zero = datetime.datetime(start.year, start.month, start.day)
    for n in range(((end - start).days + 1) * 4):
        yield start_from_zero + datetime.timedelta(hours=n * 6)


def calc_hourly(data):
    x_old = list(0 + (x * 6) for x in range(len(data[0])))
    interpolation_func = scipy.interpolate.interp1d(x_old, data, kind="cubic")

    # The left boundary is 18:00
    # So, we can't interpolate up to 23:00
    # So, we have to slice our x range
    x_new = range(len(data[0]) * 6)[:-5]
    return interpolation_func(x_new)


def main(args):
    nc_file_year = "".join(c for c in args.nc_file[0] if c.isdigit())
    if not nc_file_year:
        raise ValueError("NC_FILE's filename should contain a year")

    trip_by_location = {}
    trip_by_date = {}
    trip_end_date = None
    with open(args.trip_file, "r", encoding="utf8") as trip_file:
        for line in trip_file:
            match = TRIP_FILE_LINE.match(line)
            if match is not None:
                date, lat, lon = match.groups()

                date = datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
                lat, lon = float(lat), float(lon)

                location = (lat, lon)
                # Group the trip by location: convenient for nc data processing
                if location not in trip_by_location:
                    lat_index = approximate_position(NC_LAT, lat=lat)
                    lon_index = approximate_position(NC_LON, lon=lon)

                    trip_by_location[location] = {
                        "lat": lat,
                        "lon": lon,
                        "nc_lat": NC_LAT[lat_index],
                        "nc_lon": NC_LON[lon_index],
                        "nc_lat_index": lat_index,
                        "nc_lon_index": lon_index,
                        "start_date": date,
                        "end_date": date,
                    }
                else:
                    trip_by_location[location]["end_date"] = date

                # Group the trip by date: convenient for making the result
                trip_by_date[date] = trip_by_location[location]

                trip_end_date = date

    if trip_end_date is None:
        os.sys.exit(1)

    if trip_end_date.year != int(nc_file_year):
        raise ValueError("Trip should be in boundaries of NC_FILE")

    nc_air_by_time = netCDF4.Dataset(args.nc_file[0]).variables["air"]
    # +1 for level_1013
    # level_1013 is under index 0; level_? is under index 17
    # In total, 18 levels
    nc_data = [[] for _ in range(1 + NC_LEVELS_COUNT)]
    processed_dates = set()
    for loc in trip_by_location.values():
        lat_i = loc["nc_lat_index"]
        lon_i = loc["nc_lon_index"]
        for date in six_hour_aligned_dates(loc["start_date"], loc["end_date"]):
            # An easy solution for days that were split between locations
            # We shouldn't process the dates more than once, or else,
            # nc_data would contain data duplicates
            if date in processed_dates:
                continue

            time_index = date.timetuple().tm_yday * 4 - 1  # DOY * 4 - 1
            air = nc_air_by_time[time_index]

            lvl_1013 = air[0][lat_i][lon_i] - 273.15
            lvl_1013 -= (13/75 * (air[1][lat_i][lon_i] - air[0][lat_i][lon_i]))
            nc_data[0].append(lvl_1013)

            for level_i in range(NC_LEVELS_COUNT):
                # The levels are following from top to down: 1000 -> 100, or so
                # +1 is an offset, considering that level_1013 resides at 0
                nc_data[level_i + 1].append(air[level_i][lat_i][lon_i] - 273.15)

            processed_dates.add(date)

    # Dev sanity check
    assert int(len(trip_by_date) / 24 * 4) == len(nc_data[0])

    hourly = calc_hourly(nc_data)

    filename = args.out_file
    if filename is None:
        # No SELF_PATH join, so it could be found in PWD
        filename = "out_for_{0}_with_{1}.txt".format(
            ".".join(os.path.basename(args.trip_file).split(".")[:-1]),
            ".".join(os.path.basename(args.nc_file[0]).split(".")[:-1])
        )

    with open(filename, "w", encoding="utf8") as fd:
        line = OUT_FILE_LINE.format(
            "date", "lat_orig", "lon_orig", "lat_nc", "lon_nc",
            "level_1013", "level_1000", *("level_X" for _ in range(16))
        )
        fd.write(line)
        for (date, loc_info), *levels in zip(trip_by_date.items(), *hourly):
            line = OUT_FILE_LINE.format(
                date, loc_info['lat'], loc_info['lon'],
                loc_info['nc_lat'], loc_info['nc_lon'],
                *levels
            )
            fd.write(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        dest="nc_file",
        metavar="NC_FILE",
        type=lambda file: (
            os.path.join(SELF_PATH, file) if not os.path.isabs(file) else file
        ),
        nargs=1,
        help="NC_FILE should be supplied as positional argument (required)"
    )
    parser.add_argument(
        "--trip-file",
        dest="trip_file",
        type=lambda file: (
            os.path.join(SELF_PATH, file) if not os.path.isabs(file) else file
        ),
        default="sample.txt",
        help="Trip file with lines consisting of "
             "<datetime> <latitude> <longitude>"
    )
    parser.add_argument(
        "--out-file",
        dest="out_file",
        default=None,
        help="Filename for resulting file; "
             "if not supplied, it'll be auto-generated"
    )
    main(parser.parse_args())
