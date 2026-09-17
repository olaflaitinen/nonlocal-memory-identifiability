# Data directory

This directory documents the empirical data used in Section 6 of the
manuscript. The raw data files are **not** committed to this repository. Only
this file and `checksums.sha256` are tracked.

## Data package

The analysis uses the publicly archived data package

> Latham ADM, Boutin S (2019) Data from: Wolf ecology and caribou-primary
> prey-wolf spatial relationships in low productivity peatland complexes in
> northeastern Alberta. Movebank Data Repository.
> https://doi.org/10.5441/001/1.7vr1k987

The data package is released under the Creative Commons Zero (CC0 1.0) public
domain dedication. Every output derived from these data cites the data package
and its digital object identifier.

The related publications are

> Latham ADM (2009) Wolf ecology and caribou-primary prey-wolf spatial
> relationships in low productivity peatland complexes in northeastern Alberta.
> Dissertation, University of Alberta

and

> Latham ADM, Latham MC, Boyce MS, Boutin S (2011) Movement responses by wolves
> to industrial linear features and their effect on woodland caribou in
> northeastern Alberta. Ecological Applications 21(8):2854-2865.
> https://doi.org/10.1890/11-0666.1

## How to obtain the files

1. Open https://doi.org/10.5441/001/1.7vr1k987 and download the data package.
2. Place the two comma separated files in `data/raw/`.
3. The loader accepts the original file names, which contain spaces, and the
   same names with underscores instead of spaces:
   - `Latham Alberta Wolves.csv` or `Latham_Alberta_Wolves.csv`
   - `Latham Alberta Wolves-reference-data.csv` or
     `Latham_Alberta_Wolves-reference-data.csv`
4. Verify the downloads against the tracked manifest:

```bash
cd data/raw && sha256sum -c ../checksums.sha256
```

The expected checksums are recorded in `data/checksums.sha256`. The
preprocessing script `experiments/exp_4_1_wolf_preprocess.py` verifies them
before it reads either file and stops with an informative message on a
mismatch.

## Schema of the event file

The event file is in the Movebank attribute format. The following columns are
required by `nmi.wolf.schema` and are validated before any analysis.

| Column | Meaning |
|---|---|
| `event-id` | Unique identifier of one recorded event |
| `visible` | Flag that marks records hidden by the data owner |
| `timestamp` | Time of the fix, in coordinated universal time |
| `location-long` | Longitude of the fix, World Geodetic System 1984 |
| `location-lat` | Latitude of the fix, World Geodetic System 1984 |
| `gps:dop` | Dilution of precision reported by the collar |
| `gps:fix-type` | Dimension of the fix, two or three |
| `gps:satellite-count` | Number of satellites used for the fix |
| `sensor-type` | Sensor that produced the record |
| `individual-taxon-canonical-name` | Scientific name of the tracked species |
| `tag-local-identifier` | Identifier of the collar within the study |
| `individual-local-identifier` | Identifier of the animal within the study |
| `study-name` | Name of the study in the Movebank repository |

## Schema of the reference file

The reference file carries 22 columns, of which the following are required.

| Column | Meaning |
|---|---|
| `animal-id` | Identifier of the animal, equal to `individual-local-identifier` |
| `deployment-id` | Identifier of one deployment of a collar on an animal |
| `animal-sex` | Sex of the animal |
| `animal-life-stage` | Life stage of the animal at collaring |
| `animal-comments` | Free text whose first field, before the first semicolon, is the pack name; the dispersing wolf carries `Loner` |
| `deploy-on-date` | Date on which the collar was deployed |
| `deploy-off-date` | Date on which the collar was removed or stopped |
| `deployment-end-type` | Reason for the end of the deployment |
| `duty-cycle` | Programmed fix interval of the collar |

## Properties of the verified files

The following properties of the verified data package are used as regression
checks in the summary of Experiment 4.1. A deviation indicates that a different
version of the data package has been downloaded.

- 15 159 GPS fixes.
- 12 animals, of which 7 are female and 5 are male.
- 16 deployments, since several animals carried successive collars.
- Every record is flagged as visible.
- No duplicate timestamp occurs within an animal.
- 3 874 fixes are two-dimensional GPS fixes, that is `gps:fix-type == 2`.
- Every longitude lies within Universal Transverse Mercator zone 12.
- 8 animals, with the identifiers 1, 2, 3, 6, 10, 13, 30 and 31, have a
  monitoring span of at least 90 days.

## Processing

`experiments/exp_4_1_wolf_preprocess.py` keeps only GPS records of identified
animals, drops records flagged as not visible or as outliers, drops exact
duplicate timestamps within an animal, parses timestamps as coordinated
universal time and projects the coordinates from World Geodetic System 1984 to
NAD83 / UTM zone 12N (EPSG:26912) so that distances are measured in metres. The
resulting internal table is written to `results/raw/exp_4_1/`, which is not
tracked, with the columns `individual_id`, `timestamp_utc`, `easting_m` and
`northing_m`.

## Tests

The tests of the wolf pipeline never read the data package. They use synthetic
point patterns generated in code on an artificial domain, so that the test
suite runs without any download.
