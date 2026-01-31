# Xeno-Canto v3 Metadata — Field Reference

Source: XC API v3 (`https://xeno-canto.org/api/3/recordings`).
Per-species CSVs in this directory are produced by `Stage1_xc_fetch_metadata.py`.

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `id` | int | XC recording ID | `1072882` |
| `gen` | str | Genus | `Acridotheres` |
| `sp` | str | Species epithet | `tristis` |
| `ssp` | str | Subspecies (often blank) | `tristis` |
| `grp` | str | Taxonomic group | `birds` |
| `en` | str | English common name | `Common Myna` |
| `rec` | str | Recordist name | `Albert Lastukhin` |
| `cnt` | str | Country | `Uzbekistan` |
| `loc` | str | Locality description | `Kitab, Kitab District, Kashkadarya Province` |
| `lat` | float | Latitude (WGS84) | `39.1382` |
| `lon` | float | Longitude (WGS84) | `66.8735` |
| `alt` | str | Altitude in metres (sometimes blank) | `650` |
| `type` | str | Vocalisation type | `call`, `song`, `alarm call`, `flight call` |
| `sex` | str | Sex of individual (often blank) | `male`, `female` |
| `stage` | str | Life stage (often blank) | `adult`, `juvenile` |
| `method` | str | Recording method | `field recording` |
| `url` | str | XC page URL (protocol-relative) | `//xeno-canto.org/1072882` |
| `file` | str | Download URL. **Blank/null for blocked species** | `https://xeno-canto.org/1072882/download` |
| `file-name` | str | Original uploaded filename | `XC1072882-Acridotheres-tristis-140510_018-B.wav` |
| `lic` | str | Creative Commons licence URL | `//creativecommons.org/licenses/by-nc-nd/4.0/` |
| `q` | str | Quality rating | `A`, `B`, `C`, `D`, `E` |
| `length` | str | Duration as `M:SS` or `H:MM:SS` | `0:08`, `1:23`, `1:02:15` |
| `time` | str | Time of day recorded (`HH:MM`) | `07:00` |
| `date` | str | Date recorded (`YYYY-MM-DD`) | `2014-05-10` |
| `uploaded` | str | Date uploaded (`YYYY-MM-DD`) | `2026-01-16` |
| `also` | str | Other species heard (JSON list) | `["Pycnonotus goiavier"]` or `[]` |
| `rmk` | str | Recordist remarks (often blank) | `fighting` |
| `animal-seen` | str | Was the animal seen? | `yes`, `no`, `unknown` |
| `playback-used` | str | Was playback used? | `yes`, `no` |
| `temp` | float | Temperature in C (rarely populated) | |
| `regnr` | float | Registration number (rarely populated) | |
| `auto` | str | Automatic recording? | `yes`, `no` |
| `dvc` | str | Recording device (often blank) | `sound devices 702` |
| `mic` | str | Microphone (often blank) | `Telinga pro 9 mkII` |
| `smp` | int | Sample rate in Hz | `44100`, `48000` |
| `length_seconds` | float | Duration in seconds (computed from `length`) | `8.0` |

## Notes

- **Blocked species**: Oriental Magpie-Robin, Javan Myna, Black-naped Oriole have all metadata fields populated *except* `file` is null/blank. These cannot be downloaded.
- **`length_seconds`**: Added by Stage 1 during save. Parsed from the `length` field (`M:SS` -> seconds).
- **`sono`/`osci`**: Sonogram and oscillogram fields are dropped during save to reduce CSV size.
- **Quality ratings**: A (best) through E (worst). Assigned by XC community.
