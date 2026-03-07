# Project CSV Files

This folder contains the core metadata files for the MyGardenBird dataset curation pipeline.

## Files


### `target_species.csv`
Master list of 50 Malaysian bird species common in urban and peri-urban environments.

**Fields:**
- `Common name`: English common name
- `Scientific name`: Binomial nomenclature (genus species)
- `eBird code`: Six-character eBird taxonomy identifier
- `active`: Toggle for Stage 1 metadata download (yes/no)

**Usage:** Defines the species scope for metadata fetching (Stage 1). The `active` column controls which species are processed by the pipeline. Currently, 10 species are set to `active=yes` and constitute the MyGardenBird dataset. The remaining 40 species are candidates for future dataset expansion.

**Note:** All 50 species are commonly reported in the Malaysian Garden Birdwatch survey. Setting `active=yes` for additional species will trigger Stage 1 to download Xeno-canto metadata for those species.


### `recordings.csv`
Source recording metadata from Xeno-canto (1,123 unique recordings).

**Fields:**
- `source_id`: Xeno-canto recording ID (numeric, e.g., "1000132" for XC1000132)
- `species_common`: Common species name (e.g., "White-breasted Waterhen")
- `species_scientific`: Scientific name in binomial nomenclature (genus + species)
- `quality_grade`: Xeno-canto quality rating (A = highest, B = good, C = acceptable)
- `type_label`: Vocalization type (song, call, or other)
- `latitude`, `longitude`: Geographic coordinates (decimal degrees)
- `country`: Country of recording

**Usage:** Referenced by Stage 2 (downloading) and Stage 3-7 (segmentation). The `source_id` serves as the primary key linking recordings to extracted clips.

### `regional_ranking.csv`
Species selection criteria and regional abundance rankings.

**Fields:**
- `species_common`: Common species name
- `abundance_rank`: Regional abundance ranking
- `selection_criteria`: Reason for inclusion (abundance, distinctiveness, etc.)
- `habitat`: Primary habitat type (urban, peri-urban, forest-edge)

**Usage:** Documents the rationale for species selection in the paper.

## Data Source

All metadata is derived from [Xeno-canto](https://www.xeno-canto.org/), a citizen science bird sound repository. Species selection is based on the MY Garden Birdwatch survey.

## License Compliance

Source recordings carry various Creative Commons licenses:
- 62.8% CC BY-NC-SA (allows derivatives)
- 36.5% CC BY-NC-ND (restricts derivatives)
- 0.7% more permissive (CC BY-SA, CC BY, CC0)

MyGardenBird is released under **CC BY-NC-SA 4.0** to comply with upstream licensing constraints.
