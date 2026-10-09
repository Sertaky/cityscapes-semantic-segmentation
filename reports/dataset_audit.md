# Cityscapes dataset audit

Read-only audit of the Kaggle mirror `electraawais/cityscape-dataset`, version 2, copied to this project's ignored `leftImg8bit/` and `gtFine/` directories. Counts below were computed from the local files, not assumed from Cityscapes conventions. No training split or model artifact was created.

## Dataset and cache safety

| Location | Files | Bytes |
|---|---:|---:|
| `leftImg8bit/` on G: | 5,000 | 11,591,013,039 |
| `gtFine/` on G: | 20,000 | 808,625,557 |

Both G: trees matched their previously verified file counts and bytes, and a sample PNG from each was readable before deleting only `C:\Users\A.Rashad\.cache\kagglehub\datasets\electraawais\cityscape-dataset\versions\2`. That directory held 12,399,642,484 bytes and is absent after deletion. C: free space increased by 12,459,761,664 bytes during the operation; the free-space delta may include concurrent system activity. The remaining KaggleHub cache was not removed.

## Official split structure

| Split | Images | Image cities | Annotation files | Annotation cities | Annotation types |
|---|---:|---:|---:|---:|---|
| train | 2,975 | 18 | 11,900 | 18 | color.png: 2,975, instanceIds.png: 2,975, labelIds.png: 2,975, polygons.json: 2,975 |
| val | 500 | 3 | 2,000 | 3 | color.png: 500, instanceIds.png: 500, labelIds.png: 500, polygons.json: 500 |
| test | 1,525 | 6 | 6,100 | 6 | color.png: 1,525, instanceIds.png: 1,525, labelIds.png: 1,525, polygons.json: 1,525 |

The image and annotation city directories match within each split:

- **train (18):** aachen, bochum, bremen, cologne, darmstadt, dusseldorf, erfurt, hamburg, hanover, jena, krefeld, monchengladbach, strasbourg, stuttgart, tubingen, ulm, weimar, zurich.
- **val (3):** frankfurt, lindau, munster.
- **test (6):** berlin, bielefeld, bonn, leverkusen, mainz, munich.

## Pairing, dimensions, and split integrity

Every train and validation image has the four matching annotation files (`labelIds.png`, `instanceIds.png`, `color.png`, `polygons.json`). No missing pairs, duplicate stems or filenames, malformed names, unexpected files, or mismatched city directories were found. All polygon JSON files parsed and reported 2048×1024 metadata.

| Split | Image size / mode | `labelIds` size / mode | `instanceIds` mode | `color` mode |
|---|---|---|---|---|
| train | 2048×1024 RGB (2,975) | 2048×1024 L (2,975) | I;16 | RGBA |
| val | 2048×1024 RGB (500) | 2048×1024 L (500) | I;16 | RGBA |
| test | 2048×1024 RGB (1,525) | 2048×1024 L (1,525) | I;16 | RGBA |

All 5,000 image PNGs and all 15,000 annotation PNGs had 2048×1024 metadata; all `labelIds` masks were decoded for pixel counts. No image/mask dimension mismatch was found. Image contents were SHA-256 hashed, although the RGB PNG pixel streams were not fully decoded in this audit.

No duplicate image filenames or stems, image-content hashes, or cities overlap across official train/val/test splits. No malformed or missing city directories were found.

## Official raw-ID mapping

The project mapping exactly matches `cityscapesscripts.helpers.labels`: train IDs 0–18 use the evaluated 19 classes; every official ignored raw ID and the 255 sentinel map to ignore index 255. Unknown raw IDs raise `ValueError` under the project contract (checked with -2, 34, and 256). No unknown or unmapped raw ID occurred in train or validation masks.

| trainId | Class | Raw labelId | Category | Ignore in evaluation |
|---:|---|---:|---|---|
| 0 | road | 7 | flat | false |
| 1 | sidewalk | 8 | flat | false |
| 2 | building | 11 | construction | false |
| 3 | wall | 12 | construction | false |
| 4 | fence | 13 | construction | false |
| 5 | pole | 17 | object | false |
| 6 | traffic light | 19 | object | false |
| 7 | traffic sign | 20 | object | false |
| 8 | vegetation | 21 | nature | false |
| 9 | terrain | 22 | nature | false |
| 10 | sky | 23 | sky | false |
| 11 | person | 24 | human | false |
| 12 | rider | 25 | human | false |
| 13 | car | 26 | vehicle | false |
| 14 | truck | 27 | vehicle | false |
| 15 | bus | 28 | vehicle | false |
| 16 | train | 31 | vehicle | false |
| 17 | motorcycle | 32 | vehicle | false |
| 18 | bicycle | 33 | vehicle | false |

## Raw label IDs and ignore pixels

All percentages in the raw-ID table use **all pixels in that official split** as the denominator. Train contains raw IDs 0–33; validation contains the same IDs except 16. Official raw ID -1 and sentinel 255 do not occur in these PNG masks.

| Raw ID | Official name | Target trainId | Train pixels | Train % | Val pixels | Val % |
|---:|---|---:|---:|---:|---:|---:|
| 0 | unlabeled | 255 | 704,950 | 0.011299 | 375,854 | 0.035844 |
| 1 | ego vehicle | 255 | 286,002,726 | 4.584092 | 51,330,054 | 4.895215 |
| 2 | rectification border | 255 | 81,359,604 | 1.304043 | 19,784,166 | 1.886765 |
| 3 | out of roi | 255 | 94,111,150 | 1.508427 | 15,817,000 | 1.508427 |
| 4 | static | 255 | 83,752,079 | 1.342390 | 15,650,152 | 1.492515 |
| 5 | dynamic | 255 | 17,818,704 | 0.285601 | 4,449,872 | 0.424373 |
| 6 | ground | 255 | 75,629,728 | 1.212204 | 18,676,902 | 1.781168 |
| 7 | road | 0 | 2,036,416,525 | 32.639969 | 345,264,442 | 32.926983 |
| 8 | sidewalk | 1 | 336,090,793 | 5.386910 | 49,568,652 | 4.727235 |
| 9 | parking | 255 | 39,065,130 | 0.626141 | 4,151,398 | 0.395908 |
| 10 | rail track | 255 | 11,239,214 | 0.180144 | 638,410 | 0.060884 |
| 11 | building | 2 | 1,260,636,120 | 20.205652 | 201,005,428 | 19.169371 |
| 12 | wall | 3 | 36,199,498 | 0.580211 | 6,718,315 | 0.640708 |
| 13 | fence | 4 | 48,454,166 | 0.776630 | 7,521,741 | 0.717329 |
| 14 | guard rail | 255 | 547,202 | 0.008771 | 38,838 | 0.003704 |
| 15 | bridge | 255 | 17,860,177 | 0.286265 | 312,193 | 0.029773 |
| 16 | tunnel | 255 | 3,362,825 | 0.053900 | 0 | 0.000000 |
| 17 | pole | 5 | 67,789,506 | 1.086540 | 13,565,658 | 1.293722 |
| 18 | polegroup | 255 | 499,872 | 0.008012 | 78,175 | 0.007455 |
| 19 | traffic light | 6 | 11,477,088 | 0.183956 | 1,808,393 | 0.172462 |
| 20 | traffic sign | 7 | 30,448,193 | 0.488028 | 6,098,373 | 0.581586 |
| 21 | vegetation | 8 | 879,783,988 | 14.101301 | 158,868,008 | 15.150834 |
| 22 | terrain | 9 | 63,949,536 | 1.024992 | 7,625,026 | 0.727179 |
| 23 | sky | 10 | 221,979,646 | 3.557921 | 30,765,347 | 2.934012 |
| 24 | person | 11 | 67,326,424 | 1.079117 | 11,913,424 | 1.136153 |
| 25 | rider | 12 | 7,463,162 | 0.119621 | 1,975,596 | 0.188408 |
| 26 | car | 13 | 386,328,286 | 6.192124 | 59,731,217 | 5.696413 |
| 27 | truck | 14 | 14,772,328 | 0.236773 | 2,760,211 | 0.263234 |
| 28 | bus | 15 | 12,990,290 | 0.208210 | 3,563,120 | 0.339806 |
| 29 | caravan | 255 | 2,493,375 | 0.039964 | 53,411 | 0.005094 |
| 30 | trailer | 255 | 1,300,575 | 0.020846 | 201,086 | 0.019177 |
| 31 | train | 16 | 12,863,955 | 0.206185 | 1,031,648 | 0.098386 |
| 32 | motorcycle | 17 | 5,449,152 | 0.087340 | 729,415 | 0.069562 |
| 33 | bicycle | 18 | 22,861,233 | 0.366423 | 6,504,475 | 0.620315 |

| Split | Total pixels | Ignore pixels after mapping | Ignore % |
|---|---:|---:|---:|
| train | 6,239,027,200 | 715,747,311 | 11.472098 |
| val | 1,048,576,000 | 131,557,511 | 12.546302 |

## Per-class pixel distribution

The percentages below use **all pixels**, including ignored pixels, as the denominator. Ranks are within each split, most frequent first. The CSV also includes percentages among non-ignored pixels. No class weights were computed.

| ID | Class | Train pixels | Train % | Train rank | Val pixels | Val % | Val rank |
|---:|---|---:|---:|---:|---:|---:|
| 0 | road | 2,036,416,525 | 32.639969 | 1 | 345,264,442 | 32.926983 | 1 |
| 1 | sidewalk | 336,090,793 | 5.386910 | 5 | 49,568,652 | 4.727235 | 5 |
| 2 | building | 1,260,636,120 | 20.205652 | 2 | 201,005,428 | 19.169371 | 2 |
| 3 | wall | 36,199,498 | 0.580211 | 11 | 6,718,315 | 0.640708 | 11 |
| 4 | fence | 48,454,166 | 0.776630 | 10 | 7,521,741 | 0.717329 | 10 |
| 5 | pole | 67,789,506 | 1.086540 | 7 | 13,565,658 | 1.293722 | 7 |
| 6 | traffic light | 11,477,088 | 0.183956 | 17 | 1,808,393 | 0.172462 | 17 |
| 7 | traffic sign | 30,448,193 | 0.488028 | 12 | 6,098,373 | 0.581586 | 13 |
| 8 | vegetation | 879,783,988 | 14.101301 | 3 | 158,868,008 | 15.150834 | 3 |
| 9 | terrain | 63,949,536 | 1.024992 | 9 | 7,625,026 | 0.727179 | 9 |
| 10 | sky | 221,979,646 | 3.557921 | 6 | 30,765,347 | 2.934012 | 6 |
| 11 | person | 67,326,424 | 1.079117 | 8 | 11,913,424 | 1.136153 | 8 |
| 12 | rider | 7,463,162 | 0.119621 | 18 | 1,975,596 | 0.188408 | 16 |
| 13 | car | 386,328,286 | 6.192124 | 4 | 59,731,217 | 5.696413 | 4 |
| 14 | truck | 14,772,328 | 0.236773 | 14 | 2,760,211 | 0.263234 | 15 |
| 15 | bus | 12,990,290 | 0.208210 | 15 | 3,563,120 | 0.339806 | 14 |
| 16 | train | 12,863,955 | 0.206185 | 16 | 1,031,648 | 0.098386 | 18 |
| 17 | motorcycle | 5,449,152 | 0.087340 | 19 | 729,415 | 0.069562 | 19 |
| 18 | bicycle | 22,861,233 | 0.366423 | 13 | 6,504,475 | 0.620315 | 12 |

## Official test annotation finding

**Conclusion B: only non-evaluation/convenience labels appear present.** All 1,525 test `labelIds.png` masks exist, but their decoded pixels contain only raw IDs 0–3. All four IDs map to ignore index 255, leaving **zero** pixels in train IDs 0–18. Test polygon JSON contains 4,966 objects total: 1,525 `ego vehicle`, 1,525 `out of roi`, and 1,916 `rectification border`; it contains no semantic class objects. Train and validation masks, in contrast, contain evaluated class IDs. These local test files must not be used for semantic evaluation or model selection.

| Test raw ID | Official name | Pixels | % of test pixels | Target |
|---:|---|---:|---:|---:|
| 0 | unlabeled | 2,971,132,460 | 92.901401 | 255 |
| 1 | ego vehicle | 143,289,548 | 4.480379 | 255 |
| 2 | rectification border | 35,492,942 | 1.109794 | 255 |
| 3 | out of roi | 48,241,850 | 1.508427 | 255 |

## Official-train city analysis

The [city class distribution CSV](city_class_distribution.csv) gives exact pixels and percentages for every one of the 19 classes in each of the 18 official-train cities. This table highlights human and vehicle class coverage; counts are pixels, not instances.

| City | Images | Ignore % | Person | Rider | Car | Truck | Bus | Train | Motorcycle | Bicycle |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| aachen | 174 | 8.707922 | 3,665,555 | 181,325 | 23,746,182 | 132,838 | 836,675 | 0 | 444,007 | 1,008,273 |
| bochum | 96 | 14.567167 | 350,687 | 37,769 | 10,965,597 | 315,092 | 46,129 | 18,131 | 21,386 | 64,575 |
| bremen | 316 | 9.531427 | 2,685,866 | 786,605 | 44,897,611 | 802,000 | 1,347,404 | 2,573,772 | 283,577 | 3,153,881 |
| cologne | 154 | 9.306226 | 4,709,111 | 798,529 | 27,066,599 | 143,785 | 1,258,756 | 626,570 | 334,042 | 2,285,622 |
| darmstadt | 85 | 12.194214 | 2,917,975 | 270,282 | 11,405,898 | 2,031,244 | 328,975 | 790,273 | 249,603 | 287,406 |
| dusseldorf | 221 | 8.925456 | 1,667,227 | 125,691 | 35,208,036 | 282,942 | 440,776 | 953,517 | 286,140 | 1,156,223 |
| erfurt | 109 | 8.754377 | 1,715,780 | 185,005 | 12,484,901 | 463,183 | 285,413 | 1,448,570 | 51,458 | 375,720 |
| hamburg | 248 | 16.231970 | 13,428,704 | 1,013,372 | 30,665,789 | 2,280,626 | 4,286,027 | 24,738 | 387,575 | 3,516,590 |
| hanover | 196 | 15.772268 | 4,446,835 | 789,678 | 25,976,797 | 2,367,972 | 479,890 | 24,672 | 405,394 | 2,548,035 |
| jena | 119 | 10.484380 | 3,385,234 | 380,760 | 12,144,075 | 251,901 | 486,123 | 689,249 | 100,252 | 880,632 |
| krefeld | 99 | 13.715670 | 1,215,807 | 213,002 | 12,381,675 | 82,594 | 492,649 | 392,852 | 41,107 | 269,480 |
| monchengladbach | 94 | 13.993260 | 830,136 | 149,084 | 9,985,236 | 164,961 | 173,809 | 0 | 52,474 | 227,470 |
| strasbourg | 365 | 15.371838 | 10,328,888 | 586,465 | 38,728,720 | 195,504 | 503,318 | 1,542,634 | 683,890 | 3,806,505 |
| stuttgart | 196 | 8.834667 | 7,001,946 | 334,724 | 35,715,173 | 3,758,137 | 344,032 | 566,721 | 725,025 | 571,091 |
| tubingen | 144 | 9.074956 | 2,650,375 | 507,366 | 14,423,119 | 99,499 | 196,719 | 0 | 297,330 | 1,071,725 |
| ulm | 95 | 11.220191 | 1,074,215 | 249,809 | 10,060,070 | 1,116,627 | 547,473 | 65,899 | 494,969 | 367,525 |
| weimar | 142 | 7.608562 | 2,397,647 | 294,381 | 15,460,124 | 31,577 | 597,880 | 0 | 85,008 | 522,996 |
| zurich | 122 | 7.986218 | 2,854,436 | 559,315 | 15,012,684 | 251,846 | 338,242 | 3,146,357 | 505,915 | 747,484 |

Every city has person, rider, car, truck, bus, motorcycle, and bicycle pixels; train pixels are absent from aachen, monchengladbach, tubingen, and weimar. All 19 classes are present in official train overall.

### Recommended development validation cities

Use **jena (119 images), krefeld (99), and ulm (95)** as a deterministic city-held-out development validation set: **313/2,975 images (10.52%)**. Leave the other 15 official-train cities for development training. All 19 classes and all eight human/vehicle classes above occur in the proposed validation cities, including train, bus, truck, rider, motorcycle, and bicycle. Among three-city combinations with roughly 270–335 images, this combination had one of the closest per-class pixel distributions to official train. This is a development split recommendation only; no manifest was written.

Keep official `val` untouched for the final held-out labeled evaluation. Do not use official `test` for local tuning. For a 4 GB GPU, train on downscaled/cropped samples later; that choice does not change the city separation recommended here.

## Audit scope and files

This audit read image and annotation metadata, decoded every `labelIds` mask, parsed every polygon JSON, and hashed every image file. It did not modify raw files, compute class weights, create a development split, train a model, or run a model evaluation.

- `class_pixel_frequencies.csv`: train/val class counts, percentages, and ranks.
- `city_class_distribution.csv`: per-city counts and percentages for all 19 classes.
