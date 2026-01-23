# seabird
## Curation of Southeast Asia bird sound dataset

### SEA-Bird Dataset Statistics

| No. | Common Name                | Scientific Name                | Train | Val | Test | 16 kHz Clips | 44.1 kHz Clips | 16 kHz XC (Train/Val/Test) | Total Source XC Files |
|-----|----------------------------|--------------------------------|-------|-----|------|-------|----------------|---------------------|----------|
| 1   | Asian Koel                 | *Eudynamys scolopaceus*        | 450   | 60  | 90   | 600   | 576            | 86 / 13 / 32        | 131      |
| 2   | Collared Kingfisher        | *Todiramphus chloris*          | 450   | 60  | 90   | 600   | 571            | 73 / 18 / 26        | 117      |
| 3   | Common Iora                | *Aegithina tiphia*             | 450   | 60  | 90   | 600   | 598            | 75 / 9 / 21         | 105      |
| 4   | Common Myna                | *Acridotheres tristis*         | 450   | 60  | 90   | 600   | 592            | 81 / 10 / 32        | 123      |
| 5   | Common Tailorbird          | *Orthotomus sutorius*          | 450   | 60  | 90   | 600   | 577            | 65 / 10 / 21        | 96       |
| 6   | Large-tailed Nightjar      | *Caprimulgus macrurus*         | 450   | 60  | 90   | 600   | 581            | 56 / 11 / 16        | 83       |
| 7   | Olive-backed Sunbird       | *Cinnyris jugularis*           | 450   | 60  | 90   | 600   | 576            | 68 / 10 / 17        | 95       |
| 8   | Spotted Dove               | *Spilopelia chinensis*         | 450   | 60  | 90   | 600   | 599            | 68 / 9 / 18         | 95       |
| 9   | White-throated Kingfisher  | *Halcyon smyrnensis*           | 450   | 60  | 90   | 600   | 570            | 81 / 17 / 25        | 123      |
| 10  | Zebra Dove                 | *Geopelia striata*             | 450   | 60  | 90   | 600   | 581            | 69 / 10 / 27        | 106      |
|     | **Total**                  |                                | **4500** | **600** | **900** | **6000** | **5,821**     | **722 / 117 / 235** | **1,074** |

✅ The values in **Sources (T/V/Test)** add up to the **XC Files** column for every species and the overall total:

- Example: Asian Koel → 86 + 13 + 32 = **131** ✓
- Overall: 722 + 117 + 235 = **1,074** ✓


### Validation by Pretrained CNNs

**Test Accuracy (%) by Model and Feature Type**

| Model          | Mel    | STFT   | MFCC   |
|:---------------|:------:|:------:|:------:|
| EfficientNetB0 | 88.89  | 91.89  | 87.11  |
| MobileNetV3S   | 82.11  | 50.33 | 66.78  |
| ResNet50       | 89.78  | 88.00  | 86.89  |
| VGG16          | 85.00  | 88.44  | 85.89  |
| **Average**    | **86.45** | **79.67** | **81.67** |

