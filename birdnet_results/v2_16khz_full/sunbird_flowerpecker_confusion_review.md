# Ornate Sunbird / Scarlet-backed Flowerpecker confusion — review list

BirdNET zero-shot (v2.0, 16kHz, full dataset). 128 Scarlet-backed Flowerpecker
clips misclassified as Ornate Sunbird, across 45 source recordings. Grouped by
recordist (highest-contributing first), then by source recording.

## Review conclusion (2026-11-19)

Checked all 9 Geoff Carey source recordings (XC943430, XC943437–943446,
accounting for 39/128 of the confused clips, the largest single contributor).
Recorded with a Telinga parabolic mic (gold-standard field equipment). Web
descriptors have been live on Xeno-Canto since 2019 with zero corrections or
ID disputes in the comments. Audio is clean (not noisy/contaminated).

**Conclusion: this is not an annotation-quality or recording-quality
artifact.** Ornate Sunbird and Scarlet-backed Flowerpecker calls are
genuinely hard to differentiate acoustically, in at least this direction
(Flowerpecker mistaken for Sunbird far more often than the reverse). This
should be reported as a real class-pair difficulty in the dataset paper's
discussion/limitations, not chased further as a labeling bug. Also
consistent with the uniformly low BirdNET confidence scores on these clips
(median 0.0027) — BirdNET itself is near-uncertain on them, not confidently
miscalling them.

## Geoff Carey — 39 confused clip(s)

- [XC943440](https://xeno-canto.org/943440) (China, grade A) — 10 clip(s)
  - `xc943440_0.wav` (onset 0ms, BirdNET conf 0.002)
  - `xc943440_9448.wav` (onset 9448ms, BirdNET conf 0.005)
  - `xc943440_28185.wav` (onset 28185ms, BirdNET conf 0.001)
  - `xc943440_47209.wav` (onset 47209ms, BirdNET conf 0.013)
  - `xc943440_55427.wav` (onset 55427ms, BirdNET conf 0.005)
  - `xc943440_57167.wav` (onset 57167ms, BirdNET conf 0.012)
  - `xc943440_65177.wav` (onset 65177ms, BirdNET conf 0.003)
  - `xc943440_67511.wav` (onset 67511ms, BirdNET conf 0.003)
  - `xc943440_75404.wav` (onset 75404ms, BirdNET conf 0.043)
  - `xc943440_85524.wav` (onset 85524ms, BirdNET conf 0.002)
- [XC943439](https://xeno-canto.org/943439) (China, grade A) — 8 clip(s)
  - `xc943439_0.wav` (onset 0ms, BirdNET conf 0.023)
  - `xc943439_3535.wav` (onset 3535ms, BirdNET conf 0.005)
  - `xc943439_8175.wav` (onset 8175ms, BirdNET conf 0.001)
  - `xc943439_12264.wav` (onset 12264ms, BirdNET conf 0.001)
  - `xc943439_16172.wav` (onset 16172ms, BirdNET conf 0.004)
  - `xc943439_27240.wav` (onset 27240ms, BirdNET conf 0.017)
  - `xc943439_33996.wav` (onset 33996ms, BirdNET conf 0.009)
  - `xc943439_36996.wav` (onset 36996ms, BirdNET conf 0.048)
- [XC943441](https://xeno-canto.org/943441) (China, grade A) — 5 clip(s)
  - `xc943441_0.wav` (onset 0ms, BirdNET conf 0.018)
  - `xc943441_9151.wav` (onset 9151ms, BirdNET conf 0.001)
  - `xc943441_15751.wav` (onset 15751ms, BirdNET conf 0.002)
  - `xc943441_27845.wav` (onset 27845ms, BirdNET conf 0.002)
  - `xc943441_43287.wav` (onset 43287ms, BirdNET conf 0.000)
- [XC943446](https://xeno-canto.org/943446) (China, grade A) — 4 clip(s)
  - `xc943446_0.wav` (onset 0ms, BirdNET conf 0.006)
  - `xc943446_6640.wav` (onset 6640ms, BirdNET conf 0.002)
  - `xc943446_12977.wav` (onset 12977ms, BirdNET conf 0.002)
  - `xc943446_16063.wav` (onset 16063ms, BirdNET conf 0.022)
- [XC943437](https://xeno-canto.org/943437) (China, grade A) — 4 clip(s)
  - `xc943437_0.wav` (onset 0ms, BirdNET conf 0.003)
  - `xc943437_3973.wav` (onset 3973ms, BirdNET conf 0.000)
  - `xc943437_7120.wav` (onset 7120ms, BirdNET conf 0.000)
  - `xc943437_17071.wav` (onset 17071ms, BirdNET conf 0.000)
- [XC943444](https://xeno-canto.org/943444) (China, grade A) — 2 clip(s)
  - `xc943444_11868.wav` (onset 11868ms, BirdNET conf 0.055)
  - `xc943444_18585.wav` (onset 18585ms, BirdNET conf 0.080)
- [XC943430](https://xeno-canto.org/943430) (China, grade A) — 2 clip(s)
  - `xc943430_0.wav` (onset 0ms, BirdNET conf 0.006)
  - `xc943430_26000.wav` (onset 26000ms, BirdNET conf 0.004)
- [XC943442](https://xeno-canto.org/943442) (China, grade A) — 2 clip(s)
  - `xc943442_7816.wav` (onset 7816ms, BirdNET conf 0.002)
  - `xc943442_26949.wav` (onset 26949ms, BirdNET conf 0.038)
- [XC943438](https://xeno-canto.org/943438) (China, grade A) — 1 clip(s)
  - `xc943438_53955.wav` (onset 53955ms, BirdNET conf 0.005)
- [XC943443](https://xeno-canto.org/943443) (China, grade A) — 1 clip(s)
  - `xc943443_23209.wav` (onset 23209ms, BirdNET conf 0.001)

## Okamoto Keita Sin — 11 confused clip(s)

- [XC756484](https://xeno-canto.org/756484) (Singapore, grade B) — 6 clip(s)
  - `xc756484_0.wav` (onset 0ms, BirdNET conf 0.000)
  - `xc756484_9565.wav` (onset 9565ms, BirdNET conf 0.003)
  - `xc756484_12679.wav` (onset 12679ms, BirdNET conf 0.002)
  - `xc756484_28544.wav` (onset 28544ms, BirdNET conf 0.001)
  - `xc756484_36261.wav` (onset 36261ms, BirdNET conf 0.002)
  - `xc756484_54684.wav` (onset 54684ms, BirdNET conf 0.002)
- [XC756531](https://xeno-canto.org/756531) (Singapore, grade B) — 4 clip(s)
  - `xc756531_0.wav` (onset 0ms, BirdNET conf 0.000)
  - `xc756531_593.wav` (onset 593ms, BirdNET conf 0.002)
  - `xc756531_16641.wav` (onset 16641ms, BirdNET conf 0.005)
  - `xc756531_31199.wav` (onset 31199ms, BirdNET conf 0.000)
- [XC575544](https://xeno-canto.org/575544) (Singapore, grade B) — 1 clip(s)
  - `xc575544_643.wav` (onset 643ms, BirdNET conf 0.049)

## Sam Hambly — 11 confused clip(s)

- [XC918259](https://xeno-canto.org/918259) (Thailand, grade C) — 8 clip(s)
  - `xc918259_0.wav` (onset 0ms, BirdNET conf 0.010)
  - `xc918259_17812.wav` (onset 17812ms, BirdNET conf 0.003)
  - `xc918259_27338.wav` (onset 27338ms, BirdNET conf 0.013)
  - `xc918259_37213.wav` (onset 37213ms, BirdNET conf 0.005)
  - `xc918259_44015.wav` (onset 44015ms, BirdNET conf 0.010)
  - `xc918259_54112.wav` (onset 54112ms, BirdNET conf 0.002)
  - `xc918259_69328.wav` (onset 69328ms, BirdNET conf 0.045)
  - `xc918259_75734.wav` (onset 75734ms, BirdNET conf 0.006)
- [XC859129](https://xeno-canto.org/859129) (Thailand, grade A) — 3 clip(s)
  - `xc859129_6562.wav` (onset 6562ms, BirdNET conf 0.010)
  - `xc859129_9960.wav` (onset 9960ms, BirdNET conf 0.014)
  - `xc859129_19115.wav` (onset 19115ms, BirdNET conf 0.012)

## Bram Piot — 7 confused clip(s)

- [XC858834](https://xeno-canto.org/858834) (Laos, grade B) — 6 clip(s)
  - `xc858834_13792.wav` (onset 13792ms, BirdNET conf 0.125)
  - `xc858834_21804.wav` (onset 21804ms, BirdNET conf 0.013)
  - `xc858834_34205.wav` (onset 34205ms, BirdNET conf 0.002)
  - `xc858834_39502.wav` (onset 39502ms, BirdNET conf 0.012)
  - `xc858834_45176.wav` (onset 45176ms, BirdNET conf 0.003)
  - `xc858834_45589.wav` (onset 45589ms, BirdNET conf 0.002)
- [XC897772](https://xeno-canto.org/897772) (Laos, grade B) — 1 clip(s)
  - `xc897772_15912.wav` (onset 15912ms, BirdNET conf 0.000)

## Greg Irving — 7 confused clip(s)

- [XC782969](https://xeno-canto.org/782969) (Thailand, grade B) — 2 clip(s)
  - `xc782969_13977.wav` (onset 13977ms, BirdNET conf 0.001)
  - `xc782969_19183.wav` (onset 19183ms, BirdNET conf 0.004)
- [XC439691](https://xeno-canto.org/439691) (Thailand, grade C) — 1 clip(s)
  - `xc439691_630.wav` (onset 630ms, BirdNET conf 0.017)
- [XC319166](https://xeno-canto.org/319166) (Thailand, grade B) — 1 clip(s)
  - `xc319166_8077.wav` (onset 8077ms, BirdNET conf 0.004)
- [XC782968](https://xeno-canto.org/782968) (Thailand, grade C) — 1 clip(s)
  - `xc782968_10816.wav` (onset 10816ms, BirdNET conf 0.000)
- [XC899172](https://xeno-canto.org/899172) (Thailand, grade B) — 1 clip(s)
  - `xc899172_29455.wav` (onset 29455ms, BirdNET conf 0.003)
- [XC863013](https://xeno-canto.org/863013) (Thailand, grade C) — 1 clip(s)
  - `xc863013_64279.wav` (onset 64279ms, BirdNET conf 0.003)

## Lars Lachmann — 7 confused clip(s)

- [XC446263](https://xeno-canto.org/446263) (China, grade D) — 7 clip(s)
  - `xc446263_8915.wav` (onset 8915ms, BirdNET conf 0.002)
  - `xc446263_16264.wav` (onset 16264ms, BirdNET conf 0.001)
  - `xc446263_28902.wav` (onset 28902ms, BirdNET conf 0.002)
  - `xc446263_33906.wav` (onset 33906ms, BirdNET conf 0.003)
  - `xc446263_36906.wav` (onset 36906ms, BirdNET conf 0.000)
  - `xc446263_43977.wav` (onset 43977ms, BirdNET conf 0.000)
  - `xc446263_50032.wav` (onset 50032ms, BirdNET conf 0.000)

## Aladdin — 5 confused clip(s)

- [XC820687](https://xeno-canto.org/820687) (Cambodia, grade C) — 5 clip(s)
  - `xc820687_3342.wav` (onset 3342ms, BirdNET conf 0.000)
  - `xc820687_15126.wav` (onset 15126ms, BirdNET conf 0.000)
  - `xc820687_38208.wav` (onset 38208ms, BirdNET conf 0.002)
  - `xc820687_43823.wav` (onset 43823ms, BirdNET conf 0.004)
  - `xc820687_48670.wav` (onset 48670ms, BirdNET conf 0.000)

## Yeo SB — 5 confused clip(s)

- [XC703372](https://xeno-canto.org/703372) (Singapore, grade C) — 5 clip(s)
  - `xc703372_3605.wav` (onset 3605ms, BirdNET conf 0.001)
  - `xc703372_14096.wav` (onset 14096ms, BirdNET conf 0.001)
  - `xc703372_19351.wav` (onset 19351ms, BirdNET conf 0.000)
  - `xc703372_28898.wav` (onset 28898ms, BirdNET conf 0.000)
  - `xc703372_35201.wav` (onset 35201ms, BirdNET conf 0.000)

## Kim Chuah Lim — 5 confused clip(s)

- [XC684083](https://xeno-canto.org/684083) (Singapore, grade C) — 5 clip(s)
  - `xc684083_0.wav` (onset 0ms, BirdNET conf 0.001)
  - `xc684083_3039.wav` (onset 3039ms, BirdNET conf 0.006)
  - `xc684083_7756.wav` (onset 7756ms, BirdNET conf 0.010)
  - `xc684083_13008.wav` (onset 13008ms, BirdNET conf 0.022)
  - `xc684083_16051.wav` (onset 16051ms, BirdNET conf 0.036)

## Jue Lei — 5 confused clip(s)

- [XC1084320](https://xeno-canto.org/1084320) (China, grade B) — 3 clip(s)
  - `xc1084320_0.wav` (onset 0ms, BirdNET conf 0.007)
  - `xc1084320_6573.wav` (onset 6573ms, BirdNET conf 0.004)
  - `xc1084320_37720.wav` (onset 37720ms, BirdNET conf 0.004)
- [XC1077194](https://xeno-canto.org/1077194) (China, grade B) — 2 clip(s)
  - `xc1077194_8858.wav` (onset 8858ms, BirdNET conf 0.000)
  - `xc1077194_13045.wav` (onset 13045ms, BirdNET conf 0.007)

## David Darrell-Lambert — 4 confused clip(s)

- [XC1004376](https://xeno-canto.org/1004376) (Thailand, grade B) — 3 clip(s)
  - `xc1004376_1016.wav` (onset 1016ms, BirdNET conf 0.007)
  - `xc1004376_26900.wav` (onset 26900ms, BirdNET conf 0.003)
  - `xc1004376_30823.wav` (onset 30823ms, BirdNET conf 0.008)
- [XC1006541](https://xeno-canto.org/1006541) (Thailand, grade B) — 1 clip(s)
  - `xc1006541_23772.wav` (onset 23772ms, BirdNET conf 0.014)

## Bernard BOUSQUET — 4 confused clip(s)

- [XC359283](https://xeno-canto.org/359283) (Thailand, grade B) — 4 clip(s)
  - `xc359283_0.wav` (onset 0ms, BirdNET conf 0.001)
  - `xc359283_5250.wav` (onset 5250ms, BirdNET conf 0.000)
  - `xc359283_8270.wav` (onset 8270ms, BirdNET conf 0.001)
  - `xc359283_11383.wav` (onset 11383ms, BirdNET conf 0.000)

## Jelle Scharringa — 3 confused clip(s)

- [XC652433](https://xeno-canto.org/652433) (Thailand, grade C) — 3 clip(s)
  - `xc652433_13840.wav` (onset 13840ms, BirdNET conf 0.000)
  - `xc652433_23231.wav` (onset 23231ms, BirdNET conf 0.003)
  - `xc652433_27196.wav` (onset 27196ms, BirdNET conf 0.001)

## Shuangqi Liu — 2 confused clip(s)

- [XC954652](https://xeno-canto.org/954652) (China, grade A) — 2 clip(s)
  - `xc954652_0.wav` (onset 0ms, BirdNET conf 0.000)
  - `xc954652_6187.wav` (onset 6187ms, BirdNET conf 0.000)

## Peter Ericsson — 2 confused clip(s)

- [XC961277](https://xeno-canto.org/961277) (Thailand, grade A) — 2 clip(s)
  - `xc961277_33249.wav` (onset 33249ms, BirdNET conf 0.009)
  - `xc961277_58850.wav` (onset 58850ms, BirdNET conf 0.001)

## Albert Lastukhin &amp; Yuri Glushchenko — 2 confused clip(s)

- [XC303863](https://xeno-canto.org/303863) (Vietnam, grade C) — 2 clip(s)
  - `xc303863_471.wav` (onset 471ms, BirdNET conf 0.020)
  - `xc303863_31303.wav` (onset 31303ms, BirdNET conf 0.004)

## Albert Lastukhin — 1 confused clip(s)

- [XC499465](https://xeno-canto.org/499465) (Vietnam, grade C) — 1 clip(s)
  - `xc499465_1248.wav` (onset 1248ms, BirdNET conf 0.002)

## GABRIEL LEITE — 1 confused clip(s)

- [XC730921](https://xeno-canto.org/730921) (Malaysia, grade C) — 1 clip(s)
  - `xc730921_19733.wav` (onset 19733ms, BirdNET conf 0.000)

## FREDERIC Lionel — 1 confused clip(s)

- [XC843799](https://xeno-canto.org/843799) (Thailand, grade B) — 1 clip(s)
  - `xc843799_0.wav` (onset 0ms, BirdNET conf 0.151)

## Jacob Saucier — 1 confused clip(s)

- [XC457337](https://xeno-canto.org/457337) (Malaysia, grade C) — 1 clip(s)
  - `xc457337_923.wav` (onset 923ms, BirdNET conf 0.001)

## Jukree Sisonmak — 1 confused clip(s)

- [XC1135792](https://xeno-canto.org/1135792) (Thailand, grade B) — 1 clip(s)
  - `xc1135792_9744.wav` (onset 9744ms, BirdNET conf 0.075)

## Mike Nelson — 1 confused clip(s)

- [XC192175](https://xeno-canto.org/192175) (Malaysia, grade C) — 1 clip(s)
  - `xc192175_2046.wav` (onset 2046ms, BirdNET conf 0.001)

## Muhammad Iqbal — 1 confused clip(s)

- [XC690449](https://xeno-canto.org/690449) (Indonesia, grade C) — 1 clip(s)
  - `xc690449_23308.wav` (onset 23308ms, BirdNET conf 0.024)

## Somkiat Pakapinyo (Chai) — 1 confused clip(s)

- [XC618079](https://xeno-canto.org/618079) (Thailand, grade B) — 1 clip(s)
  - `xc618079_0.wav` (onset 0ms, BirdNET conf 0.001)

## Viral joshi — 1 confused clip(s)

- [XC474674](https://xeno-canto.org/474674) (India, grade B) — 1 clip(s)
  - `xc474674_0.wav` (onset 0ms, BirdNET conf 0.001)
