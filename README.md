# CS3713 - Image Processing Labs

This repository contains my lab work for the CS3713 Image Processing module. It focuses on:

- Point operations on grayscale images
- Linear filtering using custom 5x5 kernels
- Visual and quantitative comparison of processed outputs

## Repository Structure

```text
Image_Processing_Labs/
|- README.md
|- Lab 1 - Point Operations/
|  |- 210210G_Code.py
|  |- 210210G_SrcImage.jpg
|  |- 210210G_SubPlot.png
|  \- 210210G_OPImage_0.jpg ... 210210G_OPImage_5.jpg
\- Lab 2 - Linear Filter/
	|- 210210G.py
	|- road10.png
	|- original.jpg
	\- filter_A.jpg ... filter_D.jpg
```

## Lab 1 - Point Operations

Script: `Lab 1 - Point Operations/210210G_Code.py`

### What this lab does

1. Loads a color image and converts it to RGB.
1. Converts RGB to grayscale using:

   `Gray = 0.299R + 0.587G + 0.114B`

1. Applies several point operations:
   - Negative transformation
   - Brightness increase by 20 percent
   - Contrast reduction to the range [125, 175]
   - 4 bpp quantization
   - Horizontal mirroring
1. Saves each result as a separate image and also visualizes all results in one subplot figure.

### Output Preview

![Lab 1 Combined Output](<Lab 1 - Point Operations/210210G_SubPlot.png>)

| Grayscale                                                      | Negative                                                      | Brightened                                                      |
| -------------------------------------------------------------- | ------------------------------------------------------------- | --------------------------------------------------------------- |
| ![Grayscale](<Lab 1 - Point Operations/210210G_OPImage_0.jpg>) | ![Negative](<Lab 1 - Point Operations/210210G_OPImage_1.jpg>) | ![Brightened](<Lab 1 - Point Operations/210210G_OPImage_2.jpg>) |

| Contrast Reduced                                                      | 4 bpp                                                      | Mirror                                                      |
| --------------------------------------------------------------------- | ---------------------------------------------------------- | ----------------------------------------------------------- |
| ![Contrast Reduced](<Lab 1 - Point Operations/210210G_OPImage_3.jpg>) | ![4 bpp](<Lab 1 - Point Operations/210210G_OPImage_4.jpg>) | ![Mirror](<Lab 1 - Point Operations/210210G_OPImage_5.jpg>) |

## Lab 2 - Linear Filter

Script: `Lab 2 - Linear Filter/210210G.py`

### What this lab does

1. Loads a road image and converts it to grayscale.
1. Performs contrast enhancement using min-max stretching.
1. Defines four custom 5x5 filters (A, B, C, D).
1. Normalizes each filter and applies convolution manually (without high-level filtering APIs).
1. Saves each filtered output image.
1. Computes RMS difference between original enhanced image and each filtered output.

### Filter Summary

- Filter A: edge/detail enhancing style kernel
- Filter B: weighted smoothing (Gaussian-like blur)
- Filter C: uniform averaging blur
- Filter D: stronger edge/detail enhancement (compared to A)

### Output Preview

| Original (Contrast Enhanced)                      | Filter A                                          | Filter B                                          |
| ------------------------------------------------- | ------------------------------------------------- | ------------------------------------------------- |
| ![Original](<Lab 2 - Linear Filter/original.jpg>) | ![Filter A](<Lab 2 - Linear Filter/filter_A.jpg>) | ![Filter B](<Lab 2 - Linear Filter/filter_B.jpg>) |

| Filter C                                          | Filter D                                          |
| ------------------------------------------------- | ------------------------------------------------- |
| ![Filter C](<Lab 2 - Linear Filter/filter_C.jpg>) | ![Filter D](<Lab 2 - Linear Filter/filter_D.jpg>) |

## How to Run

Install dependencies:

```bash
pip install numpy opencv-python matplotlib
```

Run Lab 1:

```bash
cd "Lab 1 - Point Operations"
python 210210G_Code.py
```

Run Lab 2:

```bash
cd "Lab 2 - Linear Filter"
python 210210G.py
```

## Notes

- Keep the source images (`210210G_SrcImage.jpg`, `road10.png`) in the same folders as their scripts.
- Lab 2 prints RMS values to the console for quantitative comparison.
- Output images are saved in each respective lab folder.
