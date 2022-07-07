## **Khmer Language Model for Handwritten Text Recognition on Historical Documents (v3.0)**

Preservation of historical documents is a critical responsibility that we cannot disregard, or they may vanish in a matter of days. A Study of **Khmer Palm Leaf Manuscripts Digitization** was adopted to provide public access to the Khmer Palm Leaf Manuscripts, or Khmer Sastra Sluek Rith, on the internet in order to contribute to the preservation of these priceless records that are vital to Cambodians and researchers.
**Khmer Handwritten Text Recognition on Historical Documents** is a part of the above research which focus on creating a model that has ability to correct Khmer misspelling words that are extracted from the Sluek Rith set.

### Project Structure

- `model.py` : mail model (encoder-decoder) to train and test in this study
- `utils.py` : where all functions beside the main model located
- `test.py` : a place where to test any functions or codes
- `data.py` : a file where we store the functions to generate data from all files in the folder `data/sleuk-rith`
- in the folder '_data_'
    - `sleuk_rith_lines.txt` : all sentences that generated from the text fils in the folder `sleuk-rith`. 
    - `sleuk_rith_words.txt` : all filtered words (removed duplicate words) that generated from the sentences in `sleuk_rith_lines.txt`
    - `SBBICkm_KH.txt` : the main dataset for our experiments which totally size of 75040 words.
    - `Khmer_Unicode_Table_U1780.pdf` : the description of each Khmer Unicode character.

### Environment Setup
- Python version = 3.7.10
- PyTorch version = 1.8.1
- Conda version = 4.11.0
- Dataset sources:
    - `SBBICkm_KH.txt` : https://sbbic.org/2010/07/29/sbbic-khmer-word-list/
    - Khmer Unicode Table: https://unicode.org/charts/PDF/U1780.pdf

### Define Khmer Characters
- Total Characters: 106 characters
    - Start Character: 1780 (ក)
    - End Character: 17e9 (៩)

### Experimental Results

The below tables tells you the experimental results with the dataset that we extracted from Sleuk-Rith.
You can find text files that are used to train in this work in the directory: `data/sleuk-rith`.

**Dataset**

**Source:** from Sleuk-Rith

**Total words (size):** _5258 words_

**Training size:** _3680 words_ = 70% of total words (they are randomly selected from the total dataset)

**Testing size:** _1578 words_ = 30% of total words (they are randomly selected from the total dataset)

**Experiment 1:** Adjust _**Hidden Size**_
- **Note:** Larger layer size, more time to train.

| Batch Size | _Hidden Size (Layers)_ | Epoch (Loop) | Learning rate | Accuracy (%) |   Figure    |
|:----------:|:----------------------:|:------------:|:-------------:|:------------:|:-----------:|
|    200     |          128           |     2000     |     0.001     |  **43.97**   | Figure_128  |
|    200     |          256           |     2000     |     0.001     |  **52.79**   | Figure_256  |
|    200     |          512           |     2000     |     0.001     |  **52.28**   | Figure_512  |
|    200     |          728           |     2000     |     0.001     |  **53.68**   | Figure_728  |
|    200     |          1024          |     2000     |     0.001     |  **00.00**   | Figure_1024 |
|    200     |          2048          |     2000     |     0.001     |  **53.87**   | Figure_1024 |

**Experiment 2:** Adjust _**Batch Size**_
- **Note:** Larger batch size, less time to train.

| _Batch Size_ | Hidden Size (Layers) | Epoch (Loop) | Learning rate | Accuracy (%) |   Figure   |
|:------------:|:--------------------:|:------------:|:-------------:|:------------:|:----------:|
|      50      |         256          |     2000     |     0.001     |  **53.93**   | Figure_50  |
|     100      |         256          |     2000     |     0.001     |  **51.33**   | Figure_100 |
|     200      |         256          |     2000     |     0.001     |  **52.79**   | Figure_256 |
|     250      |         256          |     2000     |     0.001     |  **52.66**   | Figure_250 |
|     350      |         256          |     2000     |     0.001     |  **46.70**   | Figure_350 |

**Experiment 3:** Adjust _**Learning Rate (lr)**_
- **Note:** Larger batch size, less time to train.

| Batch Size | Hidden Size (Layers) | Epoch (Loop) | _Learning rate_ | Accuracy (%) |    Figure     |
|:----------:|:--------------------:|:------------:|:---------------:|:------------:|:-------------:|
|    250     |         256          |     2000     |      0.01       |  **00.00**   |  Figure_0.01  |
|    250     |         256          |     2000     |      0.005      |  **50.06**   | Figure_0.005  |
|    250     |         256          |     2000     |      0.001      |  **52.66**   |  Figure_250   |
|    250     |         256          |     2000     |     0.0005      |  **50.25**   | Figure_0.0005 |
|    250     |         256          |     2000     |     0.0001      |  **21.61**   | Figure_0.0001 |

**Experiment 4:** Adjust _**Epoch (Training loop)**_
- **Note:** Larger batch size, less time to train.

| Batch Size | Hidden Size (Layers) | _Epoch (Loop)_ | Learning rate | Accuracy (%) |   Figure    |
|:----------:|:--------------------:|:--------------:|:-------------:|:------------:|:-----------:|
|    250     |         256          |      2000      |     0.001     |  **52.66**   | Figure_250  |
|    250     |         256          |      4000      |    0.0001     |  **42.21**   | Figure_4000 |
|    250     |         256          |      5000      |    0.0005     |  **57.86**   | Figure_5000 |

**Experiment 5:** Adjust _**Max Parameters**_
- **Note:** Larger batch size, less time to train.

| Batch Size | Hidden Size (Layers) | _Epoch (Loop)_ | Learning rate | Accuracy (%) |   Figure    |
|:----------:|:--------------------:|:--------------:|:-------------:|:------------:|:-----------:|
|    200     |         2048         |      5000      |     0.001     |  **54.06**   | Figure_exp1 |
|    250     |         728          |      5000      |    0.0005     |  **58.05**   | Figure_exp2 |
|    250     |         256          |      4500      |    0.0008     |  **52.34**   | Figure_exp3 |
|    250     |         728          |      4000      |    0.0005     |  **62.10**   | Figure_exp4 |
|    250     |         256          |      8000      |    0.0005     |  **77.11**   | Figure_exp5 |

**Experiment 6:** Adjust _**Current Words vs Sleuk-Rith Words**_
- **Note:** Larger batch size, less time to train.

| Batch Size | Hidden Size (Layers) | _Epoch (Loop)_ | Learning rate | Accuracy (%) (Current Words) | Accuracy (%) (Sleuk-Rith Words |
|:----------:|:--------------------:|:--------------:|:-------------:|:----------------------------:|:------------------------------:|
|    250     |         256          |     10000      |    0.0005     |          **79.32**           |           **78.38**            |

**Experiment 7:** Adjust _**Epoch**_
- **Note:** 100% for training, Randomly select 30% from all words to test.

| Batch Size | Hidden Size (Layers) | _Epoch (Loop)_ | Learning rate | Accuracy (%) |
|:----------:|:--------------------:|:--------------:|:-------------:|:------------:|
|    250     |         256          |      5000      |    0.0005     |  **73.37**   |
|    250     |         256          |      6000      |    0.0005     |  **77.11**   |
|    250     |         256          |      7000      |    0.0005     |  **78.31**   |
|    250     |         256          |      8000      |    0.0005     |  **77.36**   |
|    250     |         256          |      9000      |    0.0005     |  **77.87**   |
|    250     |         256          |     10000      |    0.0005     |  **76.79**   |

**Experiment 8:** Find Average Accuracy of the model
- **Note:** 100% for training, Randomly select 30% from all words to test.

|    Batch Size    | Hidden Size (Layers) | _Epoch (Loop)_ | Learning rate | Accuracy (%) |
|:----------------:|:--------------------:|:--------------:|:-------------:|:------------:|
|       250        |         256          |      7000      |    0.0005     |  **78.31**   |
|       250        |         256          |      7000      |    0.0005     |  **77.36**   |
|       250        |         256          |      7000      |    0.0005     |  **78.12**   |
|       250        |         256          |      7000      |    0.0005     |  **78.19**   |
|       250        |         256          |      7000      |    0.0005     |  **75.08**   |
|       250        |         256          |      7000      |    0.0005     |  **77.11**   |
|       250        |         256          |      7000      |    0.0005     |  **76.16**   |
|       250        |         256          |      7000      |    0.0005     |  **75.78**   |
|       250        |         256          |      7000      |    0.0005     |  **74.83**   |
|       250        |         256          |      7000      |    0.0005     |  **76.66**   |

**Average Accuracy: 76.76%**