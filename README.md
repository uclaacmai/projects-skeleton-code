# ACM AI SPRING 2021 Project Skeleton Code

## Setup

1. Create a new conda environment.

2. Install PyTorch.

3. As you work on the project, you will end up installing many more packages.

## Running the Skeleton Code

### Running the Code Locally

After activating your conda environment, run the following command:

```
python main.py
```

### Running the Code on Google Colab

[This notebook](https://colab.research.google.com/drive/1zkxwgDItHv92iD07M8o8FlfJ0ZxLUDly?usp=sharing) will walk you through setting the skeleton code up on Google Colab.

## Downloading the Dataset From Kaggle

### Method 1: Downloading from kaggle.com

1. Go to [kaggle.com](kaggle.com) and create an account.

2. Join either the [Cassava leaf](https://www.kaggle.com/c/cassava-leaf-disease-classification) or [Humpback whale](https://www.kaggle.com/c/humpback-whale-identification) competition.

3. In the data tab, you should be able to download the data as a zip file.

### Method 2: Downloading from the Kaggle API

1. Install the Kaggle API:

   ```
   pip install kaggle
   ```

   If you're on Mac or Linux, you may have to run:

   ```
   pip install --user kaggle
   ```

2. Copy the `kaggle.json` file to the location `~/.kaggle/kaggle.json` (or `C:\Users\<Windows-username>\.kaggle\kaggle.json` if you are on Windows).

3. Download the zipped dataset.

   ```
   # Use humpback-whale-identification for 🐋 dataset
   kaggle competitions download -c cassava-leaf-disease-classification
   ```
