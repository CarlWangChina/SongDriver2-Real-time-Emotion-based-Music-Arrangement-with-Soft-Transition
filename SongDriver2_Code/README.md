# SongDriver2 code

- config_m2c.yaml: settings for chord generator.
- config_n2m.yaml: settings for melody generator.
- config_songdriver.yaml: settings for SongDriver2.
- config.py: the config class to load settings.
- get_objective_metric.py: pipeline to calculate objective metrics of model.
- metric.py: the code to compute objective metrics.
- train.py: the training pipeline of SongDriver2.
- utils.py: some supportive functions to read datas.

## Models
- emo_model.py: emotion recognition model based on an MLP.
- music_feature_extractor.py: the class to extract music theory features from the sequence of melody, chord and key.
- songdriver2.py: the implemented SongDriver2 model.
- transformer.py: the implemented Transformer.

## Data
- data_preprocess.py: the code to process datasets used by SongDriver2.
- musicDatasetFast/musicDatasetEnhance: the code to load datas from downsample or w/o downsampling datas.
- strConverter.py: tools to transform string to array.

## Obtain training data
- This paper does not make any innovative points or contribution points in terms of dataset.
- Therefore, you can refer to the following open source innovation / contribution work of other researchers to get the training data by yourself.
- Chapter 4 of the Paper gives the source, citation and download links of the dataset, and we use open source emotion labeling datasets from other researchers.
- Chapter 8 of the Supplementary gives the detailed step-by-step methods for processing the dataset and the data representation, and we used other researchers' open-source audio-to-midi methods such as Onsets & Frames, Harvest method, etc. to obtain the training data.
- https://doi.org/10.48550/arXiv.2305.08029 The Paper and Supplementary spend nearly 2 pages to explain the training data acquisition, which is very detailed, thus other researchers can follow these steps to obtain their own training data and be more free to take more suitable data representation for their own models, such as midi, musicxml, REMI, etc.

