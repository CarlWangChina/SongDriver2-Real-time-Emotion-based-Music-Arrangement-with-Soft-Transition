# SongDriver2 code

- config_m2c.yaml: settings for chord generator.
- config_n2m.yaml: settings for melody generator.
- config_songdriver.yaml: settings for SongDriver2.
- config.py: the config class to load settings.
- get_objective_metric.py: pipeline to calculate objective metrics of model.
- metric.py: the code to compute objective metrics.
- train.py: the training pipeline of SongDriver2.
- utils.py: some supportive functions to read datas.

## data
- data_preprocess.py: the code to process datasets used by SongDriver2.
- musicDatasetFast/musicDatasetEnhance: the code to load datas from downsample or w/o downsampling datas.
- strConverter.py: tools to transform string to array.

## models
- emo_model.py: emotion recognition model based on an MLP.
- music_feature_extractor.py: the class to extract music theory features from the sequence of melody, chord and key.
- songdriver2.py: the implemented SongDriver2 model.
- transformer.py: the implemented Transformer.

