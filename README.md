# STRE-Former

## Datasets

- **IEMOCAP** : approximately 12 hours of audiovisual recordings from 10 actors, with 4-class and 6-class emotion configurations.
  Available at: https://sail.usc.edu/iemocap/

- **MELD** : approximately 13,000 utterances from the TV series *Friends*, with 7 emotion categories.
  Available at: https://affective-meld.github.io/

## Setup

Install the dependencies in `req.txt`. Place [wav2vec2-base-960h](https://huggingface.co/facebook/wav2vec2-base-960h) under `features_extraction/pretrained_model/wav2vec2-base-960h/`.

Set dataset paths in `features_extraction/config.py` and feature options in `features_extraction/tools.py`.

## Feature Extraction

```bash
cd features_extraction
python extract_features.py                    # IEMOCAP
python extract_features.py --dataset "MELD"   # MELD
cd ..
```

## Training

Set paths, dataset and model options in `config.py`, then run:

```bash
python main.py
```

## Checkpoints

Weights and training records: [Hugging Face](https://huggingface.co/STRE-Former/STRE-Former).