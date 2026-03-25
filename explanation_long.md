# Pipeline Explanation

## Overall Goal

The goal was to build a set of comparable emotion-classification pipelines on IEMOCAP, while staying close to the project's existing modeling logic:

- one audio-only pipeline
- one text pipeline based on ASR + text embeddings
- one text pipeline based on the dataset's provided transcripts
- one fusion pipeline combining audio and text information

The main principle was to keep the pipelines comparable, so differences in results mostly reflect the information source rather than a completely different training setup.

## How The IEMOCAP Pipelines Were Designed

The IEMOCAP setup was adapted to the dataset rather than copied blindly from the older datasets.

The main choices were:

- Use the dataset's main emotion label as the target.
- Remove the `other` label by default, because it is not a clean emotion category for a focused classifier.
- Filter out very small classes, so the task stays learnable and the metrics are less dominated by extremely rare labels.
- Keep the same general classifier family across all pipelines, so audio, text, and fusion can be compared fairly.

The dataset is harder than something like RAVDESS because it is more natural, more imbalanced, and more speaker- and session-dependent. Because of that, the evaluation protocol matters a lot.

## Train / Validation / Test Regime

The most important choice was to avoid a naive random split across all utterances.

Instead:

- one full session is held out for test
- validation is created only from the remaining training sessions
- the final test session is kept untouched during model selection

This is healthier because it reduces leakage between training and evaluation. It makes the numbers harder, but more trustworthy.

The validation split is then used for:

- monitoring model quality during training
- early stopping
- choosing the best epoch

The test set is used only once at the end for the final reported result.

## Early Stopping

Early stopping was added because these models tend to keep fitting the training data after the useful generalization peak.

The behavior you observed is typical:

- training loss keeps improving
- validation or test loss improves at first, then flattens or worsens
- accuracy may still move a little even while loss gets worse

So the training protocol was changed to:

- allow enough epochs for improvement to happen
- monitor validation loss
- stop when validation loss no longer improves for a fixed patience window
- restore the best model, not the last model

This is a more defensible regime than training for a fixed number of epochs and keeping the final weights.

## Healthy Engineering Choices

Several choices were made mainly for reliability and comparability rather than raw score chasing.

### 1. Same Representation Logic Across Modalities

For both audio and text, the sequence output is turned into a fixed-size representation using the same family of summary statistics. That keeps the downstream classifier setup consistent across pipelines.

### 2. Separate Modalities, Comparable Heads

The audio-only, ASR-text, provided-transcript, and fusion pipelines all end in the same style of classification head. That makes comparisons much easier to interpret.

### 3. Dropout In The Classifier Head

Dropout is used in the classifier head to reduce overfitting. This matters because the classifier is trained on top of strong frozen embeddings, and without enough regularization it can overfit the training split quickly.

### 4. Class Weighting

Class weighting is used to reduce the bias toward the dominant classes. This is important on IEMOCAP because some emotions are much more frequent than others.

### 5. Fixed Random Seeds

Random seeds were set so runs are more reproducible and comparisons across pipeline variants are more meaningful.

### 6. Cached Intermediate Features

Heavy intermediate computations were cached so experiments are cheaper to rerun. This is mostly an engineering choice for iteration speed and reproducibility.

### 7. Results Saved Separately

Runs save their outputs into dedicated result folders so metrics, predictions, and histories from different experiments do not overwrite one another.

## Hyperparameter Choices

There was not an exhaustive hyperparameter search.

The approach was closer to pragmatic tuning:

- keep the original project defaults when they were still reasonable
- lower the learning rate and dropout slightly for the fusion setup when that made the training behavior more stable
- add validation and early stopping rather than pretending that one fixed epoch count is always correct

So this was not a large search over many combinations. It was a controlled set of sensible engineering choices aimed at getting cleaner and more robust baselines.

## The Three Core IEMOCAP Pipelines

### Audio-Only

This pipeline uses only the speech signal. It is the cleanest test of how much emotional information can be extracted from acoustics alone.

### Whisper + Text Embeddings

This pipeline first converts speech to text with ASR, then predicts emotion from the text representation. It answers the question: how much emotion can be recovered if the model only sees what was said, but through an automatic transcription system.

### Provided Transcript + Text Embeddings

This pipeline uses the transcriptions already present in the dataset. It is usually the fairest text baseline because it removes ASR errors and measures the predictive value of the actual linguistic content.

## How The Fusion Pipeline Works

Yes: the fusion pipeline combines the audio embedding and the text embedding by concatenating them into one larger representation, and that fused representation is passed to a classifier head.

Conceptually, it works like this:

1. build one fixed-size representation from audio
2. build one fixed-size representation from text
3. concatenate the two
4. train one classifier on top of the combined vector

This was done for the same reason fusion is usually done in multimodal systems: audio and text carry different kinds of emotional evidence.

- audio captures prosody, energy, pitch, speaking style
- text captures lexical and semantic content

If the two modalities are complementary, fusion should help. If one modality already dominates the signal, fusion may improve calibration or some classes without necessarily giving a large headline accuracy gain.

## Fusion On The Different Datasets

The same high-level fusion idea is used across datasets, but the text source changes depending on what the dataset naturally provides.

### On IEMOCAP

- audio representation from speech
- text representation from either ASR or the provided transcript
- concatenation of both
- classifier head on top

### On RAVDESS

RAVDESS does not come with natural transcripts in the same way, so the text side is created through ASR. The fusion setup therefore combines:

- audio representation from speech
- text representation from Whisper transcription
- concatenation
- classifier head

### On MELD

MELD already includes utterance text, so there is no need to add ASR. The fusion setup combines:

- audio representation from the utterance audio
- text representation from the provided utterance text
- concatenation
- classifier head

This is the natural dataset-specific adaptation: use ASR only when transcripts are missing, and use provided text when the dataset already contains it.

## What These Choices Try To Optimize

These choices aim to produce pipelines that are:

- comparable across modalities
- harder to fool with leakage
- reasonably regularized
- cheap enough to rerun
- structured enough for later improvements

So the result is not a "max-performance at any cost" setup. It is a clean baseline family that can be trusted and extended.
