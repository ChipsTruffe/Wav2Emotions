# Pipeline Explanation

## IEMOCAP Pipelines

The IEMOCAP pipelines were built to stay comparable:

- one audio-only model
- one text model using Whisper transcriptions
- one text model using the dataset's provided transcripts
- one fusion model using both audio and text

The idea was simple: keep the same general classifier style, and change only the source of information. That makes the comparison easier to trust.

## Healthy Choices

A few choices were made to keep the experiments clean:

- keep one full session aside for final testing instead of mixing everything randomly
- create validation only from the training sessions
- use early stopping based on validation loss
- restore the best model, not the last one
- use dropout in the classifier head to reduce overfitting
- use class weighting when class imbalance is an issue
- cache expensive intermediate features so runs are reproducible and cheap to rerun

This is not a huge hyperparameter search. It is more a small set of sensible engineering choices to get solid baselines.

## Validation And Early Stopping

The main point of the validation regime is to avoid tuning directly on the final test set.

So the logic is:

1. train on the training split
2. monitor validation loss during training
3. stop when validation stops improving
4. keep the best validation checkpoint
5. evaluate once on the held-out test set

This is healthier than training for a fixed number of epochs and always keeping the final weights.

## Fusion

Yes: the fusion pipeline is built by combining the audio embedding and the text embedding into one larger vector, then sending that combined vector into a classification head.

Intuitively:

- audio gives prosody and speaking style
- text gives semantic content
- fusion lets the classifier use both at once

## Fusion By Dataset

The fusion idea is the same everywhere, but the text source depends on the dataset:

- IEMOCAP: audio + either Whisper text or provided transcript
- RAVDESS: audio + Whisper text
- MELD: audio + provided utterance text

So the design stays consistent, while still adapting to what each dataset naturally provides.
