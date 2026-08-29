# Report

Group: ???

Members:
- Dang
- Bao

## Table of Contents

- [Datasets](#datasets)
- [Models](#models)
- [Evaluation Protocol](#evaluation-protocol)
- [Decision Threshold](#decision-threshold)
- [Enrollment Procedure](#enrollment-procedure)
- [References](#references)
- [Secure Virtual Assistant](#secure-virtual-assistant)

## Datasets

This is the Vietnam-Celeb dataset, which consists of 1,000 speakers and more than 87,000 utterances. The total duration of the dataset is 187 hours, with all utterances resampled to 16,000 Hz. Vietnam-Celeb includes gender and dialect labels for all speakers, which is crucial in building a Vietnamese speech dataset.

There are two test sets from Vietnam-Celeb, sampled 120 speakers from the data, with consideration to making sure the test data is gender-balanced and dialect-balanced. When creating the test sets, 120 speakers are chosen among the speakers who have the highest speech similarity scores and visual similarity scores.

- **Vietnam-Celeb-E**, an easy test set. Non-target (negative) trials are sampled randomly.
- **Vietnam-Celeb-H**, a hard test set. Non-target (negative) trials are created from speaker pairs in which each pair have the same gender and dialect labels.


|subset|# of spks|# of utters|# of utter pairs|
|-|-:|-:|-:|
|Vietnam-Celeb-E| 120|  4,207| 55,015|
|Vietnam-Celeb-H| 120|  4,217| 55,015|

And this is the incomplete version of the dataset from Kaggle.
- Vietnam-Celeb-E: Dropping 7,532 pairs referencing 194 missing audio files.
- Vietnam-Celeb-H: Dropping 3,969 pairs referencing 185 missing audio files.

|subset|# of spks|# of utters|# of utter pairs|
|-|-:|-:|-:|
|Vietnam-Celeb-E| 122|  4,013| 47,848|
|Vietnam-Celeb-H| 121|  4,032| 51,047|

## Models

The baseline model is [ECAPA](https://arxiv.org/abs/2005.07143) trained on VoxCeleb (1 and 2), using the pretrained checkpoint from SpeechBrain's [`speechbrain/spkrec-ecapa-voxceleb`](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb). A comparison with the baseline is the trained model with Vietnam-Celeb, taking the result from [Vietnam-Celeb paper](https://www.isca-archive.org/interspeech_2023/pham23b_interspeech.html).

We also experimented with fine-tuning the baseline model:
- For full parameter fine-tuning, we take the model result from [Vietnam-Celeb paper](https://www.isca-archive.org/interspeech_2023/pham23b_interspeech.html).
    - Optimized using Additive Angular Margin (AAM-Softmax) Loss.
    - Full fine-tuning method, using the same training configurations as the trained mode with less epochs, 30 epochs.
- For parameter efficient finetuning (PEFT), we used the model [`Nampfiev1995/pvad-speechbrain-ft`](https://huggingface.co/Nampfiev1995/pvad-speechbrain-ft).
    - Optimized using Additive Margin (AM-Softmax) loss.
    - Adapter-based method, using Residual Embedding Adapter with Vietnamese speaker datasets (such as VoxVietnam and Vietnam-Celeb).

## Evaluation Protocol

### a. Evaluation Trials

In speaker verification (SV), a trial is a test case involving an enrollment utterance and a test utterance. With modern embedding-based SV systems, both the enrollment utterance and the test utterance are represented by embedding vectors.

If the two embedding vectors are from the same speaker then the trial is ‘target’ or ‘positive’, and if they are from different speakers, the trial is ‘non-target’ or ‘negative’.

|subset|# of trials|# target trials| # non-target trials|
|-|-:|-:|-:|
|Vietnam-Celeb-E| 47,848| 3,965|  43,519|
|Vietnam-Celeb-H| 51,047| 3,978|  47,069|

### b. Speaker verification

For each model, embeddings are extracted once per utterance, and each trial pair is scored as the cosine similarity between its two embeddings. Performancewill be measured with metrics:

- **EER (Equal Error Rate):** the operating point where false-acceptance rate (FAR) equals false-rejection rate (FRR). Lower is better.
- **minDCF (minimum Detection Cost Function):** the minimum achievable cost over all thresholds under NIST-style costs ($P_{target} = 0.01$ and $C_{Miss} = C_{FA} = 1$), which weights false rejections (Miss) and false acceptances (FA) by their operating-point-relevant costs. Lower is better.

### c. Results

|Model| Easy - EER (%)| Easy - minDCF| Hard - EER (%)| Hard - minDCF|
|-|-:|-:|-:|-:|
|Vox| 16.15| 0.5779| 19.16| 0.5734|
|Vietnam-Celeb| 6.31| -| 8.62| -|
|Vox + Full fine-tuning| 7.33| -| 9.37| -|
|Vox + Adapter-based PEFT| 8.07| 0.3678| 8.95| 0.3499|

## Decision Threshold

In speaker verification system, a speaker claims his identity and ask for verification. The system uses his voice as a test utterance, while retrieving the enrolled utterance of the claimed identity. Having computed a score between the enrolled utterance and the test utterance, a **verification decision** is made whether to accept or reject the speaker or to request another utterance (or, without a claimed identity, an identification decision is made).

### Hypothesis Testing

Given a match score, the binary verification decision involves choosing between two hypotheses: that the user is the claimed speaker (target) or that he is not the claimed
speaker (non-target).

Given the hypothetical environment from Vietnam-Celeb-E and Vietnam-Celeb-H.

<div>
<img src="images/pretrained-easy.png" alt="Evaluate pretrained ECAPA on easy set" width=45%/>
<img src="images/pretrained-hard.png" alt="Evaluate pretrained ECAPA on hard set" width=45%/>
</div>

**Fig. 1** Distributions of target/non-target, and evaluation results of pretrained ECAPA. (a) Results on Vietnam-Celeb-E. (b) Results on Vietnam-Celeb-H.

<div>
<img src="images/finetuned-easy.png" alt="Evaluate finetuned ECAPA on easy set" width=45%/>
<img src="images/finetuned-hard.png" alt="Evaluate finetuned ECAPA on hard set" width=45%/>
</div>

**Fig. 2** Distributions of target/non-target, and evaluation results of finetuned ECAPA. (a) Results on Vietnam-Celeb-E. (b) Results on Vietnam-Celeb-H.

When a decision threshold (e.g. minDCF threshold) is made, speakers with a score larger than threshold is accepted (target) and smaller than threshold is rejected (non-target).
- Accepting a hypothesis

Making a decision result in these type of error
- FAR
- FRR

|Model|minDCF theshold|FAR|FRR|
|-|-:|-:|-:|
|*Vietnam-Celeb-E*| | | |
|Vox| 0.660| 0.000| 0.532|
|Vox + Adapter-based PEFT| 0.431| 0.001| 0.316|
|*Vietnam-Celeb-H*| | | |
|Vox| 0.638| 0.001| 0.493|
|Vox + Adapter-based PEFT| 0.435| 0.000| 0.316|

### Choosing Threshold

Making a decision, we need to consider the **trade-off** between
- Security (low FAR): Accepting an imposter (non-target) is costly.
- Usability (low FRR): Frequent rejection of a user is costly.

How to choose a **decision threshold** T:
- EER, minDCF thresholds
- Choosing T to satisfy a fixed FA or FR criterion (Neyman–Pearson);
- Varying T to find different FA/FR ratios and choosing T to give the desired FA/FR ratio.

There are times you need to make decision:
- Verification thershold: Choosing a threshold within the trade-off.
- Identification threshold: Given a highest possible speaker, determines whether the speaker is indentifiable (within threshold) or an unknown user.
- Enrollment threshold: Only keep quality utterances of a speaker if within a threshold.

Conclusion, we choose minDCF threshold:
- Verification
- Identification
- Enrollment

|Model|Vietnam-Celeb-E|Vietnam-Celeb-H|
|-|-:|-:|
|Vox| 0.5779 (threshold: 0.660)| 0.5734 (threshold: 0.638)|
|Vox + Adapter-based PEFT| 0.3678 (threshold: 0.431)| 0.3499 (threshold: 0.435)|



|Model|Vietnam-Celeb-E|Vietnam-Celeb-H|
|-|-:|-:|
|Vox| 16.15 (threshold: 0.336)| 19.16 (threshold: 0.370)|
|Vietnam-Celeb| 6.31| 8.62|
|Vox + Full fine-tuning| 7.33| 9.37|
|Vox + Adapter-based PEFT| 8.07 (threshold: 0.144)| 8.95 (threshold: 0.158)|


## Enrollment Procedure

From the [paper](https://www.semanticscholar.org/paper/Data-Centric-Optimization-of-Enrollment-Selection-Le-Ngo/090c0ccd5f8596a00a4c2c35a42d6d06844590d6) teacher suggested, there is a need for choosing quality enrollment utterances to ensure the performance of the speaker recognition system.

The ideas is choosing enrolled utterances with scores within a positive threshold. We have chosen the positive threshold from the above section (for verify, identify, enroll):
- For simplicity, the same threshold is used for every tasks in the system.
- In production, strict threshold may applied for verify (cause high security) and enroll (cause user in controlled environment), while loose threshold can be used for identification.

Enrollment procedure
- Enroll 5 utterances, and choose 3 closest utterances within threshold.
- Otherwise, request user for re-enrollment.

## Secure Virtual Assistant

### a. System Architecture

Technology stack
- Components
- Why choose these components

System architecture
- mic > ASR > Orchestrator > normal/personal/sensitive function > Deny/Execution > TTS > Speaker

### b. Processing Flow

Enrollment flow
- Enroll success -> Create user
- Enroll failed -> Ask re-enroll (how many times?)

Chatting flow
- Normal function -> Execute
- Personal function
    - Found user -> Execute
    - Not found -> Repeat new utterance, or deny?
- Sensitive function
    - Verified -> Execute
    - Failed -> Re-try (how many times?)

## References

Datasets
- \[[paper](https://www.isca-archive.org/interspeech_2023/pham23b_interspeech.html)\] Vietnam-Celeb: a large-scale dataset for Vietnamese speaker recognition.

Models
- \[[paper](https://arxiv.org/abs/2005.07143)\] ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification.
- \[[source](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb)\] Pretrained ECAPA-TDNN model from SpeechBrain - Hugging Face.
- \[[source](https://huggingface.co/Nampfiev1995/pvad-speechbrain-ft)\] Adapter-based PEFT ECAPA-TDNN model, loaded with SpeechBrain interface - Hugging Face.

Enrollment procedure
- \[[paper](https://www.semanticscholar.org/paper/Data-Centric-Optimization-of-Enrollment-Selection-Le-Ngo/090c0ccd5f8596a00a4c2c35a42d6d06844590d6)\] Data-Centric Optimization of Enrollment Selection in Speaker Identification.