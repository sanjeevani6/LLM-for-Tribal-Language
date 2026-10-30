# LLM for Cross-Lingual Communication in Indian Tribal Languages

Machine translation for **Santali** and **Bhili**, built by fine-tuning **IndicTrans2 (1B, Indic-Indic)** with LoRA and **vocabulary extension**: adding frequent whole words of the target language to the model's vocabulary and initializing their embeddings from existing ones.

**Course:** VII Semester Mini-Project, CSE B.Tech
**Mentor:** Prof. Divyansh Thakur
**Status:** Ongoing

---

## 1. Motivation

India has hundreds of tribal languages, many with millions of speakers, but almost no presence in modern NLP tooling. Translation systems cover the 22 Scheduled Languages well, while tribal languages lack large digitized parallel corpora and are often poorly served by pretrained tokenizers.

- **Santali** is written in its own script, **Ol Chiki**, and belongs to the Munda (Austroasiatic) family. It is a Scheduled Language and is covered by IndicTrans2.
- **Bhili** is an Indo-Aryan language written in **Devanagari** in our dataset (Jhabua dialect, Madhya Pradesh). It is not a supported IndicTrans2 language, so we use Marathi as a proxy (section 5.2).

## 2. Summary of findings

**Question.** Does extending IndicTrans2's target vocabulary with frequent whole words improve translation into a tribal language under LoRA fine-tuning?

- **Bhili (Hindi → Bhili):** yes. Every extension beats the baseline, and the best setting (threshold 5, 2,729 new words) gains **+1.23 BLEU and +1.44 chrF**.
- **Santali (English → Santali):** yes, but modestly, and the gain is small compared with seed-to-seed variation. A 445-word extension with mean initialization gives about **+0.57 BLEU and +0.71 chrF** over the average of two baseline seeds (13.08 versus 12.51 BLEU; 46.16 versus 45.45 chrF). chrF is the more reliable signal: all four mean-init comparisons against the two baseline seeds have chrF intervals above zero.
- A 95-word extension has no measurable effect, and a 1,220-word extension gives a similar gain to 445 words, so the gain does not keep growing past a few hundred words.
- **Random initialization keeps most of the gain**, so compositional (mean) initialization is at most slightly better, and this is not resolved with the runs we have.
- **Methodological lesson:** a first Santali comparison appeared to show a large gain (test chrF 41.0 to 44.8), but it came from unequal training of the baseline, not from the vocabulary (section 6.2.2).

## 3. Languages and experiments

| Language | Script | Experiments |
|---|---|---|
| Bhili | Devanagari | Hindi → Bhili. Baseline LoRA, plus vocabulary extension at five frequency thresholds (50, 20, 10, 5, 2). Test set: 990 sentences |
| Santali | Ol Chiki | English → Santali (`eng_Latn` → `sat_Olck`). Zero-shot, baseline LoRA (2 seeds), 95-word extension, 445-word extension with mean initialization (2 seeds) and random initialization (1 seed), 1,220-word extension. Evaluated on a held-out test split (994 sentences) and on IN22-Gen (1,024 sentences) |

## 4. Data

**MMLoSo 2025** shared task corpus ([findings paper](https://aclanthology.org/2025.mmloso-1.14/)), Santali and Bhili portions:

- 20,000 parallel sentence pairs per language pair, released under CC BY-SA 4.0.
- Bhili is paired with Hindi; Santali is paired with English.
- The English side of the Santali data was machine-translated from Hindi with IndicTrans2 and lowercased.

**Splits.** Santali is split once into **17,883 training, 993 validation and 994 test** sentences, and the same frozen splits are used in every experiment. The Bhili test set has 990 sentences.

**IN22-Gen** (AI4Bharat) is used as an additional general-domain test set for Santali (1,024 sentences).

Preprocessing is done in `common-preprocessing.ipynb` and `common-tribal-preprocessing*.ipynb`.

## 5. Method

```
 source text ─► IndicTrans2 tokenizer ─► IndicTrans2 1B (Indic-Indic) ─► target text
                 (target vocabulary           + LoRA adapters
                  extended with new           (base weights frozen;
                  frequent words)              new embedding rows trained)
```

### 5.1 Base model
`ai4bharat/indictrans2-indic-indic-1B`, fine-tuned with LoRA.

IndicTrans2 was chosen because the MMLoSo findings report that NLLB's tokenizer fragments Santali heavily (about 3.0 subwords per word), while the winning team switched to IndicTrans2 for its much better handling of Ol Chiki.

### 5.2 Marathi as a proxy for Bhili
Bhili is not one of IndicTrans2's supported languages, so we use the **Marathi tag (`mar_Deva`)** for the Bhili side. Our reasons:

- Bhili and Marathi share the **Devanagari script** and are both Indo-Aryan.
- The two languages are geographically close, which we consider a supporting factor.
- The same choice was made by the rank-1 team of MMLoSo 2025 (Raj, *LoRAs in All Directions*), who mapped Bhili to `mar_Deva`.

### 5.3 Vocabulary extension
1. **Candidates:** whole whitespace-delimited training words that occur at least `MIN_FREQ` times, contain no punctuation or symbols, and are split into more than one SentencePiece piece by the original tokenizer. Lower thresholds add more words. Candidate lists for every threshold are stored in `candidate_lists_by_threshold.json`.
2. **New parameters:** each candidate gets a new decoder-embedding row and output-layer row. These rows are trained in fp32 while the original rows stay frozen.
3. **Initialization:**
   - *Mean (compositional):* the new row is the mean of the rows of the pieces the word was previously split into.
   - *Random:* the new row is drawn from a normal distribution matching the original rows.
4. **Labels:** a candidate is replaced by its new ID only where it forms a whole word; all other text uses the original tokenizer.

Santali thresholds (Bhili uses thresholds 50, 20, 10, 5 and 2):

| Frequency threshold | New words | Held-out token reduction |
|---|---|---|
| 50 | 95 | 1.89% |
| 20 | 445 | 4.73% |
| 10 | 1,220 | 7.36% |

### 5.4 Training configuration (Santali experiments, identical for every arm)
LoRA rank 16, alpha 32, dropout 0.05 on `q_proj` and `v_proj`; batch size 4 with gradient accumulation 4; AdamW with learning rate 1e-4 and weight decay 0.01; 3% warmup then linear decay; fp16 autocast with gradient scaling and skipping of non-finite gradients; gradient clipping at 1.0; early stopping with patience 2 and minimum delta 0.001; 10 planned epochs; IndicProcessor source preprocessing. The best checkpoint by validation loss is used.

### 5.5 Evaluation pipeline (identical for every arm)
IndicProcessor prompts, batch size 8, 4-beam search, `max_length=256`, early stopping, no sampling. Generated IDs are decoded with the standard tokenizer decoding; added words are inserted whole, with a space before them and no space before a word-continuation piece or closing punctuation. Scores are corpus BLEU, chrF, chrF++ and TER from `sacrebleu` with default settings. Intervals are **paired bootstrap over test sentences** (they cover test-sentence sampling only, not training randomness).

Additional diagnostics: `word_len_ratio` (output over reference length, ideal 1.0), empty outputs, a clean IN22-Gen subset (1,023 sentences, one training-overlap sentence removed), subset analysis by whether the reference or output contains an added word, and word-level recall and precision of the added words.

### 5.6 Environment
Kaggle T4 GPU; transformers 4.46.3, tokenizers 0.20.3, IndicTransToolkit 1.1.1, peft 0.20.0, torch 2.11.0+cu128, accelerate 1.14.0, Python 3.13.15.

### 5.7 Experimental design
| System | Purpose |
|---|---|
| Pretrained, no adaptation | Zero-shot reference |
| Baseline LoRA | Fine-tuning without vocabulary change |
| Extension + random init | Effect of the extra words alone |
| Extension + mean init | Effect of the extra words plus smart initialization |
| Threshold sweep | Effect of how many words are added |

## 6. Results

### 6.1 Bhili (Hindi → Bhili, test set, n = 990, chrF)

| System | New words | BLEU | chrF | TER ↓ | word_len_ratio | Δ BLEU | Δ chrF |
|---|---|---|---|---|---|---|---|
| Baseline LoRA | 0 | 12.17 | 43.34 | 69.98 | 0.976 | – | – |
| Extension, threshold 50 | 166 | 12.94 | 44.28 | 70.26 | 0.987 | +0.77 | +0.94 |
| Extension, threshold 20 | 510 | 13.09 | 44.37 | 69.96 | 0.986 | +0.92 | +1.03 |
| Extension, threshold 10 | 1,173 | 13.27 | 44.75 | 69.09 | 0.980 | +1.10 | +1.41 |
| Extension, threshold 5 | 2,729 | **13.40** | 44.78 | 69.00 | 0.982 | **+1.23** | **+1.44** |
| Extension, threshold 2 | 2,729 | 13.32 | 44.78 | 69.00 | 0.981 | +1.15 | +1.44 |

No outputs were empty in any run.

- Every extended system beats the baseline on BLEU and chrF.
- Scores rise as the threshold is lowered and **plateau around threshold 5**. Between thresholds 5 and 2, BLEU moves by 0.08 and chrF is unchanged.
- The smallest extension (166 words) has a slightly *worse* TER than the baseline (70.26 versus 69.98); TER improves clearly only from 1,173 words upward.
- Differences of a few tenths of a BLEU point between adjacent thresholds are small and should not be over-interpreted.

### 6.2 Santali (English → Santali)

#### 6.2.1 Tokenizer audit
- The original tokenizer's fertility on Santali is **1.442 tokens per word**, matching the published figure (1.44 to 1.45).
- About **1.7% of Santali target tokens** map to the model's unknown token (train 1.72%, validation 1.63%, test 1.39%), mostly Ol Chiki digits and some whole-word pieces missing from the vocabulary. A perfect round trip through the vocabulary still reaches 98.4 chrF, so this is not what limits quality.
- **17.4% of test references contain Ol Chiki digits**, and no system produced any. Removing digits from the references changes BLEU by only +0.23 to +0.28 and chrF by +0.49 to +0.50, so the achievable gain from digit handling is small and was not pursued.

#### 6.2.2 The first comparison was misleading
The first baseline trained for 7 epochs in an unscaled fp16 loop, and the session was interrupted while validation loss was still falling. The extended model, trained for 10 epochs with mixed-precision gradient scaling, looked clearly better (test 8.39 → 10.83 BLEU, 41.04 → 44.84 chrF). Diagnostics showed this could not be credited to the vocabulary:

- The 95 added words change only about 2% of target tokens and about a third of sentences.
- The extended model's validation loss was already about 0.5 lower at epoch 1, far more than tokenization could explain (a best-case bound of about 0.05).
- Evaluation settings moved the baseline by only about 1 BLEU, so the pipeline was not the cause.
- A decoder bug that glued words next to added tokens had understated the extended model's BLEU (round-trip BLEU 86.7, versus 96.6 after the fix).

Retraining the baseline (**baseline v2**) with the extended model's training procedure removed the gap. Under one identical pipeline, the old baseline scored 9.00 BLEU and 41.45 chrF; baseline v2 scores 12.32 BLEU and 45.26 chrF, a gain of **+3.31 BLEU (95% interval +2.46 to +4.22) and +3.78 chrF**. All later comparisons use baseline v2.

| Run | Epochs | Epoch-1 validation loss | Best validation loss | Final training loss |
|---|---|---|---|---|
| Baseline v2 (scaled loop) | 10 | 2.5847 | 2.4662 | 2.6085 |
| Baseline v1 (unscaled loop) | 7 | 3.1481 | 3.0152 | 3.3140 |
| Extended (95 words, mean init) | 10 | 2.6463 | 2.5149 | 2.6497 |

#### 6.2.3 Main results (held-out test split, n = 994)

| System | BLEU | chrF | TER ↓ |
|---|---|---|---|
| Baseline, seed 42 | 12.32 | 45.26 | 76.99 |
| Baseline, seed 43 | 12.70 | 45.64 | 77.07 |
| 95 words, mean init (earlier code path) | 12.40 | 45.64 | 77.54 |
| 445 words, mean init, seed 42 | 13.15 | 46.15 | **73.98** |
| 445 words, mean init, seed 43 | 13.00 | **46.17** | 76.00 |
| 445 words, random init, seed 42 | 12.76 | 46.04 | 75.11 |
| 1,220 words, mean init, seed 42 | 13.02 | 46.10 | 77.21 |

The baseline's own seed spread is 0.38 BLEU and 0.38 chrF; the spread between the two mean-init seeds is 0.15 BLEU and 0.02 chrF.

#### 6.2.4 Paired bootstrap on the test split (difference, 95% interval)

| Run | vs baseline seed 42 (BLEU) | vs baseline seed 42 (chrF) | vs baseline seed 43 (BLEU) | vs baseline seed 43 (chrF) |
|---|---|---|---|---|
| Mean init 445, seed 42 | +0.82 [+0.32, +1.41] | +0.89 [+0.44, +1.38] | +0.45 [−0.09, +1.01] | +0.50 [+0.10, +0.92] |
| Mean init 445, seed 43 | +0.67 [+0.01, +1.38] | +0.92 [+0.38, +1.48] | +0.30 [−0.30, +0.89] | +0.53 [+0.05, +0.99] |
| Random init 445, seed 42 | +0.43 [−0.12, +1.02] | +0.78 [+0.31, +1.26] | +0.06 [−0.53, +0.65] | +0.40 [−0.04, +0.88] |

Other comparisons against baseline seed 42: the 95-word run gives +0.06 BLEU and +0.39 chrF (both intervals include zero), and the 1,220-word run gives +0.70 BLEU and +0.85 chrF.

Seed spreads: baseline seed 43 minus seed 42 is +0.37 BLEU [−0.12, +0.88] and +0.38 chrF [−0.02, +0.77]; mean-init seed 43 minus seed 42 is −0.15 BLEU [−0.66, +0.30] and +0.03 chrF [−0.38, +0.39].

**Reading.** chrF gains are consistent and mostly clear zero. BLEU gains are small, clear zero only against baseline seed 42, and are not established against the stronger baseline seed. On the 300-sentence validation sample the BLEU differences are flat (−0.23 to +0.07) and the chrF differences are positive but not significant (+0.38 to +0.65). Validation selected the checkpoints for all systems and is a small sample, so it is a weak check.

#### 6.2.5 Initialization: mean versus random

Mean minus random initialization: BLEU +0.39 [−0.02, +0.80] and chrF +0.10 [−0.26, +0.47] (seed 42); BLEU +0.24 [−0.29, +0.79] and chrF +0.13 [−0.26, +0.53] (seed 43 against random seed 42). All four intervals include zero.

Mean init learns faster early (epoch-1 validation loss 2.739 versus 2.825) and uses the added words in more test sentences (507 versus 452), but by epoch 10 the validation-loss gap is about 0.01. Compositional initialization therefore appears to help at most slightly, and one random seed cannot resolve this.

#### 6.2.6 Where the gain comes from (test split)

**Sentence level.** For the 445-word mean-init runs, the gain over baseline v2 sits in sentences that contain added words: about **+1.1 BLEU and +1.1 to +1.3 chrF** where the reference contains an added word, against about +0.1 BLEU and +0.4 chrF elsewhere. The 95-word run showed no such pattern.

**Word level.** The 445 added words occur 1,091 times in the test references. We compare them with two control sets of words that were *not* added:
- **Matched control (B):** 775 multi-piece words with training frequency 10 to 19 (499 reference occurrences), which are as hard to tokenize as the added words.
- **Loose control (A):** 1,328 frequent words not added, mostly already single tokens (12,759 reference occurrences).

| System | Added recall % | Added precision % | Control A recall % | Control A precision % | Control B recall % | Control B precision % |
|---|---|---|---|---|---|---|
| Baseline, seed 42 | 41.43 | 26.56 | 51.85 | 44.25 | 32.26 | 31.45 |
| Baseline, seed 43 | 41.70 | 25.46 | 52.39 | 44.79 | 33.27 | 32.17 |
| Mean init, seed 42 | 43.45 | 29.57 | 53.24 | 45.90 | 32.67 | 31.90 |
| Mean init, seed 43 | 42.80 | 30.48 | 53.41 | 44.55 | 32.46 | 32.46 |
| Random init, seed 42 | 44.36 | 31.76 | 53.23 | 45.11 | 32.06 | 31.56 |

Recall change versus the baseline average, in percentage points, with 95% intervals:

| Run | Added words | Control A | Control B |
|---|---|---|---|
| Mean init, seed 42 | +1.88 [−0.22, +4.04] | +1.12 [+0.65, +1.59] | −0.08 [−1.65, +1.45] |
| Mean init, seed 43 | +1.26 [−0.74, +3.33] | +1.30 [+0.79, +1.84] | −0.27 [−2.21, +1.45] |
| Random init, seed 42 | +2.81 [+0.79, +4.85] | +1.11 [+0.62, +1.59] | −0.69 [−2.30, +0.81] |

Added-word **precision** rises by about 3.6 to 5.8 points, against −0.3 to +0.7 for the matched control. The loose control also gains about 1.1 to 1.3 recall points, so a small general improvement on common vocabulary exists too. The improvement is therefore partly specific to the added words (clearest in precision) and partly general; the matched control is small, so its intervals are wide.

**Usage on the test split.** The 445-word mean-init models emit an added word in 507 test sentences (853 to 873 occurrences, 262 to 265 distinct words); random init in 452 sentences; the 1,220-word model in 620 sentences (1,262 occurrences, 511 distinct words).

#### 6.2.7 Additional evaluation on IN22-Gen (n = 1,024)

| System | BLEU | chrF | chrF++ | TER ↓ | word_len_ratio | Empty |
|---|---|---|---|---|---|---|
| Pretrained, no adaptation | 4.82 | 34.72 | 31.36 | 104.33 | 1.210 | 0 |
| Baseline LoRA, seed 42 | 5.88 | 37.53 | 33.75 | 96.53 | 1.144 | 0 |
| Baseline LoRA, seed 43 | 5.92 | 37.51 | 33.77 | 96.82 | 1.143 | 0 |
| Mean init, 445 words, seed 42 | 6.06 | 38.18 | 34.43 | 94.98 | 1.139 | 1 |
| Mean init, 445 words, seed 43 | **6.32** | **38.33** | **34.59** | **93.75** | 1.128 | 1 |
| Random init, 445 words, seed 42 | 5.98 | 38.12 | 34.34 | 96.73 | 1.158 | 0 |

On the clean subset (n = 1,023) every score is within 0.02 of the values above.

For orientation, Singh, Ekbal and Pakray ([paper](https://aclanthology.org/2025.mmloso-1.9/)) report 7.3 BLEU and 40.3 chrF++ for English → Santali on IN22-Gen with IndicTrans2 fine-tuned on MMLoSo plus extra public data. Their training data and scoring details differ from ours, so this is a rough reference range, not a target.

**Paired bootstrap on IN22-Gen (95% intervals)**

| Comparison | Δ BLEU | Δ chrF++ |
|---|---|---|
| Mean init (seed 42) vs baseline seed 42 | +0.18 [−0.19, +0.54] | +0.69 [+0.25, +1.10] |
| Mean init (seed 42) vs baseline seed 43 | +0.13 [−0.26, +0.51] | +0.66 [+0.23, +1.10] |
| Mean init (seed 43) vs baseline seed 42 | +0.44 [+0.15, +0.73] | +0.85 [+0.46, +1.28] |
| Mean init (seed 43) vs baseline seed 43 | +0.40 [+0.09, +0.70] | +0.82 [+0.43, +1.27] |
| Random init (seed 42) vs baseline seed 42 | +0.10 [−0.25, +0.42] | +0.59 [+0.17, +1.00] |
| Random init (seed 42) vs baseline seed 43 | +0.06 [−0.32, +0.46] | +0.56 [+0.08, +1.04] |
| Baseline seed 43 vs seed 42 (seed spread) | +0.04 [−0.20, +0.29] | +0.03 [−0.37, +0.43] |
| Baseline seed 42 vs pretrained | +1.06 [+0.62, +1.49] | +2.38 [+1.67, +3.03] |
| Baseline seed 43 vs pretrained | +1.11 [+0.66, +1.54] | +2.41 [+1.71, +3.06] |

**Subsets (IN22-Gen, difference versus baseline seed 42; ΔBLEU / ΔchrF++)**

| Run | All (1,024) | Reference has an added word (532) | Reference has none (492) | Output uses an added word | Output uses none |
|---|---|---|---|---|---|
| Mean init, seed 42 | +0.18 / +0.69 | +0.07 / +0.76 | +0.33 / +0.58 | (377) +0.26 / +1.31 | (647) +0.12 / +0.30 |
| Mean init, seed 43 | +0.44 / +0.85 | +0.60 / +1.25 | +0.22 / +0.29 | (368) +0.44 / +1.34 | (656) +0.42 / +0.54 |
| Random init, seed 42 | +0.10 / +0.59 | −0.01 / +0.71 | +0.24 / +0.43 | (312) +0.33 / +1.39 | (712) +0.01 / +0.25 |

**Usage on IN22-Gen**

| Run | Sentences using an added word | Added-word occurrences | Distinct added words used | Available |
|---|---|---|---|---|
| Mean init, seed 42 | 377 | 523 | 194 | 445 |
| Mean init, seed 43 | 368 | 509 | 188 | 445 |
| Random init, seed 42 | 312 | 407 | 177 | 445 |

Findings on IN22-Gen:
- Fine-tuning clearly helps: baseline LoRA gains +1.06 to +1.11 BLEU and about +2.4 chrF++ over the pretrained model, with intervals well above zero.
- All six extension-versus-baseline chrF++ intervals are above zero (+0.56 to +0.85); BLEU gains are smaller (+0.06 to +0.44), and only the seed-43 mean-init run has an interval above zero.
- chrF++ improves by 1.3 to 1.4 on sentences whose output uses an added word, against 0.3 to 0.5 on sentences that do not.
- Scores are roughly half those on the held-out split (BLEU 5 to 6 versus 12 to 13), reflecting the shift from the training domain to a general-domain benchmark.

## 7. Conclusion

Under matched training and evaluation, extending the target vocabulary with several hundred frequent whole words gives a **small but consistent improvement of roughly +0.6 BLEU and +0.7 chrF** for Santali on the held-out test split, concentrated in sentences containing the added words and visible as more precise production of those words. A 95-word extension is too small to matter, and 1,220 words add nothing beyond 445. Compositional initialization is not clearly better than random initialization. The gain is of the same order as the baseline's seed-to-seed variation, so it should be read as a modest effect with a reproducible direction, not a large improvement. For Bhili, the gain is larger and rises with the number of added words up to about 2,700. The much larger apparent gain in the first Santali comparison was an artefact of unequal training procedures, which is itself a lesson for comparing vocabulary-adaptation methods.

## 8. Limitations

- **Small number of runs.** Santali has two baseline seeds, two seeds for the 445-word mean-init model, and one run each for random init and for 1,220 words. Bhili has one run per setting and no significance tests yet. The baseline's seed spread is comparable to the effect.
- **Test-set reuse.** The test split scored several configurations, so a mild multiple-comparisons concern applies, and the intervals do not account for training randomness.
- **Validation does not confirm the BLEU gain.** The 300-sentence validation sample is small and was used for checkpoint selection.
- **The 95-word run came from an earlier code path.** It was interrupted after epoch 3 and resumed from a rebuilt checkpoint with a different clipping implementation, and its data order was not recorded, so comparisons against it are indicative only. The 445- and 1,220-word runs and baseline seed 43 use one unified script.
- **Added capacity is not separated from tokenization.** Random initialization shows the starting values matter little, but a control with added rows that the labels never use was not run, so "extra trainable rows" and "shorter token sequences" are not separated.
- **Mean versus random initialization is unresolved** with one random-init seed.
- **One direction per language** (English → Santali, Hindi → Bhili), and Bhili relies on a Marathi proxy tag rather than native model support.
- **Metrics.** BLEU, chrF, chrF++ and TER use `sacrebleu` defaults; IndicNLP-tokenized scoring as in the official IndicTrans2 scripts was not computed. Exact-match word metrics understate quality for a morphologically rich language, and native-speaker evaluation has not been done.
- **The English side of the Santali data is machine-generated.**
- **A 3,020-word Santali run (threshold 5)** was not performed.

## 9. Scope note: published winning pipeline

The project plan listed a reimplementation of the published winning MMLoSo pipeline (joint multi-direction fine-tuning, directional LoRA and noisy-channel reranking) as a comparison. It was **not carried out**, and no claim is made about that pipeline or about leaderboard scores. The reasons were compute (about six training stages of roughly 3 hours each for a single language pair), full-parameter joint fine-tuning of a 1B-parameter model not fitting a 15 GB T4 GPU, and the study being scoped to the vocabulary-extension question in a controlled single-direction setting. The ablation arms that depend on reranking were therefore not run; the non-reranked comparison reported here is the experiment.

## 10. Reproducibility

All Santali runs use the same frozen splits, seed-controlled data order (seed plus epoch), one evaluation script and one decoder. Per-run outputs (predictions, training histories, tokenization statistics and `run_summary.json`) were saved for every run of the unified script. Software versions are listed in section 5.6.

## 11. Repository structure

```
.
├── IndicTrans2_bhilli/                  # Hindi → Bhili experiments
│   ├── bhili-baseline.ipynb             # baseline LoRA
│   ├── bhili_extended_f02.ipynb         # vocabulary extension, threshold 2
│   ├── bhili_extended_f05.ipynb         # threshold 5
│   ├── bhili_extended_f10.ipynb         # threshold 10
│   ├── bhili_extended_f20.ipynb         # threshold 20
│   └── bhili_extended_f50.ipynb         # threshold 50
├── IndicTrans2_santali/                 # Santali experiments
├── IndicTrans2_eval/
│   ├── candidate_lists_by_threshold.json
│   ├── comparative-analysis.ipynb
│   ├── santali-in22gen-evaluation*.ipynb
│   ├── santali-vocab-tokenization-audit*.ipynb
│   └── santali_unk_and_subset_diagnostics*.ipynb
├── common-preprocessing.ipynb           # shared preprocessing
├── common-tribal-preprocessing*.ipynb   # preprocessing for the tribal-language data
├── comparative-analysis.ipynb           # comparison across systems, training curves
├── santali-in22gen-evaluation*.ipynb    # Santali evaluation on IN22-Gen
├── santali-vocab-tokenization-audit*.ipynb  # tokenizer audit (fertility, unknown tokens, digits)
└── santali_unk_and_subset_diagnostics*.ipynb # unknown-token and subset diagnostics
```

## 12. Setup and usage

1. Join the [MMLoSo 2025 competition](https://www.kaggle.com/competitions/mm-lo-so-2025/overview) on Kaggle and download the data.
2. Run the preprocessing notebooks first.
3. Run the experiment notebooks for the language you want: `IndicTrans2_bhilli/` for Bhili, `IndicTrans2_santali/` for Santali.
4. Run `comparative-analysis.ipynb` and the evaluation notebooks to reproduce the comparison tables.

The experiments ran on a Kaggle T4 GPU with the package versions listed in section 5.6.

## 13. Future work

- Run the Bhili threshold sweep with multiple seeds, add a random-initialization control and bootstrap intervals.
- Add more seeds for random initialization in Santali, and a control with added rows that are never used, to separate extra capacity from shorter token sequences.
- Run the 3,020-word Santali setting (threshold 5).
- Evaluate the reverse directions.
- Direction-specific LoRA adapters and noisy-channel reranking, as used by the rank-1 system.
- Exact-match retrieval and back-translation.
- Native-speaker evaluation and a demo interface.

## 14. References

- Singh et al., *Findings of the MMLoSo 2025 Shared Task on Machine Translation into Tribal Languages*. https://aclanthology.org/2025.mmloso-1.14/
- Raj, *Challenge Track: LoRAs in All Directions*. https://aclanthology.org/2025.mmloso-1.10/
- Singh, Ekbal and Pakray, *Evaluating IndicTrans2 and ByT5 for English–Santali Machine Translation Using the Ol Chiki Script*. https://aclanthology.org/2025.mmloso-1.9/
- Gala et al., *IndicTrans2*, TMLR 2023.
- Hu et al., *LoRA*, ICLR 2022.
- Ministry of Tribal Affairs, Adivaani: https://adivaani.tribal.gov.in

## 15. Team

| Registration No. | Name |
|---|---|
| 20233252 | Sanjeevani Jaiswal |
| 20233551 | Roshna Rajesh |
| 20233218 | Prateek |
| 20233236 | Ruchir Joshi |

## 16. Data license

The MMLoSo data is released under CC BY-SA 4.0; please respect its terms when reusing or redistributing it.
