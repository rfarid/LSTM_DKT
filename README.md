# Deep Knowledge Tracing Implementation - Case Study

### Objectives

The aim of this case study is to 
- build, train, and evaluate a deep learning (DL) based knowledge tracing (KT) model [1] designed to accurately predict students' knowledge state across a set of knowledge concepts. 
- reproduce existing results and adapt the model to a larger, more modern dataset.

#### Dataset
For this challenge, I use the following real-world public dataset to train and evaluate all DL models: XES3G5M [2].
XES3G5M is a large-scale dataset that contains numerous questions and auxiliary information about related knowledge concepts
(KCs). The dataset was collected from a real online mathematics learning platform and includes 7,652 questions, 865 knowledge concepts (KCs), and
5,549,635 learning interactions from 18,066 students.

As reported in the dataset paper [2], statistical patterns informed preprocessing decisions:
- Sequence Length: ~89% of students have 200–500 interactions.
- Answer Correctness: 54% of students correctly answer up to 80% of attempted questions.
- Temporal Gaps: On average, there's an 80-hour variance between consecutive interactions.
- Interaction Duration: Most students engage over periods longer than a year.

These characteristics can be used in sub-sampling and filtering strategies to ensure representative training data while avoiding extreme outliers (e.g., very short sequences or low-reliability users).

---

### Methodology Details

1. **Literature Review and Code Availability Check**  
   Initial investigations revealed that the original authors of [1] had not released the implementation of their LSTM-based Deep Knowledge Tracing (DKT) model.

2. **Alternative Source Identification**  
   A publicly available Python implementation of DKT [1] was found on GitHub [3], but it was developed in Python 2 and not directly usable.

3. **Code Porting and Validation**  
   The DKT implementation was successfully ported to **Python 3.10** and **Ubuntu 22.04** (`dkt_3_10_k2_15.py`). Necessary library updates and compatibility adjustments were made.  
   The ported model was then validated on the **assistments** dataset, reproducing results consistent with those reported in the original paper.

4. **Dataset Switch to XES3G5M**  
   The project then transitioned to using the **XES3G5M** dataset, which provides significantly larger and richer interaction sequences.

5. **Data Preprocessing and Conversion**  
   Since the XES3G5M dataset format differed from Python implementation[3], a custom data conversion pipeline was developed (`convert_stat_k_fold_cv_with_sampling.py`) to transform it into the expected `dataset.txt` and `dataset_split.txt` format. Key steps included:
   - Flattening sequence-level rows into (user, skill, correct) triples.
   - Filtering based on `selectmasks` to exclude invalid interactions.
   - Removing short sequences (less than a threshold length).
   - Optional filtering of unreliable students based on average correctness.
   - Filtering thresholds for sequence length (`min_seq_len ≥ 100`) and user correctness (`avg_correct ≥ 0.4`) were chosen based on summary statistics reported in the dataset paper (Figure 3), aiming to retain representative, high-quality data.

6. **Cross-Validation and Sampling**  
   Due to the size of the dataset, stratified k-fold cross-validation (k=5) was implemented with optional sub-sampling to reduce compute time while maintaining representative user performance distributions. This required:
   - Calculating per-user average correctness.
   - Discretising users into strata for stratified splits.
   - Writing `dataset_split_X.txt` files to guide the DKT training script.

7. **Model Evaluation on K-Fold Splits**  
   Training was conducted on each fold. AUC and accuracy metrics were collected. Results were consistent across the five folds, demonstrating the robustness of the model and sampling strategy.

8. **Subsampling Consideration and Final Decision**  
   Initially, cross-validation was performed using a subsample fraction of `0.35` to reduce computational load. A follow-up experiment was considered with a larger `sample_frac = 0.70` for finer-grained training.  
   However, the first epoch of full dataset training (90/10 split) already yielded a strong baseline AUC of **0.759** and accuracy of **0.820**, indicating good generalisation from cross-validation.  
   Based on this result, it was deemed unnecessary to rerun 5-fold CV with a larger subsample, as diminishing returns were expected.

9. **Final Training on Full Dataset**  
   After cross-validation, the model was trained on the full dataset using a 90% training / 10% test split. Users with low reliability (average correctness < 0.4) or very short sequences were excluded, similar to the cross-validation approach.

---

### Result Details

#### Training Details
- Model: LSTM-based DKT
- Framework: TensorFlow 2.15 / Keras 2.15
- Environment: Python 3.10, Ubuntu 22.04
- Dataset: XES3G5M (question-level format)
- Filters: `avg_correct ≥ 0.4`, `min_seq_len ≥ 100`
- Training Config:  
  - 5-fold cross-validation with `sample_frac = 0.35`  
  - Full dataset training with `sample_frac = None`  
- requirements.txt file is provided.
 
#### High-Level Evaluation Results

- **5-Fold Cross-Validation**

  | Fold | Loss       | AUC       | Accuracy  |
  |------|------------|-----------|-----------|
  | 1    | -23305.68  | 0.783934  | 0.823746  |
  | 2    | -23854.98  | 0.781848  | 0.814484  |
  | 3    | -23485.15  | 0.778143  | 0.813507  |
  | 4    | -23855.47  | 0.782858  | 0.814854  |
  | 5    | -24099.69  | 0.781663  | 0.815251  |

  | Metric       | Mean       | Standard Deviation |
  |--------------|------------|--------------------|
  | AUC          | 0.78129    | 0.00198            |
  | Accuracy     | 0.81637    | 0.00381            |
  | Loss         | -23720.59  | 313.64             |

  These results show stable and consistent performance across all folds, validating both the sampling strategy and the training configuration.

- **Final Training (Full Dataset, 90/10 Split)**

  Training was conducted over 10 epochs. Early stopping was configured but not triggered.

  | Epoch | Loss       | AUC       | Accuracy  |
  |-------|------------|-----------|-----------|
  | 0     | -74672.36  | 0.75947   | 0.81977   |
  | 1     | -75933.11  | 0.77291   | 0.82292   |
  | 2     | -76409.50  | 0.77754   | 0.82440   |
  | 3     | -75484.51  | 0.77920   | 0.82500   |
  | 4     | -76615.41  | 0.78030   | 0.82415   |
  | 5     | -76384.21  | 0.78063   | 0.82569   |
  | 6     | -76611.16  | 0.78181   | 0.82569   |
  | 7     | -76279.94  | 0.78398   | 0.82574   |
  | 8     | -76301.28  | 0.78274   | 0.82442   |
  | 9     | -75982.76  | 0.78344   | 0.82557   |

  These results demonstrate that the model continues to improve steadily in AUC and accuracy as training progresses, with peak AUC ~0.784 and accuracy ~0.826.

* Full statistics are available in the accompanying result logs:
    * dkt_xes3g5m_k_cv folder for 5-fold cross-validation: `all_hisory_summary.txt`
    * dkt_xes3g5m_90_10 folder for final training: `dataset.txt.history`


#### Implementation Note
##### Excessive padding
During training (both 5-fold CV and full dataset 90/10 run), warnings were raised due to **excessive padding** in some batches (e.g., 2 out of 5 samples = 40%).  
This padding was automatically added to complete the final (or partial) batches when the total number of sequences was not divisible by the `batch_size`.  
It occurred at different batch indices depending on the dataset split and did not indicate a data issue.  
Model performance was unaffected, with AUC and accuracy progressing steadily across epochs. 
To suppress this warning in future or prevent padding:
- Consider setting `drop_last=True` when batching (if your framework supports it), or
- Choose a `batch_size` that divides the number of sequences evenly.

##### Visualising the output
dkt_plots.py is provided to visualise the AUC and Accuracy values throughout epochs.

---

### Concluding Remarks

#### Strengths
- Successfully adapted a legacy DKT implementation to a modern Python environment.
- Validated model reproducibility on benchmark datasets.
- Designed a general-purpose, reproducible data conversion and k-fold splitting pipeline for large KT datasets.
- Demonstrated strong predictive performance even on the first epoch.
- Evaluation showed consistency and reliability across cross-validation folds.

#### Weaknesses
- The original DKT implementation was based on TensorFlow 1.x, which posed extensibility challenges; this has been addressed by porting to TensorFlow 2.x.

#### Opportunities
- Incorporate other features into the input encoding.
- Explore attention-based KT models (e.g., SAKT[4]) for potentially improved performance and interpretability.
- Analyse performance by user segments (e.g., low vs. high performers, novices vs. experienced learners) to uncover model biases and guide personalised interventions or curriculum adjustments.


---

### References

[1] [Deep Knowledge Tracing](https://github.com/chrispiech/DeepKnowledgeTracing/tree/master?tab=readme-ov-file)  

[2] [A Knowledge Tracing Benchmark Dataset with Auxiliary Information](https://github.com/ai4ed/XES3G5M)

[3] [Deep Knowledge Tracing Implementation](https://github.com/shinyflight/Deep-Knowledge-Tracing/tree/master)

[4] [A Self-Attentive model for Knowledge Tracing](https://arxiv.org/pdf/1907.06837)


## License
This repository contains adapted code originally from [shinyflight/Deep-Knowledge-Tracing](https://github.com/shinyflight/Deep-Knowledge-Tracing), which is licensed under the MIT License.
All original MIT-licensed code remains under the MIT License (see `LICENSE_ORIGINAL.txt`).
All modifications and new code in this repository are licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License](https://creativecommons.org/licenses/by-nc/4.0/).
You are free to use, adapt, and redistribute this work for **non-commercial purposes**, provided you give appropriate credit.
