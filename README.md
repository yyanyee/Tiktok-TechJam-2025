# Tiktok-TechJam-2025
# Review Moderation with Gemma-3 12B and Baseline ML

This notebook demonstrates a workflow for automatically moderating Google location reviews. It uses a combination of the Gemma-3 12B Instruction Tuned model for initial labeling of a small dataset and a baseline machine learning model (TF-IDF + Logistic Regression) trained on this labeled data for faster batch inference on a larger dataset.

The notebook covers the following steps:

1.  **Installs & Imports**: Installs necessary libraries and imports modules.
2.  **Config, Mount Drive, Paths, Seeds**: Sets up configuration, mounts Google Drive for data and model persistence, defines paths, and sets random seeds.
3.  **Load & prep raw data → df**: Reads raw review and metadata JSONL files, merges them, removes duplicates, and renames columns.
4.  **Hugging Face login**: Allows logging in to the Hugging Face Hub for access to gated models.
5.  **Load Gemma‑3 12B from Drive cache**: Loads the Gemma-3 12B model, downloading it to a Drive cache if not found.
6.  **Prompt, parser, guard (LLM labeling)**: Defines the prompt structure, parsing logic, and a guard function for efficient LLM labeling.
7.  **Label a small preview (N_PREVIEW) & peek**: Labels a small sample of reviews using the Gemma model for a quick check.
8.  **Label N_LABEL with checkpoints and consolidate**: Labels a larger sample with checkpointing to save progress and consolidates the results.
9.  **Build ML‑ready split (preview‑only, multilabel stratified)**: Prepares the labeled data for ML by creating target columns and performing a multilabel stratified split into train, validation, and test sets.
10. **Train baseline (TF‑IDF + Logistic Regression)**: Trains a baseline Logistic Regression model for each label using TF-IDF features and finds optimal thresholds on the validation set.
11. **Error analysis (val/test), save misclassifications**: Evaluates the baseline model on validation and test sets and saves misclassified examples.
12. **Threshold calibration from PR curves (val)**: (Optional) Recalibrates thresholds using Precision-Recall curves on the validation set.
13. **Batch inference utility (score any DataFrame)**: Provides a function to apply the trained baseline models to any DataFrame for batch prediction.
14. **Apply trained model to entire dataset**: Applies the trained baseline model to the full dataset.
15. **Show predicted label distribution**: Displays the counts for each predicted label on the full dataset.
16. **Examples of Reviews Predicted as 'ads'**: Shows examples of reviews predicted to contain advertisements.
17. **Examples of Reviews Predicted as 'irrelevant'**: Shows examples of reviews predicted as irrelevant.
18. **Examples of Reviews Predicted as 'non_visit_rant'**: Shows examples of reviews predicted as rants without a visit.
19. **Examples of Reviews Predicted as 'spam'**: Shows examples of reviews predicted as spam.

## Setup

1.  **Google Drive**: Ensure your Google Drive is mounted and the `OUT_DIR` is correctly set in Cell 2.
2.  **Data**: Place your `review-Vermont.json` and `meta-Vermont.json` files in the `OUT_DIR`. Update `REVIEW_PATH` and `META_PATH` in Cell 2 if your file names or locations are different.
3.  **Hugging Face Token**: If using a gated model like Gemma-3 12B, log in to the Hugging Face Hub in Cell 4.
4.  **Runtime**: Use a GPU runtime for faster model loading and inference.

## Usage

Run the cells sequentially. Cells 7 and 8 involve LLM inference and may take time depending on the sample size (`N_PREVIEW` and `N_LABEL`). Cells 10 and onwards train and apply the baseline ML model.

The final `scored_df` DataFrame (generated in Cell 14) will contain the original review data along with predicted probabilities and binary labels for each moderation category.

## Output

-   Checkpoint files for LLM labeling in `OUT_DIR/labels_ckpt_1k`.
-   Labeled preview data in Parquet and CSV format (`labeled_preview_1000.parquet`, `labeled_preview_1000.csv`) in `OUT_DIR`.
-   ML-ready split data (`labeled_reviews_ml_ready.parquet`) in `OUT_DIR`.
-   Trained baseline models, vectorizers, thresholds, and metrics in `OUT_DIR/models/baseline_tfidf_lr`.
-   CSVs of misclassified instances in `OUT_DIR/models/baseline_tfidf_lr`.
-   The full dataset with predicted labels (`scored_df`) available in the notebook environment.
