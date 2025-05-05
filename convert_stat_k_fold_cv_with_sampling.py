import pandas as pd
import numpy as np
import os
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import KBinsDiscretizer

RELIABILITY_THRESHOLD = 0.4


def generate_stratified_kfold_from_xes3g5m(train_path, test_path, output_dir, k_folds=5, sample_frac=None, min_seq_len=20):
    os.makedirs(output_dir, exist_ok=True)

    # === Load CSVs ===
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Combine train/test, ignore their original split
    df = pd.concat([train_df, test_df], ignore_index=True)

    # === Expand each row into (uid, concept, response) triples ===
    records = []
    for _, row in df.iterrows():
        uid = row['uid']
        concepts = list(map(int, row['concepts'].split(',')))
        responses = list(map(int, row['responses'].split(',')))
        selectmasks_str = str(row['selectmasks']) if pd.notna(row['selectmasks']) else ''
        masks = list(map(int, selectmasks_str.split(','))) if selectmasks_str.strip() else []

        for c, r, m in zip(concepts, responses, masks):
            if m == 1:
                records.append((uid, c, r))

    long_df = pd.DataFrame(records, columns=["user_id", "skill_id", "correct"])

    # === Step 1: Compute user-level average correctness ===
    user_stats = long_df.groupby("user_id")["correct"].mean().reset_index()
    user_stats.columns = ["user_id", "avg_correct"]

    # === Step 2: Bin avg_correct into quantiles (only for k-fold splits) ===
    if k_folds > 1:
        binner = KBinsDiscretizer(n_bins=min(k_folds, len(user_stats)), encode='ordinal', strategy='quantile')
        user_stats["stratum"] = binner.fit_transform(user_stats[["avg_correct"]]).astype(int)
    else:
        user_stats["stratum"] = 0  # dummy stratum for full split

    # Optionally filter low-performing users only for full train/test split
    if k_folds == 1:
        before_count = len(user_stats)
        user_stats = user_stats[user_stats["avg_correct"] >= RELIABILITY_THRESHOLD]
        after_count = len(user_stats)
        print(f"ℹ️ Excluded {before_count - after_count} users with avg_correct < {RELIABILITY_THRESHOLD}")

    # === Step 3: Filter users for sub-sampling (keep performance diversity) ===
    user_seq_counts = long_df.groupby("user_id").size()
    eligible_users = user_seq_counts[user_seq_counts >= min_seq_len].index
    before_count = len(user_stats)
    user_stats = user_stats[user_stats["user_id"].isin(eligible_users)]
    after_count = len(user_stats)
    print(f"ℹ️ Excluded {before_count - after_count} users with fewer than {min_seq_len} interactions")

    # Stratified sampling by stratum
    if sample_frac is None or sample_frac <= 0 or sample_frac>=1:
        sampled_users = user_stats  # Use all eligible users
    else:
        sampled_users = user_stats.groupby("stratum").sample(frac=sample_frac, random_state=42)

    sampled_uids = set(sampled_users["user_id"])

    # Write sampled users for record
    with open(os.path.join(output_dir, "sampled_users.txt"), "w") as f:
        for uid in sorted(sampled_uids):
            f.write(f"{uid}\n")

    # === Step 4: Merge stratum info ===
    long_df = long_df[long_df["user_id"].isin(sampled_uids)]
    long_df = long_df.merge(sampled_users[["user_id", "stratum"]], on="user_id")

    # === Step 5: Write dataset.txt ===
    dataset_path = os.path.join(output_dir, "dataset.txt")
    with open(dataset_path, "w") as f:
        for uid, skill, correct in long_df[["user_id", "skill_id", "correct"]].values:
            f.write(f"{uid} {skill} {correct}\n")

    # === Step 6: Stratified GroupKFold on user IDs ===
    user_ids = sampled_users["user_id"].values
    stratums = sampled_users["stratum"].values
    # Create a consistent UID-to-index map
    all_uids_sorted = sorted(user_ids)

    if k_folds == 1:
        # Custom 90/10 split (not k-fold)
        np.random.seed(42)
        shuffled_uids = np.random.permutation(user_ids)
        n_total = len(shuffled_uids)
        n_train = int(0.9 * n_total)
        train_uids = set(shuffled_uids[:n_train])
        binary_split = ['1' if uid in train_uids else '0' for uid in all_uids_sorted]
        split_file = os.path.join(output_dir, f"dataset_split.txt")
        with open(split_file, "w") as f:
            f.write(" ".join(binary_split) + "\n")
        print(f"✅ Wrote 90/10 train-test split to {split_file}")
    else:
        # K-Fold stratified split
        gkf = GroupKFold(n_splits=k_folds)
        for i, (train_idx, test_idx) in enumerate(gkf.split(X=user_ids, y=stratums, groups=user_ids), 1):
            train_uids = set(user_ids[train_idx])
            # Construct binary split: 1 for train, 0 for test
            binary_split = ['1' if uid in train_uids else '0' for uid in all_uids_sorted]
            split_file = os.path.join(output_dir, f"dataset_split_{i}.txt")
            with open(split_file, "w") as f:
                f.write(" ".join(binary_split) + "\n")
        print(f"✅ Wrote K={k_folds} splits to dataset_split_*.txt")
    print(f"✅ Wrote {dataset_path}")
    print(f"✅ Wrote sampled_users.txt")
    print(f"✅ Wrote K={k_folds} splits to dataset_split_*.txt")


# Example usage
def run_k_fold_cv(data_folder, train_csv, test_csv, k_fold=5):
    sample_frac = 0.35
    min_seq_len = 100
    generate_stratified_kfold_from_xes3g5m(
        train_csv, test_csv, data_folder, k_folds=k_fold, sample_frac=sample_frac, min_seq_len=min_seq_len)

def run_full_data(data_folder, train_csv, test_csv):
    sample_frac = None
    min_seq_len = 100
    k_fold = 1
    generate_stratified_kfold_from_xes3g5m(
        train_csv, test_csv, data_folder, k_folds=k_fold, sample_frac=sample_frac, min_seq_len=min_seq_len)


if __name__ == "__main__":
    DATA_FOLDER = os.path.join("XES3G5M", "question_level")
    TRAIN_CSV = os.path.join(DATA_FOLDER, "test_window_sequences_quelevel.csv")
    TEST_CSV = os.path.join(DATA_FOLDER, "train_valid_sequences_quelevel.csv")
    run_k_fold_cv(DATA_FOLDER, TRAIN_CSV, TEST_CSV, 5)
    # run_full_data(DATA_FOLDER, TRAIN_CSV, TEST_CSV)
