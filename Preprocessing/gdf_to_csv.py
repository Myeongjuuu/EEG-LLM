"""Convert BCI Competition IIIa subject 1 GDF to the project's label45 data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from mne.decoding import CSP
from sklearn.model_selection import train_test_split

from Preprocessing.csv_to_json_4o import csv_to_json, csv_to_json_without_label, save_to_jsonl


ROOT = Path(__file__).resolve().parents[1]
CLASS_EVENTS = {"769": 1, "770": 2, "771": 3, "772": 4}
TEST_EVENT = "783"
TRIAL_EVENT = "768"
WINDOW_SIZE = 1000
BANDS = [(start, start + 2) for start in range(4, 36, 2)]


def load_trials(gdf_path: Path, labels_path: Path):
    raw = mne.io.read_raw_gdf(gdf_path, preload=False, verbose="ERROR")
    if raw.info["sfreq"] != 250:
        raise ValueError(f"Expected 250 Hz, found {raw.info['sfreq']} Hz")

    annotations = raw.annotations
    descriptions = np.asarray(annotations.description)
    onsets = np.asarray(annotations.onset)
    starts = np.flatnonzero(descriptions == TRIAL_EVENT)
    labels = np.loadtxt(labels_path, dtype=int).reshape(-1)
    if len(labels) != len(starts):
        raise ValueError(f"Label count {len(labels)} != trial count {len(starts)}")

    epochs, cue_types = [], []
    for start_idx in starts:
        trial_start = onsets[start_idx]
        cue_idxs = np.flatnonzero(
            (onsets >= trial_start + 2.9)
            & (onsets <= trial_start + 3.1)
            & np.isin(descriptions, [*CLASS_EVENTS, TEST_EVENT])
        )
        if len(cue_idxs) != 1:
            raise ValueError(f"Expected one class cue near {trial_start + 3:.2f}s")
        cue_idx = cue_idxs[0]
        first_sample = int(round((onsets[cue_idx] - raw.first_time) * raw.info["sfreq"]))
        epoch = raw.get_data(start=first_sample, stop=first_sample + WINDOW_SIZE)
        if epoch.shape != (len(raw.ch_names), WINDOW_SIZE) or not np.isfinite(epoch).all():
            raise ValueError(f"Invalid 4-second epoch at {trial_start:.2f}s")
        epochs.append(epoch)
        cue_types.append(descriptions[cue_idx])

    supervised = np.isin(cue_types, list(CLASS_EVENTS))
    embedded = np.array([CLASS_EVENTS.get(cue, -1) for cue in cue_types])
    if not np.array_equal(embedded[supervised], labels[supervised]):
        raise ValueError("Official chronological labels do not match GDF training events")
    return raw, np.stack(epochs), labels, np.asarray(cue_types), descriptions, onsets, starts


def select_psd_features(epochs, labels, train_indices, sfreq):
    psds, freqs = mne.time_frequency.psd_array_welch(
        epochs[train_indices], sfreq=sfreq, fmin=4, fmax=36, n_fft=128, verbose="ERROR"
    )
    train_labels = labels[train_indices]
    target, rest = train_labels == 4, train_labels == 5
    ranked = []
    for channel_idx in range(psds.shape[1]):
        for band in BANDS:
            mask = (freqs >= band[0]) & (freqs < band[1])
            values = psds[:, channel_idx, mask].sum(axis=-1)
            pos, neg = values[target], values[rest]
            score = (pos.mean() - neg.mean()) ** 2 / (pos.var() + neg.var() + np.finfo(float).tiny)
            ranked.append((float(score), channel_idx, band))

    chosen, used_channels = [], set()
    for score, channel_idx, band in sorted(ranked, reverse=True):
        if channel_idx not in used_channels:
            chosen.append({"score": score, "raw_channel_index": channel_idx, "band_hz": list(band)})
            used_channels.add(channel_idx)
        if len(chosen) == 5:
            break
    selected_columns = [[i, [tuple(item["band_hz"])]] for i, item in enumerate(chosen)]
    return chosen, selected_columns


def make_trial_feature_matrix(epochs, csp_features, chosen, sfreq):
    raw_channels = [item["raw_channel_index"] for item in chosen]
    psds, freqs = mne.time_frequency.psd_array_welch(
        epochs[:, raw_channels, :], sfreq=sfreq, fmin=4, fmax=36,
        n_fft=128, verbose="ERROR"
    )
    features, names = [], []
    for selected_idx, item in enumerate(chosen):
        low, high = item["band_hz"]
        mask = (freqs >= low) & (freqs < high)
        features.append(psds[:, selected_idx, mask].sum(axis=-1))
        names.append(f"CH{item['raw_channel_index'] + 1:02d}_{low}-{high}Hz")
    names.extend(["csp_0", "csp_1"])
    matrix = np.column_stack([*features, csp_features])
    return matrix, names


def write_partition(out_dir, name, epochs, labels, csp_features, raw_channel_indices,
                    selected_columns, is_test=False):
    channel_names = [f"CH{i + 1:02d}" for i in raw_channel_indices]
    signal = epochs[:, raw_channel_indices, :].transpose(0, 2, 1).reshape(-1, len(channel_names))
    repeated_labels = np.repeat(labels, WINDOW_SIZE)
    frame = pd.DataFrame(signal, columns=channel_names)
    frame["label"] = repeated_labels
    csv_path = out_dir / f"{name}.csv"
    frame.to_csv(csv_path, index=False, float_format="%.8g")

    csp_frame = pd.DataFrame(csp_features, columns=["csp_0", "csp_1"])
    csp_frame["label"] = labels
    csp_path = out_dir / "csp" / f"{name}_features.csv"
    csp_path.parent.mkdir(parents=True, exist_ok=True)
    csp_frame.to_csv(csp_path, index=False, float_format="%.10g")

    data = frame.iloc[:, :-1]
    csp_array = csp_frame.iloc[:, :-1].to_numpy()
    if is_test:
        entries = csv_to_json_without_label(data, csp_array, WINDOW_SIZE, selected_columns)
    else:
        entries = csv_to_json(data, csp_array, WINDOW_SIZE, selected_columns, repeated_labels)
    json_path, jsonl_path = out_dir / "json" / f"{name}.json", out_dir / "jsonl" / f"{name}.jsonl"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
    save_to_jsonl(entries, jsonl_path)
    return {"trials": len(labels), "signal_rows": len(frame), "csv": str(csv_path),
            "csp_csv": str(csp_path), "jsonl": str(jsonl_path)}


def convert(gdf_path, labels_path, out_dir, seed=42):
    mne.set_log_level("ERROR")
    raw, epochs, original_labels, cue_types, descriptions, onsets, starts = load_trials(gdf_path, labels_path)
    train_mask = np.isin(cue_types, list(CLASS_EVENTS))
    test_mask = cue_types == TEST_EVENT
    if train_mask.sum() != 180 or test_mask.sum() != 180:
        raise ValueError(f"Expected 180 train and 180 test trials; found {train_mask.sum()} and {test_mask.sum()}")

    # Match the repository's existing label45 experiment: class 4 vs classes 1-3.
    labels = np.where(original_labels == 4, 4, 5)
    labeled_indices = np.flatnonzero(train_mask)
    train_indices, val_indices = train_test_split(
        labeled_indices, test_size=0.2, random_state=seed, stratify=labels[labeled_indices]
    )
    test_indices = np.flatnonzero(test_mask)

    csp = CSP(n_components=2, reg=None, log=True, norm_trace=False)
    csp.fit(epochs[train_indices], labels[train_indices])
    csp_features = csp.transform(epochs)
    chosen, selected_columns = select_psd_features(epochs, labels, train_indices, raw.info["sfreq"])
    selected_channels = [item["raw_channel_index"] for item in chosen]
    trial_features, trial_feature_names = make_trial_feature_matrix(
        epochs, csp_features, chosen, raw.info["sfreq"]
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    partitions = {}
    for name, indices, is_test in [
        ("train", train_indices, False), ("val", val_indices, False), ("test", test_indices, True)
    ]:
        partitions[name] = write_partition(
            out_dir, name, epochs[indices], labels[indices], csp_features[indices],
            selected_channels, selected_columns, is_test
        )
        feature_dir = out_dir / "ml_features"
        feature_dir.mkdir(parents=True, exist_ok=True)
        feature_frame = pd.DataFrame(trial_features[indices], columns=trial_feature_names)
        feature_frame["label"] = labels[indices]
        feature_path = feature_dir / f"{name}.csv"
        feature_frame.to_csv(feature_path, index=False, float_format="%.10g")
        partitions[name]["ml_features_csv"] = str(feature_path)

    metadata = {
        "source_gdf": str(gdf_path), "source_test_labels": str(labels_path),
        "sample_rate_hz": raw.info["sfreq"], "samples_per_trial": WINDOW_SIZE,
        "epoch_window": "3.0s to 7.0s after trial onset (cue through imagery)",
        "binary_target": {"4": "original class 4", "5": "original classes 1, 2, and 3"},
        "official_labels_used_for_test": True, "train_validation_seed": seed,
        "raw_channels_selected_for_psd": [
            {**item, "channel_name": raw.ch_names[item["raw_channel_index"]]} for item in chosen
        ],
        "selected_columns_for_project_pipeline": selected_columns,
        "trial_feature_columns": [*trial_feature_names, "label"],
        "csp_fit_on": "train split only; all 60 source channels",
        "artifact_marked_trials_in_source": sum(
            int(np.any((onsets >= onsets[i] - 0.001) & (onsets <= onsets[i] + 0.001)
                       & (descriptions == "1023"))) for i in starts
        ),
        "partitions": partitions,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gdf", type=Path, default=ROOT / "Dataset/raw/k3b.gdf")
    parser.add_argument("--labels", type=Path, default=ROOT / "Dataset/raw/true_label_k3.txt")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "Dataset/processed/k3b/label45")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(convert(args.gdf, args.labels, args.out_dir, args.seed), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
