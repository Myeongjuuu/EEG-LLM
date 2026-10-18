<h2>Preprocessing</h2>
<p>The files in this folder are scripts designed to perform feature extraction on scaled and Laplacian-filtered CSV data, and then save the results in JSON and JSONL formats.</p>
<ul>
  <li><strong>feature_extraction.py</strong>: This script loads EEG data and computes Welch PSD band-power features for selected channels and frequency bands.</li>
  <li><strong>csv_to_json_4o.py</strong>: This script transforms extracted features and labels into JSON suitable for GPT-4o, using functions from <code>feature_extraction.py</code>.</li>
  <li><strong>csv_to_json_davinci.py</strong>: This script transforms extracted features and labels into JSON suitable for davinci-002, using the same feature module.</li>
  <li><strong>preprocessing.py</strong>: This script provides a pipeline for loading EEG data, extracting features, and saving JSON and JSONL files.</li>
</ul>

<p>The top-level <code>features/</code> directory contains the repository's CSP and PSD/Fisher analysis code.</p>

<img width=100% alt="preprocessing" src="https://github.com/user-attachments/assets/1f334d41-e110-4053-8dcf-7c6693431b7d">
