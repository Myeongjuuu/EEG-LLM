<h2>Fine-tuned LLM for Electroencephalography(EEG) data classification</h2>
<p>This is LLM Fine tuning model that classifies four movements (left hand, right hand, tongue, foot) from EEG.</p>
<ul>
  <li>LLM performs its own classification operations based on EEG data.</li>
  <li>We trained gpt-4o model utilizing fine-tuning for better performance.</li>
</ul>
<img width="927" alt="figure1" src="https://github.com/user-attachments/assets/18f66f60-1b33-45af-9929-befd06ef0d55">

<br>
<h2>Requirements</h2>
<p><strong>Python>=3.8,   openai>=1.30.2,   mne>=1.6.1</strong><br>You can install all libraries entering the code: </p>
<blockquote>!pip install -r requirements.txt</blockquote>
<br>
<h2>Data</h2>
<p><strong>Data description</strong> : https://www.bbci.de/competition/iii/desc_IIIa.pdf</p>
  <ul>
    <li>cued motor imagery (multi-class) with 4 classes (left hand, right hand, foot, tongue) three subjects (ranging from quite good to fair performance)</li>
    <li>EEG, 60 channels, 60 trials per class</li>
    <li>performance measure: kappa-coefficient</li>
  </ul>
<br>
<p><strong>Download</strong> : BBCI Competition III (https://www.bbci.de/competition/iii/download/index.html?agree=yes&submit=Submit)</p>

<h3>Prepare subject 1 data locally</h3>
<p>After downloading the subject 1 GDF file (<code>k3b.gdf</code>) and the released true-label file (<code>true_label_k3.txt</code>) into <code>Dataset/raw/</code>, activate the Conda environment and run:</p>
<pre>conda activate eeg-llm
python -m Preprocessing.gdf_to_csv</pre>
<p>The converter writes train, validation, and test CSV/CSP files plus JSONL prompts under <code>Dataset/processed/k3b/label45/</code>. It reproduces the existing binary setup (class 4 vs. classes 1-3 grouped as 5), uses 4-second cue-to-imagery windows, and keeps the released test labels out of the test prompts. The test labels remain in the CSV for local scoring.</p>

<p>To run the classical baselines first, run:</p>
<pre>python -m Task.ML.run_baselines</pre>
<p>This tunes SVM, Random Forest, and MLP with stratified 5-fold cross-validation on the training split, using the parameter grids and accuracy objective from the original ML notebooks. It reports validation and official test metrics, and saves results and fitted models under <code>Dataset/processed/k3b/label45/baselines/</code>. It uses five training-selected PSD features plus two CSP features. The official test split is only used for final reporting.</p>

<br>
<h2>Features</h2>
<p>1) <strong>Power spectral density (PSD)</strong> is computed in 2Hz steps from 4Hz to 36Hz.</p>
<p>For feature selection, <strong>Fisher Ratio</strong> is utilized.</p>
<table style="border-collapse: collapse; width: 100%; data-ke-align="alignLeft">
  <tbody>
        <tr>
            <td style="width: 50%;"><img width="500" alt="fr_label1" src="https://github.com/user-attachments/assets/619a317b-f0d0-4600-9fe7-14da35aec61a"></td>
            <td style="width: 50%;"><img width="500" alt="fr_label2" src="https://github.com/user-attachments/assets/ccb4fc57-7db1-4902-b231-6400043e33ef"></td>
            <td style="width: 50%;"><img width="500" alt="fr_label3" src="https://github.com/user-attachments/assets/9bc6ef09-93ac-4f9a-82c7-c173cacb798e"></td>
            <td style="width: 50%;"><img width="500" alt="fr_label4" src="https://github.com/user-attachments/assets/528634b7-558e-4704-97b2-38dd7b427eea"></td>
        </tr>
  </tbody>
</table>
<p>2) <strong>Common spatial pattern (CSP)</strong> is used to extract spatial features that maximize discriminability between classes.</p>

<br>
<h2>Evaluation</h2>
<p>To compare fine-tuned LLM classifier's performance with traditional ML models, we additionally trained <strong>SVM</strong>, <strong>RF</strong> and <strong>MLP</strong> in the same data and same preprocessing method.</p>
<p>Performance metrics:
<ul>
  <li><strong>Accuracy</strong></li>
  <li><strong>F1 score</strong></li>
  <li><strong>ROC-AUC</strong></li>
</ul>
