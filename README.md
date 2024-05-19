# ECG based Photo Lock

The ECG-based Photo Lock is a biometric authentication system that uses electrocardiogram (ECG) signals to identify and authenticate users. The system processes ECG data through several stages, including baseline drift correction, bandpass filtering, and segmentation to isolate individual heartbeats. Each heartbeat segment is analyzed using feature extraction techniques like Discrete Cosine Transform (DCT) or wavelet transformation. Machine learning models, such as Support Vector Machines (SVM) and Random Forest classifiers, are trained on these features to recognize individual users based on their unique ECG patterns. When a user attempts to access the system, their ECG signal is captured, processed, and compared against stored models to verify their identity and unlock a photo associated with their profile.

### **Function Definition: `preprocess_with_drift_correction(ecg_signal)`**

This function is designed to preprocess ECG signals by removing baseline drift and applying bandpass filtering. The preprocessing steps help in enhancing the signal quality by removing noise and unwanted components, making it more suitable for subsequent analysis such as feature extraction and classification.

### **Steps and Explanation**

1. **Baseline Drift Correction**:
    - **Purpose**: Baseline drift refers to the slow variations in the ECG signal baseline caused by respiration, body movements, and other factors. Removing this drift is crucial for accurate ECG analysis.
    - **Implementation**: The **`ecg()`** function from the **`biosppy.signals.ecg`** module is used. This function processes the ECG signal and includes a baseline drift removal step. The **`corrected_signal`** is obtained from the 'filtered' key of the dictionary returned by the **`ecg()`** function.
    - **Parameters**:
        - **`signal=ecg_signal`**: The raw ECG signal to be processed.
        - **`sampling_rate=1000`**: The sampling rate of the ECG signal in Hz.
        - **`show=False`**: Disables the automatic plotting of the signal.
2. **Bandpass Filtering**:
    - **Purpose**: Bandpass filtering is applied to retain only the frequency components of interest (0.5 Hz to 45 Hz in this case) and remove the rest. This helps in reducing noise and improving the quality of the signal for analysis.
    - **Implementation**:
        - **Nyquist Frequency**: Calculated as half of the sampling rate (1000 Hz), which is 500 Hz.
        - **Low and High Cutoff Frequencies**: The desired cutoff frequencies for the bandpass filter are set to 0.5 Hz and 45 Hz. These frequencies are normalized by the Nyquist frequency to get **`low`** and **`high`** values.
        - **Butterworth Filter**: A Butterworth filter of order 1 is designed using **`butter()`**, and the filter coefficients **`b`** and **`a`** are obtained.
        - **Filtering**: The **`filtfilt()`** function is used to apply the filter to the signal. This function performs forward and backward filtering to ensure zero phase distortion.
    - **Parameters**:
        - **`btype='band'`**: Specifies that a bandpass filter is to be created.
        - **`filtered_ecg`**: The output signal after applying the bandpass filter.
    - **Return**: The function returns the preprocessed ECG signal, which has undergone baseline drift correction and bandpass filtering.

### **Segmenting ECG Signals and Extracting Features**

### **Segmenting ECG Signal and Extracting Features**

### **Function Definition: `segment_ecg(ecg_signal, segment_length=600, threshold=None)`**

This function segments the ECG signal into smaller parts based on the detected R-peaks. Segmentation is crucial for feature extraction as it allows focusing on relevant parts of the ECG signal.

### **Steps and Explanation**

1. **Find R-peaks**:
    - **Purpose**: R-peaks are the most prominent features in an ECG signal and are used as reference points for segmenting the signal.
    - **Implementation**: The **`find_peaks()`** function from **`scipy.signal`** is used to detect the R-peaks in the ECG signal. The **`distance`** parameter ensures that consecutive peaks are at least 200 samples apart, and the **`height`** parameter sets a threshold for peak detection.
2. **Segment the Signal**:
    - **Purpose**: Segments the ECG signal around the detected R-peaks to create smaller windows of interest.
    - **Implementation**:
        - **Segment Length**: The **`segment_length`** parameter defines the total length of each segment.
        - **Pre-peak and Post-peak Lengths**: The segment is divided into two parts: one before and one after the R-peak. This is done to ensure the R-peak is centrally located within the segment.
        - **Segment Extraction**: For each detected peak, a segment starting from **`start`** to **`end`** is extracted. The segment is added to the **`segments`** list if it fits within the bounds of the signal.
3. **Return**:
    - **Segments**: The function returns a list of segments extracted from the ECG signal

### **Feature Extraction**

### **Function Definition: `preprocess_using_ACDCT(filtered_signal)`**

This function preprocesses the ECG segments using Autocorrelation and Discrete Cosine Transform (DCT).

1. **Autocorrelation**:
    - **Purpose**: Measures the similarity between the signal and a delayed version of itself. This helps in emphasizing repetitive patterns.
    - **Implementation**: The **`acf()`** function from the **`statsmodels`** library is used to compute the autocorrelation function up to 1000 lags.
2. **Discrete Cosine Transform (DCT)**:
    - **Purpose**: Converts the autocorrelation function to the frequency domain, helping in feature extraction by representing the signal in terms of its frequency components.
    - **Implementation**: The **`dct()`** function from **`scipy.fftpack`** is used to apply DCT on the autocorrelation function.
3. **Return**: The function returns the DCT coefficients of the autocorrelation function.

### **Function Definition: `preprocess_using_wavelet(filtered_signal)`**

This function preprocesses the ECG segments using Wavelet Transform.

1. **Wavelet Transform**:
    - **Purpose**: Decomposes the signal into different frequency components using wavelets. This helps in capturing both time and frequency information.
    - **Implementation**: The **`wavedec()`** function from the **`pywt`** library is used to perform a 5-level wavelet decomposition using the 'db8' wavelet.
    - **High-frequency Component Removal**: Sets the high-frequency components to zero, emphasizing the low-frequency components.
    - **Reconstruction**: The **`waverec()`** function is used to reconstruct the signal from the modified wavelet coefficients.
2. **Return**: The function returns the reconstructed signal with the high-frequency components removed.

### **Function Definition: `extract_features(segment, method='dct')`**

This function extracts features from a given ECG segment using either DCT or Wavelet Transform.

1. **Method Selection**:
    - **DCT**: Calls **`preprocess_using_ACDCT()`** if the method is 'dct'.
    - **Wavelet**: Calls **`preprocess_using_wavelet()`** if the method is 'wavelet'.
    - **Error Handling**: Raises a **`ValueError`** if an unsupported method is provided.
2. **Return**: The function returns the features extracted from the segment using the specified method.

### **Data Preparation for Classification**

### **Function Definition: `prepare_data(features)`**

This function organizes the features and labels into a format suitable for training machine learning models. Specifically, it converts the extracted features into input arrays and assigns corresponding labels.

### **Steps and Explanation**

1. **Initialize Lists**:
    - **X**: An empty list to store feature vectors.
    - **y**: An empty list to store labels corresponding to the feature vectors.
2. **Iterate Through Subjects**:
    - **Purpose**: Loop through each subject and their associated features.
    - **Implementation**:
        - **seg_features**: Features extracted for each segment of the subject.
        - **subject**: Identifier for the subject (used as the label).
3. **Extend Feature List**:
    - **Purpose**: Append all feature vectors from **`seg_features`** to **`X`**.
    - **Implementation**: **`X.extend(seg_features)`** adds all elements of **`seg_features`** to **`X`**.
4. **Extend Label List**:
    - **Purpose**: Append the subject label repeated for each segment to **`y`**.
    - **Implementation**: **`[subject] * len(seg_features)`** creates a list with the subject repeated for each segment.
5. **Return Arrays**:
    - **Convert to NumPy Arrays**: **`np.array(X)`** and **`np.array(y)`** convert the lists **`X`** and **`y`** into NumPy arrays for compatibility with machine learning algorithms.

### **Data Preparation**

After defining the **`prepare_data`** function, it is used to prepare the data for both DCT and Wavelet features.

- **`features_dct`**: Dictionary containing DCT features for each subject.
- **`features_wavelet`**: Dictionary containing Wavelet features for each subject.
- **Purpose**: These lines prepare the data by extracting features and labels, converting them into arrays suitable for machine learning.

### **Splitting Data into Training and Testing Sets**

- **Purpose**: To evaluate the performance of machine learning models, the data is split into training and testing sets.
- **Function Used**: **`train_test_split`** from **`sklearn.model_selection`**.
    - **Parameters**:
        - **`X_dct`**, **`y_dct`**: Feature and label arrays for DCT features.
        - **`X_wavelet`**, **`y_wavelet`**: Feature and label arrays for Wavelet features.
        - **`test_size=0.2`**: Specifies that 20% of the data should be used for testing, and 80% for training.
- **Outputs**:
    - **`X_train_dct`**, **`X_test_dct`**, **`y_train_dct`**, **`y_test_dct`**: Training and testing sets for DCT features.
    - **`X_train_wavelet`**, **`X_test_wavelet`**, **`y_train_wavelet`**, **`y_test_wavelet`**: Training and testing sets for Wavelet features.

### **Model Training and Evaluation**

### **Model Training**

The provided code trains Support Vector Machine (SVM) and Random Forest classifiers using DCT and Wavelet features, respectively.

- **Model Directories**: Creates a directory named 'models' to store trained models. If the directory already exists, it will not raise an error (**`exist_ok=True`**).
- **Model Paths**: Defines paths for saving the trained models (**`svm_dct_model.pkl`**, **`svm_wavelet_model.pkl`**, **`rf_dct_model.pkl`**, **`rf_wavelet_model.pkl`**).
- **Support Vector Machine (SVM)**:
    - Two SVM classifiers are trained: one for DCT features (**`svm_dct`**) and another for Wavelet features (**`svm_wavelet`**).
    - **Kernel**: Radial Basis Function (RBF) kernel is used.
    - **Regularization Parameter (C)**: Set to 1.
    - **Probability**: **`probability=True`** enables probability estimates.
- **Random Forest**:
    - Two Random Forest classifiers are trained: one for DCT features (**`rf_dct`**) and another for Wavelet features (**`rf_wavelet`**).
    - **Number of Estimators**: Set to 100.

### **Model Saving**

- **Purpose**: Saves the trained models to disk using **`joblib.dump()`** function.
- **File Extensions**: **`.pkl`** extension is commonly used for serialized Python objects.

### **Model Evaluation**

- **Purpose**: Evaluates the performance of trained classifiers on the test data.
- **Accuracy Calculation**: The **`score()`** method calculates the accuracy of each classifier by comparing predictions with the true labels.
- **Print Accuracy**: Prints the accuracy of SVM and Random Forest classifiers trained on DCT and Wavelet features.

### **Subject Identification Function**

### **Function Definition: `identify_subject(ecg_segment, method='dct', threshold=0.5)`**

This function predicts the subject identity based on extracted features from an ECG segment using either Discrete Cosine Transform (DCT) or Wavelet Transform methods. It utilizes previously trained SVM and Random Forest classifiers to make predictions and assesses the confidence level of each prediction.

### **Parameters:**

- **`ecg_segment`**: The ECG segment for which subject identification is performed.
- **`method='dct'`**: Specifies the feature extraction method to be used. Default is DCT.
- **`threshold=0.5`**: Confidence threshold for subject identification. Default is 0.5.

### **Steps and Explanation:**

1. **Feature Extraction**:
    - **Purpose**: Extracts features from the input ECG segment using the specified method.
    - **Implementation**: Calls the **`extract_features`** function with the specified method to obtain features.
2. **Prediction**:
    - **SVM and Random Forest Predictions**:
        - For DCT method:
            - Predicts subject labels using the trained SVM and Random Forest classifiers for DCT features (**`svm_dct`** and **`rf_dct`**).
        - For Wavelet method:
            - Predicts subject labels using the trained SVM and Random Forest classifiers for Wavelet features (**`svm_wavelet`** and **`rf_wavelet`**).
    - **Confidence Calculation**:
        - Calculates the prediction probabilities using **`predict_proba`**.
        - Finds the maximum confidence for each model.
3. **Final Prediction**:
    - **Comparison of Confidence**:
        - Compares the maximum confidence of SVM and Random Forest models.
        - Selects the prediction with higher confidence.
    - **Threshold Check**:
        - Checks if the final confidence is above the specified threshold.
        - If yes, returns the predicted subject label and confidence.
        - If not, returns 'unidentified' label and confidence.

## **Test with Identified Segment**
![Untitled](https://github.com/El-Sayed-Mustafa/ECG-based-photo/assets/110793510/ddebfe9f-64e6-42eb-85c2-e1b154a5fa8b)


## Test with Unidentified Segment
![Untitled (2)](https://github.com/El-Sayed-Mustafa/ECG-based-photo/assets/110793510/351b9119-ea9a-460c-9035-e195b43ff09d)

