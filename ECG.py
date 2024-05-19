from PIL import Image, ImageTk
import wfdb
import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split
from biosppy.signals import ecg
from scipy.signal import find_peaks
from scipy.fftpack import dct
import pywt
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
from tkinter import Tk, Label, Button
import statsmodels.api as sm
import scipy.fftpack

# Paths to the ECG data for the subjects
subjects = {
    's1': r'subjects\p_156\s0299lre',
    's2': r'subjects\p_165\s0322lre',
    's3': r'subjects\p_180\s0374lre',
    's4': r'subjects\p_184\s0363lre',
    's5': r'subjects\p_166\s0275lre',
    's6': r'subjects\sub_150\s0287lre',
    's7': r'subjects\sub260\s0496_re'
}


def load_ecg(subject_path):
    record = wfdb.rdrecord(subject_path)
    ecg_signal = record.p_signal[:, 0]
    return ecg_signal

def check_channels(ecg_signal):
    if ecg_signal.ndim == 1:
        return "Single-channel"
    elif ecg_signal.ndim == 2:
        num_channels = ecg_signal.shape[1]
        return f"Multi-channel with {num_channels} channels"
    else:
        return "Unknown format"


for subject, path in subjects.items():
    ecg_signal = load_ecg(path)
    channel_info = check_channels(ecg_signal)
    print(f"Subject {subject}: {channel_info}")


def preprocess_with_drift_correction(ecg_signal):
    corrected_ecg = ecg.ecg(signal=ecg_signal, sampling_rate=1000, show=False)
    corrected_signal = corrected_ecg['filtered']
    nyquist = 0.5 * 1000
    low = 0.5 / nyquist
    high = 45 / nyquist
    b, a = butter(1, [low, high], btype='band')
    filtered_ecg = filtfilt(b, a, corrected_signal)

    return filtered_ecg


ecg_data = {subject: preprocess_with_drift_correction(load_ecg(path)) for subject, path in subjects.items()}

from scipy.signal import find_peaks

from scipy.signal import find_peaks


def segment_ecg(ecg_signal, segment_length=600, threshold=None):
    # Find R-peaks
    peaks, _ = find_peaks(ecg_signal, distance=200, height=threshold)
    segments = []
    for peak in peaks:
        pre_peak_length = segment_length // 3
        post_peak_length = segment_length - pre_peak_length
        start = max(0, peak - pre_peak_length)
        end = start + segment_length
        if end <= len(ecg_signal):
            segments.append(ecg_signal[start:end])
    return segments



segmented_data = {subject: segment_ecg(ecg, threshold=0.2) for subject, ecg in ecg_data.items()}

segment_counts = {subject: len(segments) for subject, segments in segmented_data.items()}

for subject, count in segment_counts.items():
    print(f"Subject {subject} has {count} segments.")


# def preprocess_using_ACDCT(filtered_signal):
#     sig = np.array(filtered_signal)
#     AC = sm.tsa.acf(sig, nlags=1000)
#     DCT = scipy.fftpack.dct(AC, type=1)
#     return DCT
#
#
def preprocess_using_wavelet(filtered_signal):
    wavelet = 'db8'
    level = 5
    coeffs = pywt.wavedec(filtered_signal, wavelet, level=level)
    for i in range(1, level):
        coeffs[i] = np.zeros_like(coeffs[i])  # Remove high freq components and set to 0
    res = pywt.waverec(coeffs, wavelet)  # Wave reconstruction
    return res
#
#
# def extract_features(segment, method='dct'):
#     if method == 'dct':
#         return preprocess_using_ACDCT(segment)
#     elif method == 'wavelet':
#         return preprocess_using_wavelet(segment)
#     else:
#         raise ValueError("Unsupported method. Use 'dct' or 'wavelet'.")


#
def extract_features(segment, method='dct'):
    if method == 'dct':
        return dct(segment, norm='ortho')
    elif method == 'wavelet':
        coeffs, _ = pywt.dwt(segment, 'db1')
        return coeffs


# Extract features for all segments
features_dct = {subject: [extract_features(seg, method='dct') for seg in segments] for subject, segments in
                segmented_data.items()}
features_wavelet = {subject: [extract_features(seg, method='wavelet') for seg in segments] for subject, segments in
                    segmented_data.items()}


def prepare_data(features):
    X, y = [], []
    for subject, seg_features in features.items():
        X.extend(seg_features)
        y.extend([subject] * len(seg_features))
    return np.array(X), np.array(y)


X_dct, y_dct = prepare_data(features_dct)
X_wavelet, y_wavelet = prepare_data(features_wavelet)

X_train_dct, X_test_dct, y_train_dct, y_test_dct = train_test_split(X_dct, y_dct, test_size=0.2)
X_train_wavelet, X_test_wavelet, y_train_wavelet, y_test_wavelet = train_test_split(X_wavelet, y_wavelet, test_size=0.2)

model_dir = 'models'
os.makedirs(model_dir, exist_ok=True)
svm_dct_path = os.path.join(model_dir, 'svm_dct_model.pkl')
svm_wavelet_path = os.path.join(model_dir, 'svm_wavelet_model.pkl')
rf_dct_path = os.path.join(model_dir, 'rf_dct_model.pkl')
rf_wavelet_path = os.path.join(model_dir, 'rf_wavelet_model.pkl')

# Train SVM
svm_dct = SVC(kernel='rbf', C=1, probability=True)
svm_wavelet = SVC(kernel='rbf', C=1, probability=True)
svm_dct.fit(X_train_dct, y_train_dct)
svm_wavelet.fit(X_train_wavelet, y_train_wavelet)

# Train Random Forest
rf_dct = RandomForestClassifier(n_estimators=100)
rf_wavelet = RandomForestClassifier(n_estimators=100)
rf_dct.fit(X_train_dct, y_train_dct)
rf_wavelet.fit(X_train_wavelet, y_train_wavelet)

# Save the models
joblib.dump(svm_dct, svm_dct_path)
joblib.dump(svm_wavelet, svm_wavelet_path)
joblib.dump(rf_dct, rf_dct_path)
joblib.dump(rf_wavelet, rf_wavelet_path)


svm_dct_accuracy = svm_dct.score(X_test_dct, y_test_dct)
svm_wavelet_accuracy = svm_wavelet.score(X_test_wavelet, y_test_wavelet)
rf_dct_accuracy = rf_dct.score(X_test_dct, y_test_dct)
rf_wavelet_accuracy = rf_wavelet.score(X_test_wavelet, y_test_wavelet)

print(f"SVM DCT Accuracy: {svm_dct_accuracy}")
print(f"SVM Wavelet Accuracy: {svm_wavelet_accuracy}")
print(f"Random Forest DCT Accuracy: {rf_dct_accuracy}")
print(f"Random Forest Wavelet Accuracy: {rf_wavelet_accuracy}")


svm_dct = joblib.load(svm_dct_path)
svm_wavelet = joblib.load(svm_wavelet_path)
rf_dct = joblib.load(rf_dct_path)
rf_wavelet = joblib.load(rf_wavelet_path)


def identify_subject(ecg_segment, method='dct', threshold=0.5):
    features = extract_features(ecg_segment, method=method)
    if method == 'dct':
        svm_pred = svm_dct.predict([features])
        rf_pred = rf_dct.predict([features])
        svm_confidence = svm_dct.predict_proba([features])[0]
        rf_confidence = rf_dct.predict_proba([features])[0]
    else:
        svm_pred = svm_wavelet.predict([features])
        rf_pred = rf_wavelet.predict([features])
        svm_confidence = svm_wavelet.predict_proba([features])[0]
        rf_confidence = rf_wavelet.predict_proba([features])[0]

    # Get the class with the highest confidence for each model
    svm_max_confidence = max(svm_confidence)
    rf_max_confidence = max(rf_confidence)

    if svm_max_confidence >= rf_max_confidence:
        final_pred = svm_pred[0]
        final_confidence = svm_max_confidence
    else:
        final_pred = rf_pred[0]
        final_confidence = rf_max_confidence

    if final_confidence > threshold:
        return final_pred, final_confidence
    else:
        return 'unidentified', final_confidence


subjects_test = {
    's1': r'subjects\p_155\s0301lre',
    's2': r'subjects\sub_170\s0274lre',
}
ecg_data_test = {subject: preprocess_with_drift_correction(load_ecg(path)) for subject, path in subjects_test.items()}
segmented_data_test = {subject: segment_ecg(ecg, threshold=0.2) for subject, ecg in ecg_data_test.items()}
ecg_segment_test = segmented_data_test['s1'][0]

photos = {
    's1': r'imgs/photo1.jpg',
    's2': r'imgs/photo2.jpg',
    's3': r'imgs/photo3.jpg',
    's4': r'imgs/photo4.jpg',
    's5': r'imgs/photo5.jpg',
    's6': r'imgs/photo6.jpg',
    's7': r'imgs/photo7.jpg',
    'notFound': r'imgs/image.png',

}


# Load images
def load_image(subject, size=(500, 500)):
    photo_path = photos.get(subject, 'imgs/photo3.jpg')
    image = Image.open(photo_path)
    image = image.resize(size, Image.ANTIALIAS)  # Resize the image
    photo = ImageTk.PhotoImage(image)
    return photo


def show_photo(subject):
    photo = load_image(subject)
    label.config(image=photo)
    label.image = photo  # Keep a reference to the image to prevent garbage collection


# Function to simulate ECG segment acquisition and identification for an identified subject
def on_ecg_acquired_identified():
    ecg_segment = segmented_data['s3'][129]  # Use a segment from subject s10 as an example
    identified_subject, confidence = identify_subject(ecg_segment, method='dct', threshold=0.7)
    print(f"Identified Subject: {identified_subject} with confidence {confidence}")
    if identified_subject != 'unidentified':
        show_photo(identified_subject)  # Show the default image for unidentified subjects
    else:
        show_photo('notFound')


# Function to simulate ECG segment acquisition and identification for an unidentified subject
def on_ecg_acquired_unidentified():
    ecg_segment = segmented_data_test['s2'][0]  # Random segment that should not be identified
    identified_subject, confidence = identify_subject(ecg_segment, method='dct', threshold=0.7)
    print(f"unidentified Subject: {identified_subject} with confidence {confidence}")
    if identified_subject == 'unidentified':
        show_photo('notFound')  # Show the default image for unidentified subjects
    else:
        show_photo(identified_subject)


# Create the main window
root = Tk()
root.title("ECG-based Photo Lock")

# Create and place a label to display the photo
label = Label(root)
label = Label(root)
label.pack()

# Add a button to simulate ECG segment acquisition for identified subjects
button_identified = Button(root, text="Test with Identified Segment", command=on_ecg_acquired_identified)
button_identified.pack()

# Add a button to simulate ECG segment acquisition for unidentified subjects
button_unidentified = Button(root, text="Test with Unidentified Segment", command=on_ecg_acquired_unidentified)
button_unidentified.pack()

# Start the Tkinter event loop
root.mainloop()
