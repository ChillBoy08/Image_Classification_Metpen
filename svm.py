import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Step 1: Membaca gambar dan label kelas
def load_images_from_folder(folder):
    images = []
    labels = []
    label_map = {'Healthy': 0, 'Bacterial Canker': 1, 'Powdery Mildew': 2, 'Anthracnose': 3, 'Spider Mite': 4}
    
    for subfolder in os.listdir(folder):
        label = label_map[subfolder]  # Menggunakan kamus untuk mengonversi nama subfolder menjadi label kelas
        subfolder_path = os.path.join(folder, subfolder)
        for filename in os.listdir(subfolder_path):
            img_path = os.path.join(subfolder_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Ubah menjadi grayscale
            images.append(img)
            labels.append(label)
    return images, labels


# Ubah folder menjadi direktori tempat gambar-gambar Anda disimpan
folder_path = r"D:\Perkuliahan\semester 6\Metodologi Penelitian\cnn&svm\penyakit_daun_mangga"
images, labels = load_images_from_folder(folder_path)

# Step 2: Ekstraksi Fitur (gunakan metode HOG sebagai contoh)
hog_features = []
for img in images:
    fd, hog_image = hog(img, orientations=8, pixels_per_cell=(4, 4), cells_per_block=(1, 1), visualize=True)
    hog_features.append(fd)

X_hog = np.array(hog_features)
y = np.array(labels)

# Step 3: Pastikan jumlah kelas lebih dari satu
if len(np.unique(y)) < 2:
    raise ValueError("The number of classes has to be greater than one.")

# Step 4: Pembagian Dataset menjadi data pelatihan dan data pengujian
X_train, X_test, y_train, y_test = train_test_split(X_hog, y, test_size=0.2, random_state=42)

# Step 5: Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 6: Pelatihan SVM
clf = svm.SVC(kernel='linear', C=1.0)
clf.fit(X_train, y_train)

# Step 7: Prediksi pada data pengujian
y_pred = clf.predict(X_test)

# Step 8: Evaluasi Model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(accuracy * 100))
print(np.unique(y))
