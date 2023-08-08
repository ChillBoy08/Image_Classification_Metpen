import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import to_categorical

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
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)  # Baca gambar dalam format warna (3 channel)
            img = cv2.resize(img, (150, 150))  # Ubah ukuran gambar menjadi 150x150 piksel
            images.append(img)
            labels.append(label)
    return images, labels

# Ubah folder menjadi direktori tempat gambar-gambar Anda disimpan
folder_path = r"D:\Perkuliahan\semester 6\Metodologi Penelitian\cnn&svm\penyakit_daun_mangga"
images, labels = load_images_from_folder(folder_path)

# Step 2: Preprocessing Data
X = np.array(images)
y = np.array(labels)

# Normalisasi nilai piksel dari 0-255 menjadi 0-1
X = X / 255.0

# One-hot encode label kelas
y = to_categorical(y, num_classes=8)

# Step 3: Pembagian Dataset menjadi data pelatihan dan data pengujian
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Membangun Model CNN
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(5, activation='softmax'))  # Jumlah kelas = 5

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Step 5: Melatih Model
model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test))

# Step 6: Evaluasi Model
accuracy = model.evaluate(X_test, y_test)[1]
print("Accuracy: {:.2f}%".format(accuracy * 100))
