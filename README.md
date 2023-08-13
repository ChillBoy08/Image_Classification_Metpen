# Image_Classification_Metpen
Tugas Metode Penelitian dengan judul 
```text
"Perbandingan Model Convolutional Neural Network(CNN) dan Model Support Vector Machine(SVM) untuk Mendeteksi Penyakit Daun Mangga".
```

Saya mengambil Data set dari [Kaggle](https://www.kaggle.com/datasets/aryashah2k/mango-leaf-disease-dataset)

Menggunakan Python library CNN
```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout, BatchNormalization
from tensorflow.keras import regularizers
```

Menggunakan Python Library SVM
```python
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
```

Data set yang di training
```ts
Class Anthracnose: 348 samples
Class Bacterial Canker: 354 samples
Class Cutting Weevil: 322 samples
Class Die Back: 343 samples
Class Gall Midge: 350 samples
Class Healthy: 353 samples
Class Powdery Mildew: 365 samples
Class Sooty Mould: 365 samples
```

Data set yang di Valid
```ts
Class Anthracnose: 78 samples
Class Bacterial Canker: 67 samples
Class Cutting Weevil: 95 samples
Class Die Back: 84 samples
Class Gall Midge: 75 samples
Class Healthy: 70 samples
Class Powdery Mildew: 70 samples
Class Sooty Mould: 61 samples
```



