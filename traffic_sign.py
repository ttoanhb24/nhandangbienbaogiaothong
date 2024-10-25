import numpy as np 
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
import os
import sys
import io

data = []
labels = []
classes = 43
cur_path = os.getcwd()

# Kiểm tra và xử lý lỗi encoding nếu cần, nếu không cần có thể bỏ dòng này
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Lấy ảnh và nhãn của chúng
for i in range(classes):
    path = os.path.join(cur_path, 'train', str(i))  # Sử dụng os.path.join để đảm bảo đường dẫn chính xác
    if not os.path.exists(path):
        print(f"Directory not found: {path}")
        continue  # Bỏ qua nếu thư mục không tồn tại

    images = os.listdir(path)
    
    for a in images:
        try:
            image_path = os.path.join(path, a)  # Sử dụng os.path.join để ghép đường dẫn ảnh
            image = Image.open(image_path)
            image = image.resize((30,30))
            image = np.array(image)
            data.append(image)
            labels.append(i)
        except Exception as e:
            print(f"Error loading image {a} from {path}: {e}")

# Chuyển đổi danh sách thành mảng numpy
data = np.array(data)
labels = np.array(labels)

# In ra kích thước dữ liệu đã đọc
print(f"Total images loaded: {len(data)}, Total labels: {len(labels)}")
print(data.shape, labels.shape)

# Chia dữ liệu thành tập huấn luyện và tập kiểm thử
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=43)

# In ra kích thước của các tập dữ liệu
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Chuyển đổi nhãn thành dạng one-hot encoding
y_train = to_categorical(y_train, 43)
y_test = to_categorical(y_test, 43)

# Xây dựng mô hình
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=X_train.shape[1:]))
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(43, activation='softmax'))

# Biên dịch mô hình
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Huấn luyện mô hình
epochs = 15
history = model.fit(X_train, y_train, batch_size=32, epochs=epochs, validation_data=(X_test, y_test))

# Lưu mô hình đã huấn luyện
model.save("my_model.keras")

# Vẽ biểu đồ cho độ chính xác
plt.figure(0)
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()

# Vẽ biểu đồ cho hàm mất mát
plt.figure(1)
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()
