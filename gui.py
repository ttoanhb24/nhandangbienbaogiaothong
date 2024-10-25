import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import sys
import io
import numpy as np
from keras.models import load_model
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
# Tải mô hình đã huấn luyện để phân loại biển báo
model = load_model('my_model.keras')

# Đảm bảo đầu ra ở định dạng UTF-8
if sys.stdout.encoding != 'utf-8':
    print("Cảnh báo: Đầu ra không phải ở định dạng UTF-8. Một số ký tự có thể không hiển thị đúng.")

# Từ điển để gán nhãn cho tất cả các lớp biển báo giao thông
classes = {
    1: 'Giới hạn tốc độ (20km/h)',
    2: 'Giới hạn tốc độ (30km/h)',
    3: 'Giới hạn tốc độ (50km/h)',
    4: 'Giới hạn tốc độ (60km/h)',
    5: 'Giới hạn tốc độ (70km/h)',
    6: 'Giới hạn tốc độ (80km/h)',
    7: 'Hết giới hạn tốc độ (80km/h)',
    8: 'Giới hạn tốc độ (100km/h)',
    9: 'Giới hạn tốc độ (120km/h)',
    10: 'Cấm vượt',
    11: 'Cấm vượt xe có trọng tải trên 3.5 tấn',
    12: 'Giao nhau với đường không ưu tiên',
    13: 'Đường ưu tiên',
    14: 'Giao nhau với đường ưu tiên',
    15: 'Dừng lại',
    16: 'Cấm phương tiện',
    17: 'Cấm phương tiện > 3.5 tấn',
    18: 'Cấm đi ngược chiều',
    19: 'Cảnh báo chung',
    20: 'Đường cong nguy hiểm bên trái',
    21: 'Đường cong nguy hiểm bên phải',
    22: 'Đường cong nguy hiểm liên tiếp',
    23: 'Đường gồ ghề',
    24: 'Đường trơn trượt',
    25: 'Đường hẹp bên phải',
    26: 'Công trình giao thông',
    27: 'Biển báo giao thông',
    28: 'Người đi bộ',
    29: 'Trẻ em qua đường',
    30: 'Xe đạp qua đường',
    31: 'Cảnh giác với băng tuyết',
    32: 'Động vật hoang dã qua đường',
    33: 'Hết giới hạn tốc độ + cấm vượt',
    34: 'Rẽ phải phía trước',
    35: 'Rẽ trái phía trước',
    36: 'Chỉ đi thẳng',
    37: 'Đi thẳng hoặc rẽ phải',
    38: 'Đi thẳng hoặc rẽ trái',
    39: 'Giữ bên phải',
    40: 'Giữ bên trái',
    41: 'Vòng xuyến bắt buộc',
    42: 'Hết cấm vượt',
    43: 'Hết cấm vượt xe > 3.5 tấn'
}

# Khởi tạo giao diện
top = tk.Tk()
top.geometry('800x600')
top.title('Nhận dạng biển báo giao thông')
top.configure(background='#ffffff')

label = Label(top, background='#ffffff', font=('arial', 15, 'bold'))
sign_image = Label(top)

def classify(file_path):
    try:
        image = Image.open(file_path)
        image = image.resize((30, 30))
        image = np.array(image)
        image = np.expand_dims(image, axis=0)  # Thêm kích thước batch

        # Dự đoán lớp biển báo
        pred_probabilities = model.predict(image)[0]
        pred = pred_probabilities.argmax(axis=-1)
        sign = classes[pred + 1]

        # Hiển thị biển báo dự đoán
        label.configure(foreground='#011638', text=sign)
        print(sign.encode('utf-8', 'replace').decode('utf-8'))  # Ghi nhật ký dự đoán
    except Exception as e:
        print(f"Lỗi trong quá trình phân loại: {e}")

def show_classify_button(file_path):
    classify_b = Button(top, text="Nhận dạng", command=lambda: classify(file_path), padx=10, pady=5)
    classify_b.configure(background='#c71b20', foreground='white', font=('arial', 10, 'bold'))
    classify_b.place(relx=0.79, rely=0.46)

def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width()/2.25), (top.winfo_height()/2.25)))
        im = ImageTk.PhotoImage(uploaded)

        sign_image.configure(image=im)
        sign_image.image = im
        label.configure(text='')
        show_classify_button(file_path)
    except Exception as e:
        print(f"Lỗi khi tải ảnh lên: {e}")

upload = Button(top, text="Tải lên hình ảnh:", command=upload_image, padx=10, pady=5)
upload.configure(background='#c71b20', foreground='white', font=('arial', 10, 'bold'))

upload.pack(side=BOTTOM, pady=50)
sign_image.pack(side=BOTTOM, expand=True)
label.pack(side=BOTTOM, expand=True)

# Tiêu đề giao diện
heading = Label(top, text="Nhận dạng biển báo giao thông", pady=10, font=('arial', 20, 'bold'))
heading.configure(background='#ffffff', foreground='#364156')

heading1 = Label(top, text="Môn Học: Trí tuệ nhân tạo", pady=10, font=('arial', 20, 'bold'))
heading1.configure(background='#ffffff', foreground='#364156')

heading2 = Label(top, text="Giảng viên hướng dẫn : Đoàn Thị Thanh Hằng", pady=5, font=('arial', 20, 'bold'))
heading2.configure(background='#ffffff', foreground='#364156')

heading3 = Label(top, text="Lớp : 73DCHT23", pady=5, font=('arial', 20, 'bold'))
heading3.configure(background='#ffffff', foreground='#364156')

heading.pack()
heading1.pack()
heading2.pack()
heading3.pack()

top.mainloop()
