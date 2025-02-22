import sys
import random
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton, QComboBox, QTextEdit, QFormLayout

# Mẫu câu quảng cáo
TEMPLATES = {
    "destination": [
        "Hãy đến với {destination}, nơi bạn sẽ được chiêm ngưỡng vẻ đẹp tuyệt vời!",
        "Khám phá {destination} cùng chúng tôi với trải nghiệm đáng nhớ!",
        "{destination} - điểm đến lý tưởng cho kỳ nghỉ của bạn!"
    ],
    "duration": [
        "Chuyến đi kéo dài {duration} ngày, đảm bảo bạn có đủ thời gian tận hưởng.",
        "Lịch trình {duration} ngày sẽ mang đến cho bạn những khoảnh khắc đáng giá.",
        "Trải nghiệm trọn vẹn với hành trình {duration} ngày đầy hấp dẫn!"
    ],
    "price": [
        "Chỉ với {price} VNĐ, bạn đã có ngay một tour tuyệt vời!",
        "Giá tour chỉ từ {price} VNĐ – quá hời cho một chuyến đi khó quên!",
        "Nhanh tay đặt ngay với mức giá ưu đãi chỉ {price} VNĐ!"
    ],
    "activities": [
        "Tour bao gồm: {activities}. Hãy chuẩn bị cho những trải nghiệm tuyệt vời!",
        "Bạn sẽ tham gia các hoạt động thú vị như {activities}.",
        "Những hoạt động {activities} sẽ khiến chuyến đi của bạn thêm phần hấp dẫn!"
    ]
}

class TourAdApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Hệ thống tạo quảng cáo du lịch')
        self.setGeometry(0, 0, 1920, 980)

        layout = QVBoxLayout()

        form_layout = QFormLayout()

        # Các trường nhập thông tin
        self.destination_input = QLineEdit()
        self.duration_input = QComboBox()
        self.duration_input.addItems([str(i) for i in range(1, 16)])  # Chọn số ngày từ 1-15
        self.price_input = QLineEdit()
        self.activities_input = QLineEdit()

        form_layout.addRow("Điểm đến:", self.destination_input)
        form_layout.addRow("Thời gian (ngày):", self.duration_input)
        form_layout.addRow("Giá (VNĐ):", self.price_input)
        form_layout.addRow("Hoạt động:", self.activities_input)

        # Nút tạo quảng cáo
        self.generate_button = QPushButton('Tạo quảng cáo')
        self.generate_button.clicked.connect(self.generate_ad)

        # Kết quả
        self.result_label = QLabel("Kết quả sẽ hiển thị ở đây")
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)

        layout.addLayout(form_layout)
        layout.addWidget(self.generate_button)
        layout.addWidget(self.result_label)
        layout.addWidget(self.result_text)

        self.setLayout(layout)

    def generate_ad(self):
        tour_info = {
            "destination": self.destination_input.text().strip(),
            "duration": self.duration_input.currentText(),
            "price": self.price_input.text().strip(),
            "activities": self.activities_input.text().strip()
        }

        if not all(tour_info.values()):
            self.result_text.setText("Vui lòng nhập đầy đủ thông tin!")
            return

        ad_text = self.generate_advertisement(tour_info)
        self.result_text.setText(ad_text)

    def generate_advertisement(self, tour_info):
        ad_text = ""

        for key, templates in TEMPLATES.items():
            if tour_info[key]:
                template = random.choice(templates)
                ad_text += template.format(**tour_info) + "\n"

        return ad_text.strip()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = TourAdApp()
    window.show()
    sys.exit(app.exec_())
