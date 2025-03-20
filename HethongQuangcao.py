import random
import json
import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QTextEdit, QPushButton, 
    QVBoxLayout, QFormLayout, QHBoxLayout
)
from PyQt5.QtGui import QFont, QColor, QPalette
from PyQt5.QtCore import Qt
from collections import defaultdict, Counter
import re

with open("country_categories.json", "r", encoding="utf-8") as file:
    COUNTRY_CATEGORIES = json.load(file)

with open("new_templates.json", "r", encoding="utf-8") as file:
    TEMPLATES = json.load(file)

class MarkovChain:
    def __init__(self, n_grams=3):
        self.n_grams = n_grams
        self.model = defaultdict(Counter)

    def train(self, text):
        """Huấn luyện mô hình Markov với văn bản đầu vào."""
        words = text.split()
        if len(words) < self.n_grams:
            raise ValueError("Text is too short to train the model with the specified n_grams.")
        
        for i in range(len(words) - self.n_grams):
            n_gram = tuple(words[i:i + self.n_grams])
            next_word = words[i + self.n_grams]
            self.model[n_gram][next_word] += 1
        
        # Áp dụng smoothing để giảm thiểu lặp lại từ phổ biến
        for n_gram in self.model:
            total = sum(self.model[n_gram].values())
            for word in self.model[n_gram]:
                self.model[n_gram][word] = (self.model[n_gram][word] + 1) / (total + len(self.model[n_gram]))

    def _choose_next_word(self, current_gram, used_words, retry=3):
        """Chọn từ tiếp theo dựa trên xác suất, giữ câu dài hơn."""
        if current_gram not in self.model or not self.model[current_gram]:
            return None  

        next_words = list(self.model[current_gram].keys())
        weights = list(self.model[current_gram].values())

        # Giảm bớt ảnh hưởng của việc tránh lặp từ
        for i, word in enumerate(next_words):
            if word in used_words:
                weights[i] *= 0.9  # Giảm nhẹ trọng số nếu từ đã được sử dụng

        # Làm mềm trọng số để có tính ngẫu nhiên cao hơn
        weights = [w ** 0.7 for w in weights]

        for _ in range(retry):  # Thử chọn lại nếu cần
            chosen_word = random.choices(next_words, weights=weights, k=1)[0]
            if chosen_word not in used_words:
                return chosen_word  

        return random.choice(next_words) if next_words else None  # Chọn fallback nếu tất cả đã dùng

    def _post_process_text(self, text, max_length):
        """Xử lý hậu kỳ để câu không bị ngắn quá mức."""
        words = text.split()

        # Nếu câu quá ngắn, tiếp tục thêm từ
        if len(words) < max_length * 0.8:
            return self._extend_sentence(text, max_length)

        # Nếu dài quá 120% max_length thì cắt lại
        if len(words) > max_length * 1.2:
            text = " ".join(words[:int(max_length * 1.2)])

        # Đảm bảo câu kết thúc bằng dấu câu
        if not text.endswith(('.', '!', '?')):
            text += '.'

        return text

    def _extend_sentence(self, text, max_length):
        """Kéo dài câu nếu nó quá ngắn."""
        words = text.split()
        while len(words) < max_length * 1.0:
            current_gram = tuple(words[-self.n_grams:])
            next_word = self._choose_next_word(current_gram, set(words))
            if next_word is None:
                break  # Không có từ tiếp theo phù hợp
            words.append(next_word)
        return " ".join(words)

    def generate_text(self, max_length=100, beam_width=8):
        """Tạo văn bản bằng Beam Search với khả năng kéo dài câu."""
        if not self.model:
            raise ValueError("Model has not been trained.")

        possible_starters = [n_gram for n_gram in self.model.keys() if n_gram[0].istitle()]
        if not possible_starters:
            possible_starters = list(self.model.keys())

        if not possible_starters:
            raise ValueError("Không tìm thấy n-gram hợp lệ!")

        beams = [(list(random.choice(possible_starters)), 1.0, set())]

        for _ in range(max_length - self.n_grams):
            new_beams = []
            for words, score, used_words in beams:
                # Sửa: sử dụng self.n_grams cố định thay vì random.randint(3,4)
                current_gram = tuple(words[-self.n_grams:])

                if current_gram not in self.model:
                    continue

                try:
                    next_word = self._choose_next_word(current_gram, used_words)
                    if next_word is None:
                        continue  # Nếu không có từ hợp lệ, bỏ qua
                except IndexError:
                    continue  

                new_used_words = used_words.copy()
                new_used_words.add(next_word)

                new_score = score * self.model[current_gram].get(next_word, 1e-5)
                if next_word not in used_words:
                    new_score *= 1.3  # Tăng điểm nếu từ chưa xuất hiện

                new_beams.append((words + [next_word], new_score, new_used_words))

            if not new_beams:
                break

            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]

        if not beams:
            return "Không thể tạo câu hợp lệ."

        best_sentence = " ".join(beams[0][0])
        raw_text = self._post_process_text(best_sentence, max_length)
        return self._clean_generated_text(raw_text)
    
    def _clean_generated_text(self, text):
        """Loại bỏ từ lặp và sửa lỗi câu."""
        words = text.split()
        cleaned_words = []
        prev_word = None

        for word in words:
            if word.lower() != prev_word:  # Loại bỏ từ lặp
                cleaned_words.append(word)
            prev_word = word.lower()
        
        cleaned_text = " ".join(cleaned_words)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()  # Xóa khoảng trắng thừa
        return cleaned_text
    
def train_header_markov_models():
        """
        Huấn luyện các mô hình Markov riêng cho các trường header từ TEMPLATES.
        Giả sử TEMPLATES có các key: "tour_name", "target_audience", "accommodation", 
        "tour_type", "schedule", "hotline", "discount".
        """
        header_keys = ["tour_name", "target_audience", "accommodation", "tour_type", "schedule", "hotline", "discount"]
        header_markov = {}
        for key in header_keys:
            if key in TEMPLATES and isinstance(TEMPLATES[key], list):
                model = MarkovChain(n_grams=3)
                for template in TEMPLATES[key]:
                    model.train(template)
                header_markov[key] = model
        return header_markov

class GeneticAlgorithm:
    def __init__(self, population_size, max_generations, min_similarity=0.85):
        self.population_size = population_size
        self.max_generations = max_generations
        self.min_similarity = min_similarity  # Ngưỡng tương đồng tối thiểu giữa câu gốc và biến thể

    def similarity(self, s1, s2):
        """Tính toán độ tương đồng giữa hai câu dưới dạng tỉ lệ số từ chung."""
        words1 = set(s1.split())
        words2 = set(s2.split())
        if not words1 or not words2:
            return 0
        return len(words1.intersection(words2)) / max(len(words1), len(words2))

    def fitness(self, sentence):
        """Đánh giá độ phù hợp của câu quảng cáo dựa trên độ dài và từ khóa."""
        keywords = ["du lịch", "giá rẻ", "khuyến mãi", "hấp dẫn"]
        score = len(sentence)
        for keyword in keywords:
            if keyword in sentence:
                score += 10
        return score

    def crossover(self, parent1, parent2):
        """
        Lai tạo hai câu quảng cáo bằng cách ghép phần đầu của cha thứ nhất với phần cuối của cha thứ hai.
        Nếu kết quả không đủ tương đồng với ít nhất một trong hai, giữ lại cha ban đầu.
        """
        words1 = parent1.split()
        words2 = parent2.split()
        split1 = len(words1) // 2
        split2 = len(words2) // 2
        child1 = ' '.join(words1[:split1] + words2[split2:])
        child2 = ' '.join(words2[:split2] + words1[split1:])
        if max(self.similarity(child1, parent1), self.similarity(child1, parent2)) < self.min_similarity:
            child1 = parent1
        if max(self.similarity(child2, parent1), self.similarity(child2, parent2)) < self.min_similarity:
            child2 = parent2
        return child1, child2

    def evolve(self, population):
        """Tiến hóa quần thể câu quảng cáo trong một thế hệ."""
        new_population = []
        fitness_scores = [self.fitness(sentence) for sentence in population]
        total_fitness = sum(fitness_scores)
        if total_fitness == 0:
            return population
        for _ in range(self.population_size):
            parent1, parent2 = random.choices(population, weights=fitness_scores, k=2)
            child1, child2 = self.crossover(parent1, parent2)
            new_population.append(child1)
            new_population.append(child2)
        return new_population[:self.population_size]

    def optimize(self, initial_sentences):
        """Tối ưu hóa quần thể qua nhiều thế hệ và trả về câu quảng cáo có fitness cao nhất."""
        population = initial_sentences
        for _ in range(self.max_generations):
            population = self.evolve(population)
        return max(population, key=self.fitness)

def classify_country(destination):
        # Phân loại quốc gia dựa trên điểm đến để chọn mẫu quảng cáo phù hợp
        destination_lower = destination.lower()
        
        for category, names in COUNTRY_CATEGORIES.items():
            if any(name in destination_lower for name in names):
                return category

        return "general"  # Nếu không tìm thấy, chọn danh mục chung

def classify_place(places):
    """
    Phân loại các địa điểm thành các loại như thiên nhiên, cổ kính, địa danh, v.v.
    """
    place_types = set()  # Sử dụng set để tránh trùng lặp

    nature_keywords = ["thác", "biển", "hồ", "núi", "đồi", "sông"]
    landmark_keywords = ["tháp", "tượng đài", "cầu", "thành phố", "biểu tượng", "nhà thờ", "quảng trường"]
    historical_keywords = ["cung điện", "kim tự tháp", "di tích", "lăng tẩm", "đền", "chùa"]

    for place in places:
        place_lower = place.lower()
        if any(keyword in place_lower for keyword in nature_keywords):
            place_types.add("nature")
        elif any(keyword in place_lower for keyword in landmark_keywords):
            place_types.add("landmark")
        elif any(keyword in place_lower for keyword in historical_keywords):
            place_types.add("historical")
        else:
            place_types.add("general")  # Loại chung nếu không thuộc các loại trên

    return list(place_types)

def load_best_ads(filename="best_ads.json"):
        """Tải các câu quảng cáo tốt nhất từ file JSON."""
        try:
            with open(filename, "r", encoding="utf-8") as file:
                return json.load(file)
        except (FileNotFoundError, json.JSONDecodeError):
            return []

def filter_best_ads(best_ads, destination, places):
    """
    Lọc các quảng cáo tốt nhất dựa trên điểm đến và loại địa điểm.
    Chỉ giữ lại các quảng cáo có chứa từ khóa liên quan đến điểm đến và loại địa điểm.
    """
    filtered_ads = []
    destination_lower = destination.lower()

    # Danh sách các từ khóa liên quan đến điểm đến
    destination_keywords = COUNTRY_CATEGORIES.get(destination_lower, [destination_lower])

    # Phân loại loại địa điểm
    place_types = classify_place(places)

    for ad in best_ads:
        ad_lower = ad.lower()
        # Kiểm tra xem quảng cáo có chứa từ khóa liên quan đến điểm đến không
        if any(keyword in ad_lower for keyword in destination_keywords):
            # Kiểm tra xem quảng cáo có chứa từ khóa liên quan đến loại địa điểm không
            if any(place_type in ad_lower for place_type in place_types):
                filtered_ads.append(ad)
    
    return filtered_ads

def train_markov_models(destination, places):
    # Huấn luyện mô hình Markov cho từng loại dữ liệu và từng quốc gia
    destination_markov = {}
    duration_markov = MarkovChain(n_grams=3)
    price_markov = MarkovChain(n_grams=3)
    places_markov = {}

    # Tải các quảng cáo tốt nhất đã lưu
    best_ads = load_best_ads()

    # Lọc các quảng cáo tốt nhất dựa trên điểm đến và loại địa điểm
    filtered_best_ads = filter_best_ads(best_ads, destination, places)

    # Huấn luyện mô hình cho điểm đến theo từng quốc gia
    for country, templates in TEMPLATES["destination"].items():
        markov = MarkovChain(n_grams=5)
        for template in templates:
            markov.train(template)  # Huấn luyện trên các mẫu template
        # Thêm các quảng cáo tốt nhất đã lọc vào dữ liệu huấn luyện
        for ad in filtered_best_ads:
            markov.train(ad)  # Huấn luyện trên các quảng cáo tốt nhất
        destination_markov[country] = markov

    # Huấn luyện mô hình cho thời gian
    for template in TEMPLATES["duration"]:
        duration_markov.train(template)  # Huấn luyện trên các mẫu template
    # Thêm các quảng cáo tốt nhất đã lọc vào dữ liệu huấn luyện
    for ad in filtered_best_ads:
        duration_markov.train(ad)  # Huấn luyện trên các quảng cáo tốt nhất

    # Huấn luyện mô hình cho giá
    for template in TEMPLATES["price"]:
        price_markov.train(template)  # Huấn luyện trên các mẫu template
    # Thêm các quảng cáo tốt nhất đã lọc vào dữ liệu huấn luyện
    for ad in filtered_best_ads:
        price_markov.train(ad)  # Huấn luyện trên các quảng cáo tốt nhất

    # Huấn luyện mô hình cho các địa điểm theo từng loại
    for category, templates in TEMPLATES["places"].items():
        markov = MarkovChain(n_grams=3)
        for template in templates:
            markov.train(template)  # Huấn luyện trên các mẫu template
        # Thêm các quảng cáo tốt nhất đã lọc vào dữ liệu huấn luyện
        for ad in filtered_best_ads:
            markov.train(ad)  # Huấn luyện trên các quảng cáo tốt nhất
        places_markov[category] = markov

    return destination_markov, duration_markov, price_markov, places_markov

def format_discount(discount):
    """Định dạng giảm giá: Nếu có %, giữ nguyên. Nếu là số, thêm VND."""
    discount = discount.strip()
    if "%" in discount:  # Nếu nhập dạng "10%"
        return discount
    elif discount.isdigit():  # Nếu chỉ nhập số, tự động thêm "VND"
        discount = int(discount.replace(".", "").replace(",", ""))  # Loại bỏ dấu . hoặc ,
        return f"{discount:,}".replace(",", ".") + " VND"  # Định dạng lại với dấu .
    return discount  # Nếu nhập sai, trả về nguyên gốc

def calculate_discounted_price(price, discount):
        
        """Tính toán giá mới sau khi áp dụng giảm giá."""
        try:
            # Chuyển giá về số nguyên (loại bỏ ký tự VNĐ, dấu phẩy)
            price = int(re.sub(r"\D", "", price))

            # Nếu giảm giá là phần trăm (vd: 20%)
            if "%" in discount:
                discount_percent = int(re.sub(r"\D", "", discount))
                new_price = price * (1 - discount_percent / 100)
            
            # Nếu giảm giá là số tiền cụ thể (vd: 500.000 VNĐ)
            else:
                discount_amount = int(re.sub(r"\D", "", discount))
                new_price = price - discount_amount
            
            # Đảm bảo giá không bị âm
            new_price = max(new_price, 0)

            return f"{int(new_price):,} VNĐ"  # Định dạng số có dấu phẩy
        except:
            return price  # Nếu lỗi, trả về giá gốc

# Tạo giao diện
class TourAdApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):

        self.setWindowTitle('Tạo quảng cáo du lịch')
        self.setGeometry(0, 0, 1920, 980)
        self.setStyleSheet("background-color: #f5f5f5;")  # Đặt màu nền

        main_layout = QHBoxLayout()  # Chia bố cục thành hai phần ngang

        # Phần nhập thông tin (bên trái)
        input_layout = QVBoxLayout()
        form_layout = QFormLayout()

        # Định dạng chung cho nhãn và ô nhập
        label_style = "font-size: 18px; font-weight: 500; color: #333;"
        input_style = "font-size: 18px; padding: 5px; border: 1px solid #aaa; border-radius: 5px; margin-bottom: 5px;"

        # Thêm các trường nhập thông tin mới
        self.tour_name_input = QLineEdit()
        self.tour_name_input.setPlaceholderText("Ví dụ: Tour khám phá Đà Lạt 3N2Đ")
        self.tour_name_input.setStyleSheet(input_style)
        tour_name_label = QLabel("Tên Tour:")
        tour_name_label.setStyleSheet(label_style)
        form_layout.addRow(tour_name_label, self.tour_name_input)

        self.audience_input = QLineEdit()
        self.audience_input.setPlaceholderText("Ví dụ: Gia đình, cặp đôi, nhóm bạn, 1 mình")
        self.audience_input.setStyleSheet(input_style)
        audience_label = QLabel("Phù hợp với:")
        audience_label.setStyleSheet(label_style)
        form_layout.addRow(audience_label, self.audience_input)

        self.accommodation_input = QLineEdit()
        self.accommodation_input.setPlaceholderText("Ví dụ: Khách sạn 5 sao, Resort cao cấp, Homestay")
        self.accommodation_input.setStyleSheet(input_style)
        accommodation_label = QLabel("Loại hình lưu trú:")
        accommodation_label.setStyleSheet(label_style)
        form_layout.addRow(accommodation_label, self.accommodation_input)

        self.tour_type_input = QLineEdit()
        self.tour_type_input.setPlaceholderText("Ví dụ: Nghỉ dưỡng, khám phá, phiêu lưu, sinh thái")
        self.tour_type_input.setStyleSheet(input_style)
        tour_type_label = QLabel("Loại hình du lịch:")
        tour_type_label.setStyleSheet(label_style)
        form_layout.addRow(tour_type_label, self.tour_type_input)

        self.schedule_input = QTextEdit()
        self.schedule_input.setPlaceholderText("Ví dụ: Ngày 1: Tham quan Hồ Xuân Hương. Ngày 2: Check-in đồi chè.")
        self.schedule_input.setStyleSheet(input_style)
        self.schedule_input.setFixedHeight(120)  # Giảm chiều cao của ô nhập
        schedule_label = QLabel("Lịch trình chi tiết:")
        schedule_label.setStyleSheet(label_style)
        form_layout.addRow(schedule_label, self.schedule_input)

        self.hotline_input = QLineEdit()
        self.hotline_input.setPlaceholderText("Ví dụ: 0987 654 321")
        self.hotline_input.setStyleSheet(input_style)
        hotline_label = QLabel("Hotline:")
        hotline_label.setStyleSheet(label_style)
        form_layout.addRow(hotline_label, self.hotline_input)

        self.discount_input = QLineEdit()
        self.discount_input.setPlaceholderText("Ví dụ: 20% hoặc 1.000.000 VNĐ")
        self.discount_input.setStyleSheet(input_style)
        discount_label = QLabel("Giảm giá (nếu có):")
        discount_label.setStyleSheet(label_style)
        form_layout.addRow(discount_label, self.discount_input)

        # Các trường nhập thông tin
        self.destination_input = QLineEdit()
        self.destination_input.setPlaceholderText("Ví dụ: Đà Lạt, Phú Quốc, Nha Trang")
        self.destination_input.setStyleSheet(input_style)
        destination_label = QLabel("Điểm đến:")
        destination_label.setStyleSheet(label_style)
        form_layout.addRow(destination_label, self.destination_input)

        self.duration_input = QLineEdit()
        self.duration_input.setPlaceholderText("Ví dụ: 3 ngày 2 đêm, 1 tuần, 1 tháng")
        self.duration_input.setStyleSheet(input_style)
        duration_label = QLabel("Thời gian (ngày-đêm/tuần/tháng):")
        duration_label.setStyleSheet(label_style)
        form_layout.addRow(duration_label, self.duration_input)

        self.price_input = QLineEdit()
        self.price_input.setPlaceholderText("Ví dụ: 5.000.000")
        self.price_input.setStyleSheet(input_style)
        price_label = QLabel("Giá (VNĐ):")
        price_label.setStyleSheet(label_style)
        form_layout.addRow(price_label, self.price_input)

        self.places_input = QLineEdit()
        self.places_input.setPlaceholderText("Ví dụ: Hồ Xuân Hương, Đồi chè Cầu Đất, Thác Datanla")
        self.places_input.setStyleSheet(input_style)
        places_label = QLabel("Các địa điểm tham quan:")
        places_label.setStyleSheet(label_style)
        form_layout.addRow(places_label, self.places_input)

        # Nút Tạo quảng cáo
        self.generate_button = QPushButton('Tạo quảng cáo')
        self.generate_button.setStyleSheet(
            "background-color: #007bff; color: white; font-size: 20px; padding: 8px; border-radius: 5px;"
        )
        self.generate_button.clicked.connect(self.generate_ad)
        form_layout.addRow(self.generate_button)

        input_layout.addLayout(form_layout)
         # Phần hiển thị kết quả (bên phải)
        display_layout = QVBoxLayout()
        self.result_label = QLabel("Kết quả:")
        self.result_label.setStyleSheet(label_style)
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setStyleSheet("background-color: white; padding: 5px; font-size: 20px; border: 1px solid #aaa; border-radius: 5px;")

        display_layout.addWidget(self.result_label)
        display_layout.addWidget(self.result_text)

        # Thêm hai phần vào bố cục chính
        main_layout.addLayout(input_layout, 1)
        main_layout.addLayout(display_layout, 1)

        self.setLayout(main_layout)
    
    def replace_placeholders(self, template, destination, duration, price, places, tour_name, audience, accommodation, tour_type, schedule, hotline, discount):
        """Thay thế placeholder trong template bằng dữ liệu thực tế, đảm bảo không bị lỗi nội dung."""
        # Định nghĩa dictionary chứa icon cho từng trường
        icons = {
            "{destination}": ["📍", "🏙️", "🌆"],
            "{duration}": ["⏳", "🕒", "⌛"],
            "{price}": ["💰", "💵", "💸"],
            "{places}": ["📌", "🗺️", "🏞️"],
            "{tour_name}": ["📝", "🎒", "🏖️"],
            "{audience}": ["👥", "👫", "👨‍👩‍👧‍👦"],
            "{accommodation}": ["🏨", "🛏️", "🏠"],
            "{tour_type}": ["🚗", "✈️", "🚢"],
            "{schedule}": ["📆", "🗓️", "⌚"],
            "{hotline}": ["📞", "☎️", "📱"],
            "{discount}": ["🏷️", "💲", "🔖"]
        }
        mapping = {
            "{destination}": destination.title(),
            "{duration}": duration,
            "{price}": price,
            "{places}": places,
            "{tour_name}": tour_name,
            "{audience}": audience,
            "{accommodation}": accommodation,
            "{tour_type}": tour_type,
            "{schedule}": schedule,
            "{hotline}": hotline,
            "{discount}": discount if discount.strip() else "Không áp dụng"
        }

         # Thêm icon ngẫu nhiên cho mỗi trường nếu có trong dictionary icons
        for placeholder, value in mapping.items():
            if placeholder in icons:  # Chỉ thêm icon nếu placeholder có trong danh sách icon
                icon = random.choice(icons[placeholder])
                mapping[placeholder] = icon + " " + value

        # Thay thế các placeholder trong template
        for placeholder, value in mapping.items():
            template = template.replace(placeholder, value)

        return template
    
    def is_similar(self, sentence1, generated_sentences, threshold=0.5):
        """
        Kiểm tra xem câu mới có quá giống với các câu đã tạo trước đó không.
        """
        # Kiểm tra xem sentence1 có phải là chuỗi không
        if not isinstance(sentence1, str):
            return False  # Nếu không phải chuỗi, coi như không giống

        for sentence2 in generated_sentences:
            # Kiểm tra xem sentence2 có phải là chuỗi không
            if not isinstance(sentence2, str):
                continue  # Bỏ qua nếu không phải chuỗi

            words1 = set(sentence1.split())
            words2 = set(sentence2.split())
            intersection = words1.intersection(words2)
            similarity = len(intersection) / max(len(words1), len(words2))
            if similarity > threshold:
                return True
        return False   

    def generate_advertisement(self, destination, duration, price, places):
        """
        Tạo quảng cáo bằng Markov Chains và tối ưu hóa bằng Genetic Algorithm.
        """
        ad_text = []
        generated_sentences = set()  # Lưu lại các câu đã tạo

        # Lấy giá trị đầu vào từ giao diện
        tour_name = self.tour_name_input.text().strip()
        audience = self.audience_input.text().strip()
        accommodation = self.accommodation_input.text().strip()
        tour_type = self.tour_type_input.text().strip()
        schedule = self.schedule_input.toPlainText().strip()
        hotline = self.hotline_input.text().strip()
        discount = self.discount_input.text().strip()

        # Định dạng giảm giá trước khi tạo quảng cáo
        if discount:
            discount = format_discount(discount)

        if discount.strip():
            price = calculate_discounted_price(price, discount)
        else:
            price = calculate_discounted_price(price, "0%")
            discount = ""

        # Phân loại quốc gia dựa trên điểm đến
        country_category = classify_country(destination)
        
        # Tạo quảng cáo cho điểm đến (sử dụng MarkovChain tương ứng với quốc gia)
        if country_category in self.destination_markov:
            destination_ad = self.destination_markov[country_category].generate_text(max_length=500)
            if isinstance(destination_ad, str) and not self.is_similar(destination_ad, generated_sentences):
                ga = GeneticAlgorithm(population_size=10, max_generations=5)
                optimized_destination_ad = ga.optimize([destination_ad])
                formatted = self.replace_placeholders(
                    optimized_destination_ad,
                    destination,
                    duration,
                    price,
                    ", ".join(places),
                    tour_name,
                    audience,
                    accommodation,
                    tour_type,
                    schedule,
                    hotline,
                    discount
                )
                ad_text.append(formatted.strip())
                generated_sentences.add(optimized_destination_ad)

        # Tạo quảng cáo cho thời gian (sử dụng MarkovChain chung)
        duration_ad = self.duration_markov.generate_text(max_length=150)
        if isinstance(duration_ad, str) and not self.is_similar(duration_ad, generated_sentences):
            ga = GeneticAlgorithm(population_size=10, max_generations=5)
            optimized_duration_ad = ga.optimize([duration_ad])
            formatted = self.replace_placeholders(
                optimized_duration_ad,
                destination,
                duration,
                price,
                ", ".join(places),
                tour_name,
                audience,
                accommodation,
                tour_type,
                schedule,
                hotline,
                discount
            )
            ad_text.append(formatted.strip())
            generated_sentences.add(optimized_duration_ad)

        # Tạo quảng cáo cho giá (sử dụng MarkovChain chung)
        price_ad = self.price_markov.generate_text(max_length=100)
        if isinstance(price_ad, str) and not self.is_similar(price_ad, generated_sentences):
            ga = GeneticAlgorithm(population_size=10, max_generations=5)
            optimized_price_ad = ga.optimize([price_ad])
            formatted = self.replace_placeholders(
                optimized_price_ad,
                destination,
                duration,
                price,
                ", ".join(places),
                tour_name,
                audience,
                accommodation,
                tour_type,
                schedule,
                hotline,
                discount
            )
            ad_text.append(formatted.strip())
            generated_sentences.add(optimized_price_ad)

        # Tạo quảng cáo cho các địa điểm (sử dụng MarkovChain theo loại địa điểm)
        place_types = {"nature": [], "landmark": [], "historical": [], "general": []}
        for place in places:
            categories = classify_place(place)
            for category in categories:
                place_types[category].append(place)

        for category, place_list in place_types.items():
            if place_list and category in self.places_markov:
                place_ad = self.places_markov[category].generate_text(max_length=180)
                if isinstance(place_ad, str) and not self.is_similar(place_ad, generated_sentences):
                    ga = GeneticAlgorithm(population_size=10, max_generations=5)
                    optimized_place_ad = ga.optimize([place_ad])
                    formatted = self.replace_placeholders(
                        optimized_place_ad,
                        destination,
                        duration,
                        price,
                        ", ".join(places),
                        tour_name,
                        audience,
                        accommodation,
                        tour_type,
                        schedule,
                        hotline,
                        discount
                    )
                    ad_text.append(formatted.strip())
                    generated_sentences.add(optimized_place_ad)

        if not hasattr(self, 'header_markov'):
            self.header_markov = train_header_markov_models()

       # Tạo phần header sử dụng template của từng trường (mỗi trường trên một dòng)
        header_parts = []
        header_fields = [
            ("tour_name", tour_name),
            ("target_audience", audience),
            ("accommodation", accommodation),
            ("tour_type", tour_type),
            ("schedule", schedule),
            ("hotline", hotline),
            
        ]
        if discount.strip():
            header_fields.append(("discount", discount))
        # Định nghĩa thứ tự mong muốn của các trường
        order = {
            "tour_name": 1,
            "target_audience": 2,
            "accommodation": 3,
            "tour_type": 4,
            "schedule": 5,
            "discount": 6,
            "hotline": 7
        }

        # Sắp xếp header_fields theo thứ tự mong muốn
        header_fields = sorted(header_fields, key=lambda x: order.get(x[0], 999))

         # Sinh ra header cho từng trường bằng MarkovChain riêng
        for  key, value in header_fields:
            if key in self.header_markov:
                generated_text = self.header_markov[key].generate_text(max_length=50)  # Giảm max_length
                formatted_text = self.replace_placeholders(
                    generated_text, destination, duration, price, ", ".join(places),
                    tour_name, audience, accommodation, tour_type, schedule, hotline, discount
                )
                if len(formatted_text.split()) > 3:  # Chỉ giữ câu có ít nhất 3 từ
                    header_parts.append(formatted_text.strip())
            else:
                header_parts.append(value)
                
                
        header = "\n".join(header_parts)+"\n"

        # Ghép nội dung quảng cáo (đoạn văn duy nhất) với header
        generated_content = " ".join(ad_text)
        final_ad = header + generated_content

        return final_ad
    
    def save_best_ad(self, ad_text, filename="best_ads.json"):
        """Lưu câu quảng cáo tốt nhất vào file JSON."""
        try:
            with open(filename, "r", encoding="utf-8") as file:
                data = json.load(file)
        except (FileNotFoundError, json.JSONDecodeError):
            data = []

        data.append(ad_text)
        with open(filename, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
            

    def generate_ad(self):
        destination = self.destination_input.text().strip()
        duration = self.duration_input.text().strip()
        price = self.price_input.text().strip()
        places = [place.strip() for place in self.places_input.text().split(",") if place.strip()]
        tour_name = self.tour_name_input.text().strip()
        audience = self.audience_input.text().strip()
        accommodation = self.accommodation_input.text().strip()
        tour_type = self.tour_type_input.text().strip()
        schedule = self.schedule_input.toPlainText().strip()
        hotline = self.hotline_input.text().strip()
        discount = self.discount_input.text().strip()  # Cho phép rỗng

        # Kiểm tra tất cả trường bắt buộc (trừ giảm giá)
        if not all([destination, duration, price, places, tour_name, audience, accommodation, tour_type, schedule, hotline]):
            self.result_text.setText("Vui lòng nhập đầy đủ thông tin !")
            return
        
        # Nếu có nhập giảm giá thì kiểm tra định dạng hợp lệ (tùy yêu cầu)
        if discount and not re.match(r'^\d+(\.\d+)?%?$|^\d+ VNĐ$', discount):
            self.result_text.setText("Định dạng giảm giá không hợp lệ! Vui lòng nhập số tiền hoặc phần trăm.")
            return
        
        # Định dạng giảm giá (nếu có nhập)
        if discount:
            discount = format_discount(discount)

        # Lấy giá trị đầu vào từ giao diện
        if not destination or not duration or not price or not places:
            self.result_text.setText("Vui lòng nhập đầy đủ thông tin!")
            return

        # Kiểm tra xem các mô hình Markov đã được huấn luyện chưa
        if not hasattr(self, 'destination_markov'):
            # Truyền điểm đến và địa điểm vào hàm train_markov_models để lọc quảng cáo tốt nhất
            self.destination_markov, self.duration_markov, self.price_markov, self.places_markov = train_markov_models(destination, places)

        # Gọi phương thức generate_advertisement với đầy đủ tham số
        ad_text = self.generate_advertisement(
            destination=destination,
            duration=duration,
            price=price,
            places=places
        )

        # Lưu quảng cáo vừa tạo vào biến tạm thời
        self.current_ad = ad_text
        # Hiển thị kết quả ra giao diện
        self.result_text.setText(ad_text)
        # Hiển thị nút "Lưu" để người dùng có thể lưu quảng cáo
        self.save_best_ad(ad_text)
            
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = TourAdApp()
    window.show()
    sys.exit(app.exec_())