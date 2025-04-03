from PIL import Image
import cv2
import numpy as np
from scipy.stats import entropy
import os

class ImageAnalysisToolkit:
    def __init__(self, preview_model_dir, log_file_path):
        self.preview_model_dir = preview_model_dir
        self.log_file_path = log_file_path
        self.count = 0
        self.success = 0

    @staticmethod
    def is_solid_color_image(image_path, threshold=0.95):
        img = cv2.imread(image_path)
        height, width, _ = img.shape
        total_pixels = height * width
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pixels = img_rgb.reshape(-1, 3)
        unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
        print(f"solid color value: {np.max(counts) / total_pixels}")
        return np.max(counts) / total_pixels > threshold

    @staticmethod
    def has_text_content(image_path):
        import pytesseract
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img)
        print(f"text_value: {len(text.strip())}")
        return len(text.strip()) < 10

    @staticmethod
    def is_likely_object_image(image_path, min_edge_ratio=0.01, max_edge_ratio=0.2):
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, threshold1=30, threshold2=100)
        total_pixels = gray.shape[0] * gray.shape[1]
        edge_pixels = np.count_nonzero(edges)
        edge_ratio = edge_pixels / total_pixels
        print(f"edge_ratio: {edge_ratio}")
        return min_edge_ratio < edge_ratio < max_edge_ratio

    @staticmethod
    def calculate_image_entropy(image_path):
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist / np.sum(hist)
        entropy_value = entropy(hist, base=2)[0]
        is_likely_object = entropy_value > 1.0
        print(f"entropy value is : {entropy_value}")
        return entropy_value, is_likely_object

    @staticmethod
    def analyze_color_histogram(image_path, bin_threshold=30):
        img = cv2.imread(image_path)
        channels = cv2.split(img)
        populated_bins = 0
        for channel in channels:
            hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
            significant_bins = np.sum(hist > (0.001 * channel.size))
            populated_bins += significant_bins
        print(f"histogram: {populated_bins}")
        return populated_bins > bin_threshold

    @staticmethod
    def is_real_object_image(image_path):
        edge_result = ImageAnalysisToolkit.is_likely_object_image(image_path)
        entropy_value, entropy_result = ImageAnalysisToolkit.calculate_image_entropy(image_path)
        histogram_result = ImageAnalysisToolkit.analyze_color_histogram(image_path)
        text_result = ImageAnalysisToolkit.has_text_content(image_path)
        print(edge_result)
        print(entropy_result)
        print(histogram_result)
        print(text_result)
        positive_counts = sum([edge_result, entropy_result, histogram_result, text_result])
        confidence = positive_counts / 4.0
        reason = "Undetermined"
        if not edge_result and confidence < 0.75:
            reason = "Unusual edge pattern"
        elif not entropy_result and confidence < 0.75:
            reason = "Low image entropy"
        elif not histogram_result and confidence < 0.75:
            reason = "Limited color distribution"
        elif not text_result and confidence < 0.75:
            reason = "has texts"
        is_real_object = confidence >= 0.67
        print(f"final result: {is_real_object}")
        return is_real_object, confidence, reason

    def analyze_directory(self):
        with open(self.log_file_path, "w") as log_file:
            sys.stdout = log_file
            if not os.path.exists(self.preview_model_dir):
                print(f"Directory '{self.preview_model_dir}' does not exist.")
            else:
                for file_name in os.listdir(self.preview_model_dir):
                    file_path = os.path.join(self.preview_model_dir, file_name)
                    if file_name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif")):
                        print("analyzing new file ***************************************************************************************************************************************")
                        content = self.is_real_object_image(file_path)
                        print(content)
                        if content[0] == 1:
                            self.success += 1
                        self.count += 1
                        if self.count > 1000:
                            break
            print(self.count)
            print(self.success)