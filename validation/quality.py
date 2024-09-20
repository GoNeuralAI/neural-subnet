import cv2
import numpy as np
from skimage import exposure, img_as_float

def assess_image_quality(image_path):
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found.")
        return

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Assess sharpness using the Laplacian method
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    print(f"Sharpness (Laplacian variance): {laplacian_var}")

    # Assess color accuracy (simple mean of pixel values as a placeholder)
    mean_color = cv2.mean(image)[:3]  # BGR
    color_accuracy_score = np.mean(mean_color)
    print(f"Mean Color (BGR): {mean_color} (Score: {color_accuracy_score})")

    # Assess contrast
    contrast = np.std(gray)
    print(f"Contrast (standard deviation): {contrast}")

    # Assess noise (simple standard deviation of pixel values as a placeholder)
    noise = np.std(image)
    print(f"Noise (standard deviation): {noise}")

    # Assess lighting (using histogram equalization)
    img_float = img_as_float(gray)
    eq_img = exposure.equalize_hist(img_float)
    lighting_quality = np.mean(eq_img)
    print(f"Lighting quality (mean of equalized histogram): {lighting_quality}")

    # Weights for each factor
    weights = {
        'sharpness': 0.25,
        'color_accuracy': 0.25,
        'contrast': 0.25,
        'noise': 0.15,
        'lighting': 0.1
    }

    # Calculate weighted score
    weighted_score = (
        weights['sharpness'] * laplacian_var +
        weights['color_accuracy'] * color_accuracy_score +
        weights['contrast'] * contrast +
        weights['noise'] * (1 / (noise + 1e-5)) +  # Inverse because less noise is better
        weights['lighting'] * lighting_quality
    )

    print(f"Overall Quality Score: {weighted_score}")

    # Display the image (optional)
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
# Example usage
assess_image_quality("output_images/preview.png")
