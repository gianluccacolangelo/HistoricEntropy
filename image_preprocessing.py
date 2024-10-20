import numpy as np
from PIL import Image
import math
from typing import List, Tuple
import matplotlib.pyplot as plt

class ImagePreprocessor:
    def __init__(self, target_size: Tuple[int, int] = (256, 256)):
        self.target_size = target_size

    def preprocess(self, image_path: str) -> np.ndarray:
        """Load, resize, and convert image to grayscale."""
        with Image.open(image_path) as img:
            # Resize image while maintaining aspect ratio
            img.thumbnail(self.target_size, Image.LANCZOS)
            
            # Create a new white background image
            background = Image.new('L', self.target_size, 255)
            
            # Paste the resized image onto the center of the background
            offset = ((self.target_size[0] - img.size[0]) // 2,
                      (self.target_size[1] - img.size[1]) // 2)
            background.paste(img, offset)
            
            # Convert to grayscale if not already
            gray_img = background.convert("L")
            
            return np.array(gray_img)

class ImageEntropyAnalyzer:
    def __init__(self, target_size: Tuple[int, int] = (256, 256)):
        self.preprocessor = ImagePreprocessor(target_size)

    def calculate_pixel_probabilities(self, pixel_values: np.ndarray) -> np.ndarray:
        """Calculate the probability distribution of pixel values."""
        frequencies = np.bincount(pixel_values.flatten(), minlength=256)
        return frequencies / frequencies.sum()

    def calculate_shannon_entropy(self, probabilities: np.ndarray) -> float:
        """Calculate Shannon entropy from probability distribution."""
        return -sum(p * math.log2(p) for p in probabilities if p > 0)

    def analyze_image(self, image_path: str) -> float:
        """Analyze a single image and return its entropy."""
        pixel_values = self.preprocessor.preprocess(image_path)
        probabilities = self.calculate_pixel_probabilities(pixel_values)
        return self.calculate_shannon_entropy(probabilities)

    def analyze_multiple_images(self, image_paths: List[str]) -> List[Tuple[str, float]]:
        """Analyze multiple images and return a list of (path, entropy) tuples."""
        return [(path, self.analyze_image(path)) for path in image_paths]

    def plot_image_and_histogram(self, image_path: str, save_path: str = None):
        """Plot the preprocessed image and its frequency histogram."""
        # Preprocess the image
        pixel_values = self.preprocessor.preprocess(image_path)
        
        # Calculate pixel probabilities
        probabilities = self.calculate_pixel_probabilities(pixel_values)
        
        # Calculate entropy
        entropy = self.calculate_shannon_entropy(probabilities)
        
        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot the preprocessed image
        ax1.imshow(pixel_values, cmap='gray')
        ax1.set_title('Preprocessed Image')
        ax1.axis('off')
        
        # Plot the histogram
        ax2.hist(pixel_values.flatten(), bins=256, range=(0, 255), density=True, alpha=0.7)
        ax2.set_title('Pixel Value Histogram')
        ax2.set_xlabel('Pixel Value')
        ax2.set_ylabel('Frequency')
        ax2.set_xlim(0, 255)
        
        plt.suptitle(f'Image Analysis - Entropy: {entropy:.2f}', fontsize=16)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
        
        return entropy  # Make sure to return the entropy value

# Example usage
if __name__ == "__main__":
    analyzer = ImageEntropyAnalyzer()
    
    # Single image analysis with plotting
    image_path = 'caravaggio.jpg'
    analyzer.plot_image_and_histogram(image_path)

    # Multiple image analysis
    #image_paths = ['artwork1.jpg', 'artwork2.jpg', 'artwork3.jpg']
    #results = analyzer.analyze_multiple_images(image_paths)
    #for path, entropy in results:
        #print(f"Entropy of {path}: {entropy}")
