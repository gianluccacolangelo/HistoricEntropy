from image_preprocessing import ImageEntropyAnalyzer
import os

def generate_readme():
    analyzer = ImageEntropyAnalyzer()
    images = ['michelangelo.jpg', 'caravaggio.jpg', 'kandinsky.jpg', 'rothko.jpg']
    
    readme_content = "# HistoricEntropy\n\n"
    
    for image in images:
        image_name = os.path.splitext(image)[0]
        plot_path = f"images/{image_name}_plot.png"
        
        # Ensure the images directory exists
        os.makedirs("images", exist_ok=True)
        
        # Generate and save the plot
        entropy = analyzer.plot_image_and_histogram(image)
        
        # Add content to README
        readme_content += f"## {image_name.capitalize()}\n\n"
        readme_content += f"![{image_name} Analysis]({plot_path})\n\n"
        readme_content += f"Entropy: {entropy:.2f}\n\n"
    
    # Write the README file
    with open("README.md", "w") as f:
        f.write(readme_content)

if __name__ == "__main__":
    generate_readme()
