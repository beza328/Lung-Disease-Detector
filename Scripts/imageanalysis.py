import sys
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

def plot_class_distribution(data_dir):
    class_counts = {}
    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)
        if os.path.isdir(class_path):
            class_counts[class_name] = len(os.listdir(class_path))

    plt.figure(figsize=(8, 5))
    plt.bar(class_counts.keys(), class_counts.values(), color='skyblue')
    plt.title("Class Distribution in Training Set")
    plt.xlabel("Disease Class")
    plt.ylabel("Number of Images")
    plt.show()



def show_sample_images(data_dir):
    class_names = os.listdir(data_dir)
    plt.figure(figsize=(12, 8))
    
    for i, class_name in enumerate(class_names):
        class_path = os.path.join(data_dir, class_name)
        image_path = os.path.join(class_path, os.listdir(class_path)[0])
        img = mpimg.imread(image_path)
        plt.subplot(2, 2, i+1)
        plt.imshow(img, cmap='gray')
        plt.title(class_name)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()



def check_image_sizes(data_dir):
    sizes = []
    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)
        for image_name in os.listdir(class_path)[:100]:  # Sample 100 per class
            image_path = os.path.join(class_path, image_name)
            with Image.open(image_path) as img:
                sizes.append(img.size)
    
    size_counts = {}
    for size in sizes:
        size_counts[size] = size_counts.get(size, 0) + 1

    print("Most common image sizes:")
    for size, count in sorted(size_counts.items(), key=lambda x: -x[1])[:5]:
        print(f"{size}: {count} images")