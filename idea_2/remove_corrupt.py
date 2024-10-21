import os
from PIL import Image, UnidentifiedImageError

def delete_corrupted_images(directory):
    # Iterate through all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".png"):
            file_path = os.path.join(directory, filename)
            try:
                # Try opening the image file
                with Image.open(file_path) as img:
                    img.verify()  # Verify that the image can be fully loaded
            except (UnidentifiedImageError, OSError, IOError):
                # If the image is corrupted, delete it
                print(f"Deleting corrupted image: {file_path}")
                os.remove(file_path)
            except Exception as e:
                print(f"Unexpected error for {file_path}: {e}")

if __name__ == "__main__":
    # Define the directory path containing .png files
    directory_path = input("Enter the directory path containing .png images: ")
    
    if os.path.exists(directory_path) and os.path.isdir(directory_path):
        delete_corrupted_images(directory_path)
        print("Completed checking for corrupted images.")
    else:
        print("Invalid directory path.")
