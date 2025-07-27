image_path = "/home/wenhao/CUHK/ECE4513/FinalProject/Data/CASIA/0000105/017.jpg"
import cv2
def get_image_shape(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image at {image_path} could not be read.")
    return image.shape[:2]  # Returns (height, width)
if __name__ == "__main__":
    shape = get_image_shape(image_path)
    print(f"Image shape: {shape}")