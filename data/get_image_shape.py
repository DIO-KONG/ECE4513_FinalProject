from cv2 import imread

image = imread("data/CK+1/anger/S010_004_00000017.png", 0)

shape = image.shape
print(f"Image shape: {shape}")