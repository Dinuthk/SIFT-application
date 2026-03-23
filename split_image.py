import cv2
import os

img_path = r"C:\Users\user\.gemini\antigravity\brain\a4ed0bf7-7bce-49fb-9b0a-9e64547e8f5c\mountain_panorama_1774270473756.png"
img = cv2.imread(img_path)
if img is None:
    print("Could not load image.")
    exit(1)

h, w, _ = img.shape

print(f"Image loaded: {w}x{h}")

part1 = img[:, 0 : w//2 + 50]
part2 = img[:, w//4 : 3*w//4 + 50]
part3 = img[:, w//2 : w]

os.makedirs("pano_photos", exist_ok=True)
cv2.imwrite(r"pano_photos\img_1.png", part1)
cv2.imwrite(r"pano_photos\img_2.png", part2)
cv2.imwrite(r"pano_photos\img_3.png", part3)

print("Saved 3 overlapping images in pano_photos/")
