from PIL import Image, ImageDraw
import random

# Create a blank canvas (drone view)
img = Image.new("RGB", (512, 512), color="lightgreen")
draw = ImageDraw.Draw(img)

# Draw some random 'bushes' (representing lantana)
for _ in range(10):
    x, y = random.randint(50, 450), random.randint(50, 450)
    r = random.randint(20, 40)
    draw.ellipse((x - r, y - r, x + r, y + r), fill="forestgreen")

    # Optional: add purple blobs (representing lantana flowers)
    fx, fy = x + random.randint(-r//2, r//2), y + random.randint(-r//2, r//2)
    draw.ellipse((fx - 5, fy - 5, fx + 5, fy + 5), fill="purple")

img.save("test_drone_lantana.png")
print("Saved test_drone_lantana.png")

