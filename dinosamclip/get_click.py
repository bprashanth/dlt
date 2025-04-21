import matplotlib.pyplot as plt
import cv2

img = cv2.imread("lantana.jpg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def onclick(event):
    print(f"Clicked at x={int(event.xdata)}, y={int(event.ydata)}")

fig, ax = plt.subplots()
ax.imshow(img_rgb)
cid = fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()
