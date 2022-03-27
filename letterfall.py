from argparse import ArgumentParser
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np



parser = ArgumentParser()
parser.add_argument("img")
parser.add_argument("out")

args = parser.parse_args()
img = cv2.imread(args.img)
grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
split = (grey > 57).astype(np.uint8)

n, labels, stats, centers = cv2.connectedComponentsWithStats(split, 8, cv2.CV_32S)
letters = []

for i in range(n):
    x, y, w, h, area = stats[i]

    if area < 100:
        letter = img[y:y+h, x:x+w]
        letter = cv2.cvtColor(letter, cv2.COLOR_BGR2BGRA)
        mask = split[y:y+h, x:x+w]

        img[y:y+h, x:x+w][mask != 0] = (63, 57, 54)
        letter[mask == 0] = 0
        letter = np.pad(letter, ((20, 20), (20, 20), (0, 0)))
        
        letters.append((Image.fromarray(letter), x-20, y-20))

new_img = Image.fromarray(img)

for l, x, y in letters:
    l = l.rotate(np.random.randint(181), Image.BILINEAR)
    new_img.paste(l, (x+np.random.randint(-3, 4), y+np.random.randint(-3, 4)), l)

new_img.save(args.out)
