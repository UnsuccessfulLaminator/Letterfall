from argparse import ArgumentParser
import cv2
from PIL import Image
import numpy as np
from tempfile import TemporaryDirectory, NamedTemporaryFile
import os
import subprocess as sp



parser = ArgumentParser()
parser.add_argument("img")
parser.add_argument("out")

args = parser.parse_args()
img = cv2.cvtColor(cv2.imread(args.img), cv2.COLOR_BGR2RGB)

bg_color = (54, 57, 63) # Discord background color
max_area = 350          # Groups with more pixels than this will not be affected
vel_std = 200           # Standard deviation of velocity components
angvel_std = 100        # Standard deviation of angular velocity

split = (np.abs(img-bg_color).sum(axis = 2) > 10).astype("u1")
n, labels, stats, centers = cv2.connectedComponentsWithStats(split, 4, cv2.CV_32S)
letters = []

for x, y, w, h, area in stats:
    if area <= max_area:
        letter = img[y:y+h, x:x+w]
        letter = cv2.cvtColor(letter, cv2.COLOR_RGB2RGBA)
        mask = split[y:y+h, x:x+w]

        img[y:y+h, x:x+w][mask != 0] = bg_color
        letter[mask == 0] = 0
        letter = np.pad(letter, ((20, 20), (20, 20), (0, 0)))
        
        letters.append([
            Image.fromarray(letter),         # Letter image with transparent bg
            np.array((x-20.0, y-20.0)),      # Initial position
            np.random.normal(0, vel_std, 2), # Initial velocity
            0,                               # Initial angle
            np.random.normal(0, angvel_std)  # Angular velocity
        ])

fps = 30                     # Output framerate
dur = 2                      # Output duration
gravity = np.array((0, 300)) # Gravitational force vector
dt = 1/fps

with TemporaryDirectory() as tempdir:
    files = []

    for j in range(fps*dur):
        new_img = Image.fromarray(img)

        for i in range(len(letters)):
            l_img, pos, vel, angle, angvel = letters[i]

            l_img = l_img.rotate(angle, Image.BILINEAR)
            new_img.paste(l_img, tuple(pos.round().astype(int)), l_img)

            pos += vel*dt
            vel += gravity*dt
            letters[i][3] += angvel*dt
        
        fname = os.path.join(tempdir, f"{j}.png")
        
        new_img.save(fname)
        files.append(fname)

    files = [os.path.join(tempdir, "0.png")]*fps+files # Add 1 sec of first frame at start
    
    if args.out.endswith(".gif"):
        # Use imagemagick because easy (slow as shit though)
        sp.run(["convert", "-delay", str(dt*100), "-loop", "0"]+files+[args.out])
    else:
        with NamedTemporaryFile() as f:
            for fname in files:
                f.write(bytes(f"file '{fname}'\n", "utf-8"))
                f.write(bytes(f"duration {dt}\n", "utf-8"))

            f.write(bytes(f"file '{fname}'\n", "utf-8")) # Last file again bc ffmpeg quirk

            sp.run([
                "ffmpeg", "-f", "concat", "-safe", "0","-i", f.name, "-vsync", "vfr",
                "-pix_fmt", "yuv420p", args.out
            ])
