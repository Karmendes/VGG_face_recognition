# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from pre_process import pre_process_frame
from imutils import paths
import cv2
import matplotlib.pyplot as plt


# %%
detector = cv2.CascadeClassifier('detector.xml')


# %%
teste = list(paths.list_images("Test/"))
treino = list(paths.list_images("Train/"))


# %%
for path in teste:
    pixels = cv2.imread(path)
    results = pre_process_frame(pixels,detector)
    if len(results) > 0:
        x1, y1, width, height = results
        x2, y2 = x1 + width, y1 + height
        face = pixels[y1:y2, x1:x2]
        name = path.split(os.path.sep)[-2]
        name_path = path.split(os.path.sep)[-1]
        if os.listdir('Test_face').count(name) < 1:
            os.mkdir(f'Test_face/{name}/')
        cv2.imwrite(f'Test_face/{name}/{name_path}',face)


# %%
for path in treino:
    pixels = cv2.imread(path)
    results = pre_process_frame(pixels,detector)
    if len(results) > 0:
        x1, y1, width, height = results
        x2, y2 = x1 + width, y1 + height
        face = pixels[y1:y2, x1:x2]
        name = path.split(os.path.sep)[-2]
        name_path = path.split(os.path.sep)[-1]
        if os.listdir('Train_face').count(name) < 1:
            os.mkdir(f'Train_face/{name}/')
        cv2.imwrite(f'Train_face/{name}/{name_path}',face)


