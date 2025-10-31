import pandas as pd
import matplotlib.pyplot as plt

# Загружаем CSV с метриками
df = pd.read_csv("runs/segment/my_yolo11_seg_new/results.csv")

plt.figure(figsize=(8, 5))
plt.plot(df["epoch"], df["train/box_loss"], label="box_loss")
plt.plot(df["epoch"], df["train/seg_loss"], label="seg_loss")
plt.plot(df["epoch"], df["train/cls_loss"], label="cls_loss")

plt.title("Графики потерь во время обучения")
plt.xlabel("Эпохи")
plt.ylabel("Значение потерь (Loss)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_curves.png", dpi=300)
plt.show()
