import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("runs/segment/my_yolo11_seg_new/results.csv")

plt.figure(figsize=(8, 5))
plt.plot(df["epoch"], df["metrics/mAP50(B)"], label="Box mAP50 (B)")
plt.plot(df["epoch"], df["metrics/mAP50(M)"], label="Mask mAP50 (M)")

plt.title("mAP50 по эпохам")
plt.xlabel("Эпохи")
plt.ylabel("mAP50")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("map50_curves.png", dpi=300)
plt.show()
