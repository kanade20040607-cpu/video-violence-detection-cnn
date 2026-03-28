import cv2
import os
import time


# ========= 1. 数据保存路径 =========
BASE_DIR = "data/train"
NORMAL_DIR = os.path.join(BASE_DIR, "normal")
ABNORMAL_DIR = os.path.join(BASE_DIR, "abnormal")

os.makedirs(NORMAL_DIR, exist_ok=True)
os.makedirs(ABNORMAL_DIR, exist_ok=True)


# ========= 2. 摄像头初始化 =========
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera")

print("===================================")
print("Data Collection Started")
print("Press:")
print("  n → save NORMAL image")
print("  a → save ABNORMAL image")
print("  q → quit")
print("===================================")

normal_count = len(os.listdir(NORMAL_DIR))
abnormal_count = len(os.listdir(ABNORMAL_DIR))


# ========= 3. 主循环 =========
while True:
    ret, frame = cap.read()
    if not ret:
        break

    display = frame.copy()
    cv2.putText(
        display,
        f"Normal: {normal_count}  Abnormal: {abnormal_count}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.imshow("Data Collection", display)

    key = cv2.waitKey(1) & 0xFF

    # 保存 normal
    if key == ord('n'):
        filename = f"normal_{normal_count}_{int(time.time())}.jpg"
        path = os.path.join(NORMAL_DIR, filename)
        cv2.imwrite(path, frame)
        normal_count += 1
        print(f"[NORMAL] Saved: {path}")

    # 保存 abnormal
    elif key == ord('a'):
        filename = f"abnormal_{abnormal_count}_{int(time.time())}.jpg"
        path = os.path.join(ABNORMAL_DIR, filename)
        cv2.imwrite(path, frame)
        abnormal_count += 1
        print(f"[ABNORMAL] Saved: {path}")

    # 退出
    elif key == ord('q'):
        break


# ========= 4. 释放资源 =========
cap.release()
cv2.destroyAllWindows()
print("Data collection finished.")
