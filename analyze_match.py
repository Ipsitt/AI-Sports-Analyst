import cv2
import numpy as np
from sklearn.cluster import KMeans
from ultralytics import YOLO
from collections import defaultdict
import math
import pytesseract
import pandas as pd
import os
from tkinter import Tk, Label, Entry, Button

last_player_id = None
last_team_id = None
last_ball_pos = None
last_touch_time = None

weights = {}

def start_analysis():
    global weights
    try:
        weights['goals'] = int(goal_entry.get())
        weights['touches'] = int(touch_entry.get())
        weights['passes'] = int(pass_entry.get())
        root.destroy()
    except ValueError:
        print("❌ Please enter valid numbers.")

root = Tk()
root.title("Match Analysis Config")

Label(root, text="Enter Weight for Goals:").grid(row=0, column=0)
goal_entry = Entry(root)
goal_entry.grid(row=0, column=1)
goal_entry.insert(0, "100")

Label(root, text="Enter Weight for Touches:").grid(row=1, column=0)
touch_entry = Entry(root)
touch_entry.grid(row=1, column=1)
touch_entry.insert(0, "1")

Label(root, text="Enter Weight for Passes:").grid(row=2, column=0)
pass_entry = Entry(root)
pass_entry.grid(row=2, column=1)
pass_entry.insert(0, "2")

Button(root, text="Start", command=start_analysis).grid(row=3, columnspan=2, pady=10)
root.mainloop()

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture("footage2.mp4")
frame_count = 0

color_stats = defaultdict(lambda: {"touches": 0, "goals": 0, "passes": 0})
jersey_samples = []
kmeans = None
color_sample_frames = 30

last_touch_time_per_player = {}
last_team_touch = None

score_roi_coords = (120, 40, 70, 55)
tess_config = "--psm 7 -c tessedit_char_whitelist=0123456789-"
prev_score = (0, 0)

fixed_colors = {
    "Team-1": (128, 255, 255),
    "Team-2": (255, 0, 255),
}

def dist(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def predict_mvp(stats):
    scores = {}
    for team, s in stats.items():
        score = (s["goals"] * weights["goals"] +
                 s["touches"] * weights["touches"] +
                 s["passes"] * weights["passes"])
        scores[team] = score
    return max(scores, key=scores.get) if scores else "N/A"

def draw_stats_panel(frame, stats, kmeans):
    panel_width = 200
    height, width, _ = frame.shape
    extended_frame = np.zeros((height, width + panel_width, 3), dtype=np.uint8)
    extended_frame[:, :width] = frame

    panel_x = width + 10
    y = 30

    cv2.putText(extended_frame, "Live Stats", (panel_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    y += 30

    if kmeans:
        for i in range(2):
            team = f"Team-{i+1}"
            color = fixed_colors[team]
            cv2.putText(extended_frame, f"{team}", (panel_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y += 25
            cv2.putText(extended_frame, f"  Touches: {stats[team]['touches']}", (panel_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y += 20
            cv2.putText(extended_frame, f"  Goals:   {stats[team]['goals']}", (panel_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y += 20
            cv2.putText(extended_frame, f"  Passes:  {stats[team]['passes']}", (panel_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y += 30

        mvp_team = predict_mvp(stats)
        cv2.putText(extended_frame, "MVP Prediction:", (panel_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        y += 30
        cv2.putText(extended_frame, f"Likely: {mvp_team}", (panel_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, fixed_colors.get(mvp_team, (255, 255, 255)), 1)

    return extended_frame

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    frame_resized = cv2.resize(frame.copy(), (640, 360))
    results = model(frame_resized, verbose=False)

    player_boxes, ball_box = [], None

    for det in results[0].boxes:
        cls = int(det.cls[0])
        x1, y1, x2, y2 = map(int, det.xyxy[0])
        conf = float(det.conf[0])
        if conf < 0.4:
            continue
        if cls == 0:
            player_boxes.append((x1, y1, x2, y2))
        elif cls == 32:
            ball_box = (x1, y1, x2, y2)

    if frame_count <= color_sample_frames:
        for (x1, y1, x2, y2) in player_boxes:
            h = y2 - y1
            jersey_area = frame_resized[y1:y1 + int(0.3 * h), x1:x2]
            if jersey_area.size == 0:
                continue
            avg_color = cv2.mean(jersey_area)[:3]
            jersey_samples.append(avg_color)

    if frame_count == color_sample_frames and len(jersey_samples) >= 2:
        jersey_samples = np.array(jersey_samples)
        kmeans = KMeans(n_clusters=2, n_init=10, random_state=0)
        kmeans.fit(jersey_samples)
        print("\n✅ Jersey color clustering completed.")

    if frame_count % 10 == 0:
        x, y, w, h = score_roi_coords
        roi = frame[y:y+h, x:x+w]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        text = pytesseract.image_to_string(thresh, config=tess_config).strip()

        if "-" in text:
            try:
                a, b = map(lambda s: int(s.strip()), text.split("-"))
                if (a, b) != prev_score:
                    print(f"Score changed from {prev_score} to {(a,b)}")
                    if a > prev_score[0]:
                        color_stats["Team-2"]["goals"] += 1
                    elif b > prev_score[1]:
                        color_stats["Team-1"]["goals"] += 1
                    prev_score = (a, b)
            except:
                pass

    ball_center = None
    if ball_box:
        bx1, by1, bx2, by2 = ball_box
        ball_center = ((bx1 + bx2) // 2, (by1 + by2) // 2)
        cv2.rectangle(frame_resized, (bx1, by1), (bx2, by2), (255, 255, 0), 2)

    current_team_touch = None

    if kmeans and ball_center:
        for (x1, y1, x2, y2) in player_boxes:
            h = y2 - y1
            jersey_area = frame_resized[y1:y1 + int(0.3 * h), x1:x2]
            if jersey_area.size == 0:
                continue

            avg_color = np.array(cv2.mean(jersey_area)[:3]).reshape(1, -1)
            cluster = kmeans.predict(avg_color)[0]
            team_key = f"Team-{cluster+1}"

            px = (x1 + x2) // 2
            py = (y1 + y2) // 2
            distance = dist((px, py), ball_center)
            player_id = (x1, y1, x2, y2)

            color = fixed_colors.get(team_key, (255, 255, 255))
            cv2.rectangle(frame_resized, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame_resized, team_key, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            import time

            now = time.time()

            if distance < 26:
                current_team_touch = team_key
            
                time_diff = now - last_touch_time if last_touch_time else 999
                ball_moved = dist(ball_center, last_ball_pos) if last_ball_pos else 999

                last_time = last_touch_time_per_player.get(player_id, 0)
                if now - last_time > 2.5:
                    color_stats[team_key]["touches"] += 1
                    last_touch_time_per_player[player_id] = now

                    if (last_player_id and last_player_id != player_id and
                        last_team_id == team_key and
                        time_diff < 3 and
                        ball_moved > 5):
                        color_stats[team_key]["passes"] += 1

                    last_player_id = player_id
                    last_team_id = team_key
                    last_touch_time = now
                    last_ball_pos = ball_center

    last_team_touch = current_team_touch

    final_display = draw_stats_panel(frame_resized, color_stats, kmeans)
    cv2.imshow("Match Analysis + Live Stats", final_display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

output_dir = r"C:\Users\IPSIT\Desktop\hackathon\output"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "match_stats.csv")

mvp_team = predict_mvp(color_stats)

data = []
for team, stats in color_stats.items():
    data.append({
        "Team": team,
        "Touches": stats["touches"],
        "Goals": stats["goals"],
        "Passes": stats["passes"],
        "MVP": "1" if team == mvp_team else ""
    })

df = pd.DataFrame(data)
df.to_csv(output_file, index=False)
print(f"\n📁 Exported match stats to: {output_file}")
