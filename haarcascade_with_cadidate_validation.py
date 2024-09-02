import cv2
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from fer import FER
import face_recognition

# Function to plot and display pie charts
def plot_pie_chart(data_dict, title):
    labels = list(data_dict.keys())
    percentages = list(data_dict.values())
    fig, ax = plt.subplots()
    wedges, texts, autotexts = ax.pie(
        percentages, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired(np.linspace(0, 1, len(labels))),
        textprops=dict(color="w"), pctdistance=0.85
    )
    ax.legend(wedges, [f'{label}: {pct:.1f}%' for label, pct in zip(labels, percentages)],
              title=title, loc="center left", bbox_to_anchor=(1, 0.5), borderaxespad=0.1)
    for text in texts:
        text.set_size(10)
    for autotext in autotexts:
        autotext.set_size(8)
    ax.axis('equal')
    plt.title(f'{title} Distribution')
    plt.show()

# Path to the video file
video_path = r'C:\Users\Microsoft\Downloads\sample_video_1.mp4'
cap = cv2.VideoCapture(video_path)

# Path to the reference image of the candidate
reference_image_path = r'C:\Users\Microsoft\Downloads\sample_video_candidate1.png'
reference_image = face_recognition.load_image_file(reference_image_path)
reference_encoding = face_recognition.face_encodings(reference_image)[0]

# Load Haar Cascade for face detection
haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
emotion_detector = FER()

# Initialize counters and variables
emotion_counters = Counter()
movement_counters = Counter({'Straight': 0})
previous_position = None
movement_threshold = 5
vertical_threshold = 10
candidate_detected = False
frame_count = 0
skip_frames = 2  # Skipping fewer frames as Haar Cascade is faster

# Reduced the frame size for faster processing
resize_width, resize_height = 640, 480

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % skip_frames != 0:
        continue

    # Resize frame for faster processing
    frame = cv2.resize(frame, (resize_width, resize_height))

    # Haar Cascade face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = haar_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]

        # Face recognition to validate the candidate
        face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_encodings = face_recognition.face_encodings(face_img_rgb)

        if face_encodings:
            candidate_encoding = face_encodings[0]
            match = face_recognition.compare_faces([reference_encoding], candidate_encoding)[0]

            if match:
                candidate_detected = True
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Detect head movement
                movement_label = ""
                if previous_position is not None:
                    prev_x, prev_y = previous_position
                    dx, dy = x - prev_x, y - prev_y
                    if abs(dx) <= movement_threshold and abs(dy) <= 2 * vertical_threshold:
                        movement_label = "Straight"
                    elif abs(dx) > movement_threshold:
                        movement_label = "Right" if dx > 0 else "Left"
                    elif abs(dy) > 2 * vertical_threshold:
                        movement_label = "Down" if dy > 0 else "Up"
                    movement_counters[movement_label] += 1

                previous_position = (x, y)

                # Analyze facial expressions
                emotions = emotion_detector.detect_emotions(face_img)
                if emotions:
                    emotion_scores = emotions[0]['emotions']
                    top_expression = max(emotion_scores, key=emotion_scores.get)
                    emotion_label = f"{top_expression}: {emotion_scores[top_expression]:.2f}"

                    # Update emotion counter
                    emotion_counters[top_expression] += 1

                    # Print labels on the right side of the face detection box
                    cv2.putText(frame, emotion_label, (x + w + 10, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 139), 2)
                    cv2.putText(frame, movement_label, (x + w + 10, y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 139), 2)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

if candidate_detected:
    # Plot and display the pie charts for the candidate
    total_emotions = sum(emotion_counters.values())
    emotion_percentages = {emotion: count / total_emotions * 100 for emotion, count in emotion_counters.items()}

    total_movements = sum(movement_counters.values())
    movement_percentages = {movement: count / total_movements * 100 for movement, count in movement_counters.items()}

    plot_pie_chart(emotion_percentages, "Emotions")
    plot_pie_chart(movement_percentages, "Head Movements")
else:
    print("Candidate not detected in the video.")
