import cv2
import customtkinter as ctk
from PIL import Image, ImageTk
from deepface import DeepFace
import threading

# Create the app
app = ctk.CTk()
app.title("Emotion Detection App")

# Create a button to close the app
def close_app():
    global update_frame
    update_frame = False
    app.destroy()

button = ctk.CTkButton(app, text="Close App", command=close_app)
button.grid(row=0, column=0, sticky="w", columnspan=2)

# Create a "Test Button"
test_button = ctk.CTkButton(app, text="Upload New Data")
test_button.grid(row=0, column=1, sticky="w")

# Create a frame for the video
frame = ctk.CTkLabel(app)
frame.grid(row=1, column=0, rowspan=7, columnspan=2)

# Create a dictionary to store the progress bars and labels for each emotion
emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
progress_bars = {}
emotion_labels = {}

for i, emotion in enumerate(emotions):
    label = ctk.CTkLabel(app, text=emotion)
    label.grid(row=i+2, column=2)
    emotion_labels[emotion] = label
    progress_bar = ctk.CTkProgressBar(app, mode='determinate')
    progress_bar.grid(row=i+2, column=3)
    progress_bars[emotion] = progress_bar

# Create labels for dominant_gender and dominant_emotion
dominant_gender_label = ctk.CTkLabel(app, text="")
dominant_gender_label.grid(row=1, column=2)


# Start video capture
cap = cv2.VideoCapture(0)
# Create a variable to control the update of the frame
update_frame = True

# Create a function to detect emotions continuously
def detect_emotion():
    while update_frame:
        ret, img = cap.read()
        
        # Convert the frame to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(35, 35))
        # Draw a rectangle around each face
        for i, (x, y, w, h) in enumerate(faces):
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img, f"Face {i+1}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        # If faces are detected, analyze the frame
        if len(faces) >0:
            results = DeepFace.analyze(img, actions=['emotion', 'gender'], enforce_detection=False)
            if isinstance(results, list):
                for i, result in enumerate(results):
                    # Update the dominant_gender and dominant_emotion labels
                    dominant_gender_label.configure(text=f"Face {i+1}: {result['dominant_gender']}, {result['dominant_emotion']}")
                    # Update the progress bars with the emotion values
                    for emotion, progress_bar in progress_bars.items():
                        progress_bar.set(result['emotion'][emotion])
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which tkinter uses)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Convert the Image object into a TkPhoto object
        im = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=im)

        # Put the image into the label widget
        frame.imgtk = imgtk
        frame.configure(image=imgtk)

# Start detecting emotions automatically in a separate thread
threading.Thread(target=detect_emotion, daemon=True).start()

# Create a variable to store the visibility state of the status bar
status_bar_visible = False

# Create a function that toggles the visibility of the status bar
def toggle_status_bar():
    global status_bar_visible
    status_bar_visible =  not status_bar_visible
    for emotion in emotions:
        if not status_bar_visible:
            progress_bars[emotion].grid()  # Show the progress bar
            emotion_labels[emotion].grid()  # Show the label
        else:
            progress_bars[emotion].grid_remove()  # Hide the progress bar
            emotion_labels[emotion].grid_remove()  # Hide the label

toggle_status_bar()
# Create a button that calls this function when clicked
toggle_button = ctk.CTkButton(app, text="View Details", command=toggle_status_bar)
toggle_button.grid(row=1, column=5)

app.mainloop()

# Release the capture when app is closed
cap.release()