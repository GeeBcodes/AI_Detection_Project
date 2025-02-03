import cv2
import numpy as np
import os


def analyze_anatomical_inconsistencies(image):
    """
    Analyze anatomical and structural inconsistencies by detecting faces and eyes.
    Uses Haar Cascade classifiers to detect faces and eyes within the face region.
    If a detected face does not show exactly two eyes, that may be an indication of inconsistency.

    Returns:
        suspicion_score (float): A score between 0 (no suspicion) and 1 (high suspicion).
    """
    # Convert the image to grayscale (required for the Haar classifiers)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Load pre-trained Haar cascade classifiers for face and eye detection.
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    # Detect faces in the grayscale image.
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    suspicion = 0.0  # Accumulator for suspicion
    count_faces = 0  # Count of faces detected

    # For each detected face, detect eyes and update suspicion if the count isn't exactly two.
    for (x, y, w, h) in faces:
        count_faces += 1
        # Extract the region-of-interest (ROI) for the face.
        face_roi = gray[y:y + h, x:x + w]
        # Detect eyes within the face region.
        eyes = eye_cascade.detectMultiScale(face_roi, scaleFactor=1.1, minNeighbors=3)
        # If the number of eyes detected is not exactly 2, increase the suspicion.
        if len(eyes) != 2:
            suspicion += 1.0

    # If at least one face was detected, normalize the suspicion score.
    if count_faces > 0:
        suspicion_score = suspicion / count_faces
    else:
        # No faces detected – anatomical consistency cannot be evaluated.
        suspicion_score = 0.0

    return suspicion_score


def analyze_texture_details(image):
    """
    Analyze texture and detail issues using the variance of the Laplacian.
    The Laplacian operator highlights edges. Low variance indicates low edge information,
    which may be a sign of over-smoothing (a potential artifact of AI-generated images).

    Returns:
        suspicion (float): A score between 0 (normal texture) and 1 (over-smoothed).
    """
    # Convert image to grayscale for texture analysis.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Compute the Laplacian of the image.
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()  # Edge "energy" measure

    # Define an arbitrary threshold for low variance (over-smoothing).
    threshold = 100.0

    if variance < threshold:
        # The lower the variance, the higher the suspicion that the image is over-smoothed.
        suspicion = (threshold - variance) / threshold
        suspicion = np.clip(suspicion, 0.0, 1.0)
    else:
        suspicion = 0.0

    return suspicion


def analyze_color_inconsistencies(image):
    """
    Analyze color inconsistencies by measuring the correlation between color channels.
    Natural photographs usually have high correlation between red, green, and blue channels.
    Lower-than-expected correlation might indicate synthetic artifacts.

    Returns:
        suspicion_score (float): A score between 0 (normal) and 1 (high color inconsistency).
    """
    # Split the image into Blue, Green, and Red channels.
    (B, G, R) = cv2.split(image)

    # Flatten each channel to 1D arrays for correlation computation.
    B = B.flatten().astype(np.float32)
    G = G.flatten().astype(np.float32)
    R = R.flatten().astype(np.float32)

    # Compute pairwise correlation coefficients between channels.
    corr_bg = np.corrcoef(B, G)[0, 1]
    corr_br = np.corrcoef(B, R)[0, 1]
    corr_gr = np.corrcoef(G, R)[0, 1]

    # For natural images, correlations are generally high (e.g., above 0.9).
    threshold_corr = 0.9
    suspicion = 0.0

    # For each pair, if the correlation is below the threshold, add proportional suspicion.
    if corr_bg < threshold_corr:
        suspicion += (threshold_corr - corr_bg) / threshold_corr
    if corr_br < threshold_corr:
        suspicion += (threshold_corr - corr_br) / threshold_corr
    if corr_gr < threshold_corr:
        suspicion += (threshold_corr - corr_gr) / threshold_corr

    # Average the suspicion over the three comparisons.
    suspicion_score = suspicion / 3.0
    suspicion_score = np.clip(suspicion_score, 0.0, 1.0)
    return suspicion_score


def analyze_lighting(image):
    """
    Analyze lighting, shadows, and reflections by examining the brightness distribution.
    Natural images tend to have a balanced range of brightness and contrast.
    Extreme brightness or darkness, or very low/high contrast, might be a sign of AI generation.

    Returns:
        suspicion (float): A score between 0 (normal lighting) and 1 (abnormal lighting).
    """
    # Convert the image to grayscale to analyze intensity.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate the mean and standard deviation of brightness.
    mean_intensity = np.mean(gray)
    std_intensity = np.std(gray)

    suspicion = 0.0
    # Heuristic: if mean intensity is very low or very high, increase suspicion.
    if mean_intensity < 60 or mean_intensity > 190:
        suspicion += 0.5
    # Heuristic: if contrast (std. deviation) is too low or too high, increase suspicion.
    if std_intensity < 20 or std_intensity > 80:
        suspicion += 0.5

    # Ensure the suspicion score is between 0 and 1.
    suspicion = np.clip(suspicion, 0.0, 1.0)
    return suspicion


def analyze_image(image_path):
    """
    Main function to analyze an image for potential AI generation.
    Analyzes:
        - Anatomical and structural inconsistencies (face & eye detection)
        - Texture and detail issues (Laplacian variance)
        - Color inconsistencies (channel correlation)
        - Lighting, shadows, and reflections (brightness statistics)

    Prints individual suspicion scores and aggregates a final score.
    """
    # Load the image from the given path.
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to load image at", image_path)
        return

    # Analyze each trait and obtain scores between 0 and 1.
    anatomical_score = analyze_anatomical_inconsistencies(image)
    print(f"Anatomical Inconsistency Score: {anatomical_score:.2f}")

    texture_score = analyze_texture_details(image)
    print(f"Texture and Detail Score: {texture_score:.2f}")

    color_score = analyze_color_inconsistencies(image)
    print(f"Color Inconsistency Score: {color_score:.2f}")

    lighting_score = analyze_lighting(image)
    print(f"Lighting and Shadows Score: {lighting_score:.2f}")

    # Aggregate the scores (averaging here; weighting can be applied if needed).
    final_score = (anatomical_score + texture_score + color_score + lighting_score) / 4.0
    print(f"Final Suspicion Score: {final_score:.2f}")

    # Define a threshold to decide whether the image is likely AI-generated.
    threshold = 0.5  # This threshold is heuristic and may need tuning.
    if final_score >= threshold:
        print("Verdict: The image is likely AI-generated.\n")
    else:
        print("Verdict: The image is likely human-generated.\n")


if __name__ == "__main__":
    # Main loop: repeatedly ask the user to input an image filename until they choose to exit.
    while True:
        # Prompt the user to enter the image filename (assumed to be in the same folder as the script).
        image_filename = input("Enter the name of the image file (with extension, e.g., image.jpg): ").strip()

        # Construct the full path assuming the file is in the current directory.
        image_path = os.path.join(".", image_filename)

        # Check if the file exists before analyzing.
        if not os.path.isfile(image_path):
            print(f"Error: The file '{image_filename}' does not exist in the current directory.\n")
        else:
            # Analyze the image.
            analyze_image(image_path)

        # Ask the user if they want to analyze another image.
        user_choice = input("Would you like to analyze another image? (y/n): ").strip().lower()
        if user_choice not in ("y", "yes"):
            print("Exiting program.")
            break
