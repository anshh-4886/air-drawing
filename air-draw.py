import cv2
import numpy as np

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Define the range of color to track (HSV format)
lower_color = np.array([0, 120, 70])
upper_color = np.array([10, 255, 255])

# Create a blank canvas to draw on
drawing_canvas = np.zeros((480, 640, 3), dtype=np.uint8)

# Initialize a list to keep track of points for smoothing
points = []

# Function to interpolate points
def interpolate_points(points, num_points):
    if len(points) < 2:
        return points
    
    interpolated_points = []
    for i in range(len(points) - 1):
        start_point = points[i]
        end_point = points[i + 1]
        num_interpolated = num_points // (len(points) - 1)
        for j in range(num_interpolated):
            alpha = j / num_interpolated
            x = int((1 - alpha) * start_point[0] + alpha * end_point[0])
            y = int((1 - alpha) * start_point[1] + alpha * end_point[1])
            interpolated_points.append((x, y))
    interpolated_points.append(points[-1])  # Add the last point
    return interpolated_points

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a mask for the specified color
    mask = cv2.inRange(hsv, lower_color, upper_color)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Clear the points list
    current_points = []

    # Process each contour
    for contour in contours:
        if cv2.contourArea(contour) > 1000:  # Filter out small contours
            M = cv2.moments(contour)
            if M['m00'] != 0:
                cX = int(M['m10'] / M['m00'])
                cY = int(M['m01'] / M['m00'])
                current_points.append((cX, cY))

    # Update the list of points with the current points
    points.extend(current_points)

    # Limit the number of points to avoid overflow
    if len(points) > 200:
        points = points[-200:]

    # Smooth drawing using interpolation
    if len(points) > 1:
        smooth_points = interpolate_points(points, len(points) * 5)
        for i in range(1, len(smooth_points)):
            cv2.line(drawing_canvas, smooth_points[i - 1], smooth_points[i], (0, 255, 0), 5)

    # Display the drawing canvas
    cv2.imshow('Drawing Canvas', drawing_canvas)

    # Display the original frame for reference
    cv2.imshow('Frame', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
