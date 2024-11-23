import cv2
import numpy as np

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(img, lines):
    if lines is None:
        return img
    blank = np.zeros_like(img)
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(blank, (x1, y1), (x2, y2), (0, 255, 0), 5)
    return cv2.addWeighted(img, 0.8, blank, 1, 0)

def process_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    height, width = frame.shape[:2]
    roi_vertices = [(0, height),(width // 2, height // 2),(width, height)]
    cropped_edges = region_of_interest(edges, np.array([roi_vertices], np.int32))
    lines = cv2.HoughLinesP(cropped_edges, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=150)
    lane_frame = draw_lines(frame, lines)
    return lane_frame

def main():
    video_capture = cv2.VideoCapture(0)
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break
        processed_frame = process_frame(frame)
        cv2.imshow('Lane Detection', processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()