import cv2
import pytesseract

# Load image
image = cv2.imread('car_plate.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Use edge detection to find the contours of the plate
edged = cv2.Canny(gray, 30, 200)
contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

# Loop over the contours to find the best possible one for the plate
for c in contours:
    perimeter = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * perimeter, True)
    if len(approx) == 4:
        x, y, w, h = cv2.boundingRect(c)
        plate = gray[y:y + h, x:x + w]
        break

# Use pytesseract to do OCR on the cropped plate
text = pytesseract.image_to_string(plate, config='--psm 8')

print(f"Detected license plate Number is: {text.strip()}")

# Display the original image with detected plate
cv2.imshow('Detected Plate', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
