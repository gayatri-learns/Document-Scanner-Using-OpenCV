import cv2
import numpy as np
from PIL import Image

# Read and resize image
img = cv2.imread("bill-book.jpg")

if img is None:
    print("Image not found!")
    exit()

img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
orig = img.copy()

# Preprocessing

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Edge detection (improved)
edges = cv2.Canny(blur, 75, 200)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
edges = cv2.dilate(edges, kernel, iterations=1)
edges = cv2.erode(edges, kernel, iterations=1)

cv2.imshow("Edges", edges)


# Find document contour

contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)

largest = None

for cnt in contours[:5]:
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

    if 4 <= len(approx) <= 8:
        largest = cv2.convexHull(approx)
        break

if largest is None:
    print("❌ Document not detected")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    exit()

# Debug: draw detected contour
debug = orig.copy()
cv2.drawContours(debug, [largest], -1, (0, 255, 0), 3)
cv2.imshow("Detected Document", debug)
cv2.waitKey(0)


# Order points
def order_points(pts):
    pts = pts.reshape(-1, 2)
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

rect = order_points(largest)

(tl, tr, br, bl) = rect


# Perspective transform
widthA = np.linalg.norm(br - bl)
widthB = np.linalg.norm(tr - tl)
maxWidth = int(max(widthA, widthB))

heightA = np.linalg.norm(tr - br)
heightB = np.linalg.norm(tl - bl)
maxHeight = int(max(heightA, heightB))

dst = np.array([
    [0, 0],
    [maxWidth - 1, 0],
    [maxWidth - 1, maxHeight - 1],
    [0, maxHeight - 1]
], dtype="float32")

M = cv2.getPerspectiveTransform(rect, dst)
scan = cv2.warpPerspective(orig, M, (maxWidth, maxHeight))

cv2.imshow("Warped", scan)
cv2.waitKey(0)

# Convert to black & white
gray_scan = cv2.cvtColor(scan, cv2.COLOR_BGR2GRAY)

bw = cv2.adaptiveThreshold(
    gray_scan,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    15,
    5
)

cv2.imshow("Final Scan", bw)
cv2.waitKey(0)


# Save output
cv2.imwrite("Scanned.jpg", bw)

pil_img = Image.fromarray(bw)
pil_img = pil_img.convert("RGB")
pil_img.save("Scanned_Document.pdf")

print("Scanned document saved as JPG and PDF")

cv2.destroyAllWindows()
