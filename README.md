# Document-Scanner-Using-OpenCV
A Python-based document scanner that detects documents from images, applies perspective correction, and generates scan-like black &amp; white output using classical computer vision techniques.

✨ Features
Automatic document detection using edge and contour analysis
Perspective (top-down) transformation
Adaptive thresholding for scan-like black & white output
Export scanned document as JPG and PDF
Debug visualizations for edge detection and contour selection

🛠️ Technologies Used
Python
OpenCV
NumPy
Pillow (PIL)

🔄 Processing Pipeline
Read and resize input image
Convert to grayscale and apply Gaussian blur
Detect edges using Canny edge detection
Identify the largest document-like contour
Approximate and order corner points
Apply perspective transformation
Convert output to black & white using adaptive thresholding
Save scanned document as image and PDF

⚠️ Limitations
This project uses classical computer vision techniques, which have inherent limitations:
Cannot guarantee 100% accuracy under all lighting conditions
Performance may degrade with:
Shadows
Low contrast backgrounds
Curved or folded documents
Highly cluttered scenes
Contour-based detection may fail if document edges are not clearly visible
These limitations highlight why modern production systems often combine classical CV with machine learning–based document detection.

🎯 Learning Outcomes
Practical understanding of image preprocessing techniques
Experience with contour detection and polygon approximation
Hands-on use of perspective transformations
Insight into real-world constraints of classical computer vision approaches
