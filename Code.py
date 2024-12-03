import cv2
import os
import pandas as pd
import matplotlib.pyplot as plt
# Create a directory for saving processed images
output_dir = "processed_images"
os.makedirs(output_dir, exist_ok=True)

# Prepare a list to store the results
results = []

# Loop through all images in the working directory
for filename in os.listdir():
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):  # Process only image files
        # Read the image
        image = cv2.imread(filename)
        if image is None:
            print(f"Skipping {filename}, could not load the image.")
            continue

        # Step 1: Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(os.path.join(output_dir, f"{filename}_step1_gray.jpg"), gray)

        # Step 2: Apply Gaussian Blur
        blur = cv2.GaussianBlur(gray, (11, 11), 0)
        cv2.imwrite(os.path.join(output_dir, f"{filename}_step2_blur.jpg"), blur)
        plt.imshow(blur, cmap='gray')


        # Step 3: Apply Canny Edge Detection
        canny = cv2.Canny(blur, 30, 150, 3)
        cv2.imwrite(os.path.join(output_dir, f"{filename}_step3_canny.jpg"), canny)


        # Step 4: Apply Dilation
        dilated = cv2.dilate(canny, (1, 1), iterations=0)
        cv2.imwrite(os.path.join(output_dir, f"{filename}_step4_dilated.jpg"), dilated)


        # Step 5: Find Contours and Draw Them
        (cnt, hierarchy) = cv2.findContours(
            dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.drawContours(rgb, cnt, -1, (0, 255, 0), 2)
        cv2.imwrite(os.path.join(output_dir, f"{filename}_step5_contours.jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


        # Display the final image
        plt.imshow(rgb)
        plt.show()


        # Count the number of objects (seeds)
        seed_count = len(cnt)

        # Append the result (filename and seed count) to the results list
        results.append({"Filename": filename, "Number of Objects": seed_count})

# Export the results to a CSV file
results_df = pd.DataFrame(results)
results_csv_path = "analysis_results.csv"
results_df.to_csv(results_csv_path, index=False)

print(f"Analysis complete. Results saved to {results_csv_path}. Processed images are in the '{output_dir}' folder.")
