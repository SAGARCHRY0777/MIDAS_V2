import cv2
import os

def stitch_images_vertically(depth_image_path_prefix, original_image_path_prefix, output_dir, num_images=30):
    """
    Stitches depth images with their corresponding original images vertically (one over another).

    Args:
        depth_image_path_prefix (str): Path to the directory containing depth images.
        original_image_path_prefix (str): Path to the directory containing original images.
        output_dir (str): Directory to save the stitched images.
        num_images (int): Number of images to stitch.
    """

    os.makedirs(output_dir, exist_ok=True)

    for i in range(num_images):
        depth_image_filename = os.path.join(depth_image_path_prefix, f"depth_{i:06d}.png")
        original_image_filename = os.path.join(original_image_path_prefix, f"{i:06d}.png")
        output_filename = os.path.join(output_dir, f"stitched_vertical_{i:06d}.png")

        if not os.path.exists(depth_image_filename):
            print(f"Warning: Depth image not found: {depth_image_filename}. Skipping.")
            continue
        if not os.path.exists(original_image_filename):
            print(f"Warning: Original image not found: {original_image_filename}. Skipping.")
            continue

        try:
            depth_img = cv2.imread(depth_image_filename)
            original_img = cv2.imread(original_image_filename)

            if depth_img is None:
                print(f"Error: Could not load depth image: {depth_image_filename}. Skipping.")
                continue
            if original_img is None:
                print(f"Error: Could not load original image: {original_image_filename}. Skipping.")
                continue

            # Ensure depth image has 3 channels for concatenation (if it's grayscale)
            if len(depth_img.shape) == 2:
                depth_img = cv2.cvtColor(depth_img, cv2.COLOR_GRAY2BGR)

            # Resize depth image to match original image's width for vertical stitching
            # The height will adjust proportionally
            depth_img_resized = cv2.resize(depth_img, (original_img.shape[1], int(depth_img.shape[0] * (original_img.shape[1] / depth_img.shape[1]))))

            # Stitch images vertically (one over another)
            stitched_img = cv2.vconcat([original_img, depth_img_resized])

            cv2.imwrite(output_filename, stitched_img)
            print(f"Stitched and saved: {output_filename}")

        except Exception as e:
            print(f"An error occurred while processing image {i}: {e}")

if __name__ == "__main__":
    depth_image_dir = r"D:\spa\SFA3D\transformers_midas\test_results"
    original_image_dir = r"D:\spa\SFA3D\dataset\kitti\training\image_2"
    output_stitch_dir = r"D:\spa\SFA3D\transformers_midas\stitched_dense_images"

    stitch_images_vertically(depth_image_dir, original_image_dir, output_stitch_dir, num_images=30)
    print("\nStitching process complete.")