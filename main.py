import numpy as np
from PIL import Image
from sklearn.cluster import KMeans


def image_rect_range(size, step):
    top = 0
    while top <= size[1]:
        bottom = min(top + step[1], size[1])
        if top == bottom:
            break
        left = 0
        while left <= size[0]:
            right = min(left + step[0], size[0])
            if left == right:
                break
            # left, upper, right, lower
            yield (left, top, right, bottom)
            left = right
        top = bottom


def main():
    # image = Image.open("Zephyr-Cove-563A9095-ap.jpg")
    # image = Image.open("Tahoe-East-Shore-Trail-563A9233.jpg")
    image = Image.open("Galena-Falls-Trail-563A9339-ap.jpg")
    print(f"image size: {image.size}")
    # image.show()

    width = 200
    height = 200

    step_size = (width, height)
    for rect in image_rect_range(image.size, step_size):
        # print(f"rect: {rect}")

        # left, top, right, bottom
        crop = image.crop(rect)

        img_arr = np.array(crop)
        # print(f"First array shape {arr.shape}")

        flattened_arr = img_arr.reshape(-1, 3)
        # print(f"Reshaped array {x.shape}")

        # kmeans = KMeans(init="k-means++", n_clusters=8).fit(flattened_arr)
        kmeans = KMeans(init="k-means++", n_clusters=8).fit(flattened_arr)
        clustered_data = kmeans.cluster_centers_[kmeans.labels_]

        segmented_arr = clustered_data.reshape(img_arr.shape).astype(np.uint8)

        # print(f"Reshaped segmented array {segmented_img.shape}")

        restored_img = Image.fromarray(segmented_arr)

        image.paste(restored_img, rect)

    image.show()


if __name__ == "__main__":
    main()
