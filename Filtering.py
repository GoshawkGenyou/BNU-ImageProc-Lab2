import numpy as np
import os
from PIL import Image
from matplotlib import pyplot as plt


def my_histeq(image, gray_levels):
    size = image.shape
    # flattens image from 3 layered to 1 layered and then creates histogram using built in np functions
    histogram, bins = np.histogram(image.flatten(), bins=256, range=(0, 256))
    cdf = histogram.cumsum()

    # normalize to desired gray_levels
    cdf_normalized = (cdf - cdf.min()) / (cdf.max() - cdf.min()) * (gray_levels - 1)
    cdf_normalized = np.round(cdf_normalized).astype(np.uint8)  # Convert to integers

    # Uses numpy inbuilt indexing to perform essentially output[i, j] = cdf_normalized[image[i, j]] very fast
    output = cdf_normalized[image]
    return output


def hist_eq(image_name):
    folder_name = 'result'
    if os.path.exists(image_name):
        image = np.array(Image.open(image_name))
    else:
        raise FileNotFoundError

    n_1 = 2
    n_2 = 64
    n_3 = 256
    n_levels = [n_1, n_2, n_3]
    images = [my_histeq(image, level) for level in n_levels]
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' created.")

    name = image_name.split('.')[0]
    for i, img in enumerate(images):
        pil_img = Image.fromarray(img)

        # Construct the full path for each image

        image_path = os.path.join(folder_name, f'{name}_eq_{n_levels[i]}.jpg')
        pil_img.save(image_path)
        print(f"Saved image to {image_path}")

    # Create a figure for displaying images
    plt.figure(figsize=(10, 8))

    # Display the original image
    plt.subplot(221)
    plt.imshow(image, cmap='gray')
    plt.title('原图')
    plt.axis('on')

    # Display the histogram equalized image with n=2
    plt.subplot(222)
    plt.imshow(images[0], cmap='gray')
    plt.title(f'直方图均衡化, n={n_levels[0]}')
    plt.axis('on')

    # Display the histogram equalized image with n=64
    plt.subplot(223)
    plt.imshow(images[1], cmap='gray')
    plt.title(f'直方图均衡化, n={n_levels[1]}')
    plt.axis('on')

    # Display the histogram equalized image with n=256
    plt.subplot(224)
    plt.imshow(images[2], cmap='gray')
    plt.title(f'直方图均衡化, n={n_levels[2]}')
    plt.axis('on')

    plt.tight_layout()
    plt.show()


from scipy.ndimage import convolve


def avg_filter(image, kernel=None):
    size = image.shape

    if kernel is None:
        kernel = np.ones((3, 3)) / 9

    # reflect ensures proper border handling
    output = convolve(image, kernel, mode='reflect')
    return output


from scipy.ndimage import median_filter


def med_filter(image, kernel_size=3):
    """
    Kernel size can be a tuple or an int. Irregular shapes use kernel size such as (5, 3).
    median_filter is a scipy written function, and is much more efficiently implemented than I can ever implement.
    """
    return median_filter(image, kernel_size, mode='reflect')


def iter_filter(image, filter_func, max_iter=500, threshold=1e-3):
    """
    checks for if the image manages to converge using a default threshold value of 1e-3 difference between the input image
    and the output

    :param image: input image
    :param filter_func: function used to filter, can be average, median or otherwise
    :param max_iter: maximum number of iterations to attempt
    :param threshold: value of difference to check for
    :return: filtered image, iteration count
    """
    prev_image = image.astype(np.float64)
    for iteration in range(max_iter):
        # Apply the filter
        current_image = filter_func(prev_image)

        # Compute the difference between current and previous images
        difference = np.abs(current_image - prev_image)
        difference_norm = np.linalg.norm(difference)

        # Check if the difference is below the threshold (indicating convergence)
        if difference_norm / np.linalg.norm(prev_image) < threshold:
            print(f"Converged after {iteration + 1} iterations.")
            return current_image.astype(np.uint8), iteration + 1

        # Update for the next iteration
        prev_image = current_image

    print(f"Reached max iterations ({max_iter}) without full convergence.")
    return current_image.astype(np.uint8), iteration



def filters(image_name):
    folder_name = 'result'
    if os.path.exists(image_name):
        image = np.array(Image.open(image_name))
    else:
        raise FileNotFoundError

    image_avg = iter_filter(image, avg_filter)
    image_median = iter_filter(image, med_filter)


    # save the image
    pil_img = Image.fromarray(image_avg[0])
    # Construct the full path for each image
    image_path = os.path.join(folder_name, f'{image_name}_a.jpg')
    pil_img.save(image_path)
    print(f"Saved sharpened image to {image_path}")
    pil_img = Image.fromarray(image_median[0])
    # Construct the full path for each image
    image_path = os.path.join(folder_name, f'{image_name}_m.jpg')
    pil_img.save(image_path)
    print(f"Saved sharpened image to {image_path}")

    plt.figure(1)

    # Subplot 1: Original Image
    plt.subplot(1, 3, 1)  # 1 row, 3 columns, 1st subplot
    plt.imshow(image, cmap='gray')
    plt.title('原图')
    plt.axis('on')  # Show axis

    # Subplot 2: Mean Filtered Image
    plt.subplot(1, 3, 2)  # 1 row, 3 columns, 2nd subplot
    plt.imshow(image_avg[0], cmap='gray')
    plt.title(f'均值滤波 iter={image_avg[1]}')
    plt.axis('on')  # Show axis

    # Subplot 3: Median Filtered Image
    plt.subplot(1, 3, 3)  # 1 row, 3 columns, 3rd subplot
    plt.imshow(image_median[0], cmap='gray')
    plt.title(f'中值滤波 iter={image_median[1]}')
    plt.axis('on')  # Show axis

    # Show the figure
    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()  # Display the plot


def my_sharpen(image, kernel, iter=3):
    weight_original = 0.9
    weight_sharpened = 1 - weight_original
    for _ in range(iter):
        sharpened = convolve(image, kernel, mode='constant')
        image = np.clip((weight_original * image + weight_sharpened * sharpened), 0, 255).astype(np.uint8)

    return image


def sharpen(image_name):
    folder_name = 'result'
    if os.path.exists(image_name):
        image = np.array(Image.open(image_name))
    else:
        raise FileNotFoundError
    # 8 directional laplace kernel
    kernel = np.array([[0, 1, 0],
                       [1, -4, 1],
                       [0, 1, 0]])
    # Compute the enhanced image using numpy's weighted addition
    enhanced = my_sharpen(image, kernel, 1)

    # Convert the result back to a Pillow Image

    # save the image
    pil_img = Image.fromarray(enhanced)
    # Construct the full path for each image
    image_path = os.path.join(folder_name, f'{image_name}_s.jpg')
    pil_img.save(image_path)
    print(f"Saved sharpened image to {image_path}")

    # Create the figure
    plt.figure(1)

    # Plot the original image
    plt.subplot(121)
    plt.imshow(image, cmap='gray')
    plt.title('原图')
    plt.axis('on')  # Turn on the axis

    # Plot the sharpened image
    plt.subplot(122)
    plt.imshow(enhanced, cmap='gray')
    plt.title('图像锐化')
    plt.axis('on')  # Turn on the axis

    # Show the plot
    plt.show()


if __name__ == '__main__':
    plt.rcParams['font.family'] = 'SimHei'  # Use 'SimHei', 'Microsoft YaHei', etc.
    plt.rcParams['axes.unicode_minus'] = False  # Ensure minus sign is displayed correctly
    # Images are 3 dimensional arrays (y, x, c) of height and width and value c representing the specific pixel
    # information. c may or may not be an array. I will assume grayscale and intensity for simplicity.
    image_file1 = "bridge.jpg"
    image_file2 = "circuit.jpg"
    image_file3 = "moon.jpg"

    #hist_eq(image_file1)
    #filters(image_file2)
    sharpen(image_file3)

