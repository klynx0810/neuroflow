import numpy as np
import cv2 as cv
import random

def flip_image(img, horizontal=True, vertical=False):
    if horizontal and vertical:
        return cv.flip(img, -1)
    elif horizontal:
        return cv.flip(img, 1)
    elif vertical:
        return cv.flip(img, 0)
    return img

def rotate_image(img, angle=15):
    h, w = img.shape[:2]
    M = cv.getRotationMatrix2D((w/2, h/2), angle, 1)
    return cv.warpAffine(img, M, (w, h), borderMode=cv.BORDER_REFLECT)

def scale_and_translate(img, scale=1.2, tx=10, ty=10):
    h, w = img.shape[:2]
    M = np.array([[scale, 0, tx], [0, scale, ty]], dtype=np.float32)
    return cv.warpAffine(img, M, (w, h), borderMode=cv.BORDER_REFLECT)

def shear_image(img, shear_factor=0.2):
    h, w = img.shape[:2]
    M = np.array([[1, shear_factor, 0], [0, 1, 0]], dtype=np.float32)
    return cv.warpAffine(img, M, (w, h), borderMode=cv.BORDER_REFLECT)

def random_crop(img, crop_size):
    h, w = img.shape[:2]
    ch, cw = crop_size
    if h < ch or w < cw:
        raise ValueError("Crop size must be smaller than image size")
    if h == ch and w == cw:
        return img.copy()  # Không cần crop
    x = np.random.randint(0, w - cw + 1)
    y = np.random.randint(0, h - ch + 1)
    return img[y:y+ch, x:x+cw]

def adjust_brightness_contrast(img, brightness=0.2, contrast=0.2):
    alpha = 1.0 + random.uniform(-contrast, contrast)  # Contrast
    beta = 255.0 * random.uniform(-brightness, brightness)  # Brightness
    return cv.convertScaleAbs(img, alpha=alpha, beta=beta)

def adjust_saturation(img, saturation_factor=1.2):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] *= saturation_factor
    hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
    return cv.cvtColor(hsv.astype(np.uint8), cv.COLOR_HSV2BGR)

def add_gaussian_noise(img, mean=0, std=25):
    noise = np.random.normal(mean, std, img.shape).astype(np.float32)
    noisy = img.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

def add_salt_pepper_noise(img, amount=0.01):
    noisy = img.copy()
    num_salt = np.ceil(amount * img.size * 0.5).astype(int)
    num_pepper = np.ceil(amount * img.size * 0.5).astype(int)

    # Salt
    coords = [np.random.randint(0, i - 1, num_salt) for i in img.shape[:2]]
    noisy[coords[0], coords[1]] = 255

    # Pepper
    coords = [np.random.randint(0, i - 1, num_pepper) for i in img.shape[:2]]
    noisy[coords[0], coords[1]] = 0
    return noisy

def blur_image(img, ksize=5):
    return cv.GaussianBlur(img, (ksize, ksize), 0)

def barrel_distortion(img, strength=0.00001):
    h, w = img.shape[:2]
    K = np.array([[w, 0, w/2],
                  [0, h, h/2],
                  [0, 0, 1]], dtype=np.float32)
    
    # hệ số méo dạng: [k1, k2, p1, p2, k3]
    dist_coeff = np.array([strength, strength * 0.5, 0, 0, 0], dtype=np.float32)

    map1, map2 = cv.initUndistortRectifyMap(K, dist_coeff, None, K, (w, h), cv.CV_32FC1)
    distorted = cv.remap(img, map1, map2, interpolation=cv.INTER_LINEAR)
    return distorted

def add_light_leak(img, intensity=0.5, leak_color=(255, 200, 150)):
    h, w = img.shape[:2]
    leak = np.zeros_like(img, dtype=np.float32)

    # Tạo một vòng tròn sáng tại góc trái trên
    center = (random.randint(0, int(w * 0.3)), random.randint(0, int(h * 0.3)))
    radius = random.randint(int(min(h, w) * 0.3), int(min(h, w) * 0.6))

    overlay_color = np.array(leak_color, dtype=np.float32)
    for y in range(h):
        for x in range(w):
            dist = np.sqrt((x - center[0])**2 + (y - center[1])**2)
            factor = max(0, 1 - dist / radius)
            leak[y, x] = overlay_color * factor

    leak = (leak * intensity).astype(np.uint8)
    blended = cv.addWeighted(img, 1.0, leak, intensity, 0)
    return blended
