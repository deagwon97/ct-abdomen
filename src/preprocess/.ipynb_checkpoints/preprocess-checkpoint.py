import numpy as np

def normalize(image):
    # 입력 받은 numpy array를 0.3 quantile과 0.99 quantile을 기준으로 min max scaleing
    min_val = np.quantile(image.reshape(-1), 0.3)
    max_val = np.quantile(image.reshape(-1), 0.99)
    # 1차 threshold 적용                   
    over_idx  = (image > max_val)
    under_idx = (image < min_val)
    image[over_idx]  = max_val
    image[under_idx] = min_val
    
    normalized = (image - min_val) / (max_val - min_val)# * 255    
    return normalized

def apply_margin(image, margin):
    # 입력 받은 이미지의 상하좌우로 margin을 적용하고
    # 원본 크기로 resize하여 반환
    image_margin = np.zeros([512, 512, image.shape[2]])
    for z in range(image.shape[2]):
        sample = image[margin:-margin,margin:-margin, z].copy()
        sample = sample.astype(np.uint8)
        image_margin[:,:,z] = cv2.resize(sample, (512, 512))
    return image_margin    