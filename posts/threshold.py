import cv2
import numpy as np
from PIL import Image

VERTICAL: int = 0
HORIZONTAL: int = 1


def crop_grip_image(img: Image.Image) -> Image.Image:
    return img.crop((100, 100, 700, 600))  # Optimized


##JPG_推論_20191130_01_ローカル環境版で追加
def polyfit_1d_arr(x,img_arr):
    #x = np.array([i for i in range(img_arr.shape[0])])
    std_arr = np.std(img_arr)
    res_tmp = np.polyfit(x, img_arr, 1)
    y_tmp = np.poly1d(res_tmp)(x)
    img_arr_copy =  img_arr.copy()
    img_arr_copy[np.abs(img_arr_copy - y_tmp) > std_arr] =y_tmp[np.abs(img_arr_copy - y_tmp) > std_arr]
    res = np.polyfit(x, img_arr_copy, 1)
    y = np.poly1d(res)(x)
    return x, y_tmp, y


def check_area_ocupaied_by_blue_bg(img_rgb,scan_direction):
    indexes_aggregated = []#返り値用
    if scan_direction == VERTICAL:
        each_line_drection = HORIZONTAL
        for k in range(img_rgb.shape[scan_direction]):
            pass
    elif scan_direction == HORIZONTAL:
        each_line_drection = VERTICAL
        for k in range(img_rgb.shape[scan_direction]):
            indexex_R = np.where(img_rgb[:,k,0]==255)
            indexex_G = np.where(img_rgb[:,k,1]==0)
            indexex_B = np.where(img_rgb[:,k,2]==0)
            indexes = np.intersect1d(indexex_R, np.intersect1d(indexex_G,indexex_B))
            indexes_aggregated.append(indexes)# np.intersect1dでRGBのインデックスについて論理積をとるように変更
    else:
        print('スキャン例外')

    return indexes_aggregated


#青色背景単色化したものを利用して、赤白黒ケーブル領域取得
def inner_cables_area_check_with_blue_background(img_rgb):
    """
    Args:
        img_rgb (np.ndarray): (H, W, C)のケーブル近くの領域のみをcropした画像
    """
    y_red_list = []
    y_green_list = []
    y_blue_list = []
    img_red, img_green, img_blue = cv2.split(img_rgb)
    for k in range(img_red.shape[HORIZONTAL]):
        x = np.array([i for i in range(img_red[:,0].shape[0])])

        x, _, y_red = polyfit_1d_arr(x, img_red[:,k])
        x, _, y_green = polyfit_1d_arr(x, img_green[:,k])
        x, _, y_blue = polyfit_1d_arr(x, img_blue[:,k])

        y_red_list.append(y_red)
        y_green_list.append(y_green)
        y_blue_list.append(y_blue)

    y_red_arr = np.array(y_red_list)
    y_green_arr = np.array(y_green_list)
    y_blue_arr = np.array(y_blue_list)

    y_red_arr = y_red_arr.T
    y_green_arr =  y_green_arr.T
    y_blue_arr = y_blue_arr.T    

    img_blue_bg_only = img_rgb.copy()

    img_blue_bg_only[(img_blue_bg_only[:,:,0]<img_blue_bg_only[:,:,1]) & 
                (img_blue_bg_only[:,:,1]<img_blue_bg_only[:,:,2]) & 
                (img_blue_bg_only[:,:,0]< y_red_arr + 40) &
                (img_blue_bg_only[:,:,1]>y_green_arr - 40)&
                (img_blue_bg_only[:,:,1]<y_green_arr + 40)&
                (img_blue_bg_only[:,:,2]>y_blue_arr -40)&
                (img_blue_bg_only[:,:,2]<y_blue_arr +40)] = [0,0,255] 

    red_thres = 15

    img_red_only = img_blue_bg_only.copy()

    img_red_only[~((img_red_only[:,:,0] == 0) &
                 (img_red_only[:,:,1] == 0) &
                   (img_red_only[:,:,2] == 255)) &
                (img_red_only[:,:,0]>img_red_only[:,:,1]+red_thres) & 
                (img_red_only[:,:,0]>img_red_only[:,:,2]+red_thres)] = [255,0,0]

    indexes_aggregated_red = check_area_ocupaied_by_blue_bg(img_red_only,HORIZONTAL)#青色単色背景版

    img_black_only = img_blue_bg_only.copy()

    img_black_only[~((img_black_only[:,:,0] == 0) &
                   (img_black_only[:,:,1] == 0)&
                   (img_black_only[:,:,2] == 255))&
                (img_black_only[:,:,0]<100) &
                (img_black_only[:,:,1]<100)&
                (img_black_only[:,:,2]<100)&
                (img_black_only[:,:,0]<=img_black_only[:,:,1]+red_thres) &
                (img_black_only[:,:,0]<=img_black_only[:,:,2]+red_thres)] = [255,0,0]

    indexes_aggregated_black = check_area_ocupaied_by_blue_bg(img_black_only,HORIZONTAL)#青色単色背景版
    img_white_only = img_blue_bg_only.copy()
    img_white_only[~((img_white_only[:,:,0] == 0) &
                   (img_white_only[:,:,1] == 0) &
                   (img_white_only[:,:,2] == 255)) &
                    (img_white_only[:,:,0]>=100) &
                    (img_white_only[:,:,1]>=100) &
                    (img_white_only[:,:,2]>=100)&
                    (img_white_only[:,:,0]<=img_white_only[:,:,1]+red_thres) &
                    (img_white_only[:,:,0]<=img_white_only[:,:,2]+red_thres)] = [255,0,0]

    indexes_aggregated_white = check_area_ocupaied_by_blue_bg(img_white_only,HORIZONTAL)#青色単色背景版

    img_gray_only= img_blue_bg_only.copy()
    img_gray_only[
            ~((img_gray_only[:,:,0] == 0) &
             (img_gray_only[:,:,1] == 0)&
             (img_gray_only[:,:,2] == 255))&
            (img_gray_only[:,:,0]>50) &
            (img_gray_only[:,:,0]<120) &
            (img_gray_only[:,:,1]>50) &
            (img_gray_only[:,:,1]<120) &
            (img_gray_only[:,:,2]>50) &
            (img_gray_only[:,:,2]<120) 
              ] = [255,0,0]

    indexes_aggregated_gray = np.unique((np.where(img_gray_only[:,-1,:]==[255,0,0]))[0])

    indexes_aggregated_bg_only = check_area_ocupaied_by_blue_bg(img_blue_bg_only,HORIZONTAL)#青色単色背景版

    if len(indexes_aggregated_red) == len(indexes_aggregated_black) == len(indexes_aggregated_white) == len(indexes_aggregated_bg_only):
        pixel_colors_aggregated = []
        for i in range(len(indexes_aggregated_red)):
            pixel_colors_aggregated_each_line = []
            for idx_each in range(img_rgb.shape[VERTICAL]):#上のcheck_area_ocupaied(img_white_only,HORIZONTAL)と連動
                pixel_colors_aggregated_each_line.append('-1')#初期化　該当色が無い指標として,''-1'（string）を使用
            
            for idx_each in range(len(indexes_aggregated_red[i])):
                pixel_colors_aggregated_each_line[indexes_aggregated_red[i][idx_each]] = 'red'
            for idx_each in range(len(indexes_aggregated_black[i])):
                pixel_colors_aggregated_each_line[indexes_aggregated_black[i][idx_each]] = 'black'
            for idx_each in range(len(indexes_aggregated_white[i])):
                pixel_colors_aggregated_each_line[indexes_aggregated_white[i][idx_each]] = 'white'
            for idx_each in range(len(indexes_aggregated_bg_only[i])):
                pixel_colors_aggregated_each_line[indexes_aggregated_bg_only[i][idx_each]] = 'bg_blue'            
            pixel_colors_aggregated.append(pixel_colors_aggregated_each_line)
    else:
        print('各indexes_aggregatedの要素数不一致')

    return img_blue_bg_only, indexes_aggregated_red, indexes_aggregated_black, indexes_aggregated_white, indexes_aggregated_gray, indexes_aggregated_bg_only, pixel_colors_aggregated


if __name__ == '__main__':
    pil_img: Image.Image = Image.open('../semseg-grip-image/VOCDataset/JPEGImages/cable_rotate2_20200521_125458_190deg.jpg').convert('RGB')
    pil_resized: Image.Image = crop_grip_image(pil_img)
    img_arr: np.ndarray = np.array(pil_resized)
    img_blue_bg_only, indexes_aggregated_red, indexes_aggregated_black, indexes_aggregated_white, indexes_aggregated_gray, indexes_aggregated_bg_only, pixel_colors_aggregated = inner_cables_area_check_with_blue_background(img_arr)
    print(indexes_aggregated_white, indexes_aggregated_black, indexes_aggregated_red, indexes_aggregated_gray)
