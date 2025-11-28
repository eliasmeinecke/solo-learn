
import cv2



def crop_image_around_gaze(img, gaze=(358,358), N=240):  # N = 336, size = 540 for actual data
    
    x_cor, y_cor = correct_gaze(gaze, N, img_size=716)

    crop_img = img[(y_cor - N//2):(y_cor + N//2), (x_cor - N//2):(x_cor + N//2)]
    return crop_img


def correct_gaze(gaze, N, img_size):
    
    x_g, y_g = gaze

    x_cor = x_g - max(0, x_g + N//2 - img_size) - min(0, x_g - N//2)
    y_cor = y_g - max(0, y_g + N//2 - img_size) - min(0, y_g - N//2)    
    return (x_cor, y_cor)


def draw_crop(img, gaze=(358,358), N=240):
  
    x_cor, y_cor = correct_gaze(gaze, N, img_size=716)

    cv2.rectangle(img, (x_cor - N//2, y_cor - N//2), (x_cor + N//2, y_cor + N//2), (255, 0, 0), 2)

    cv2.imshow('Crop', img)
    cv2.waitKey(0)


def show_cropped_img(img, gaze=(358,358), N=240):
    
    cropped_img = crop_image_around_gaze(img, gaze, N)
    cv2.imshow('Cropped Image', cropped_img)
    cv2.waitKey(0)




if __name__ == "__main__":
    img = cv2.imread("data/skating_panda.jpg")
    # draw_crop(img, gaze=(500, 500), N=240)
    show_cropped_img(img, gaze=(358, 358), N=380)

    