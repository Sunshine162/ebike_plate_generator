from copy import deepcopy
import math
import os
import os.path as osp
import random

from PIL import ImageFont, ImageDraw, Image
import cv2
import numpy as np
import qrcode
from qrcode.image.styledpil import StyledPilImage
from qrcode.image.styles.moduledrawers import SquareModuleDrawer
from tqdm import tqdm


class PlateGenerator:
    def __init__(self, width=680, height=340):
        self.width = width
        self.height = height

        self.valid_cities = [
            "广州", "深圳", "珠海", "汕头", "韶关", "河源", "梅州", "惠州", "汕尾",
            "东莞", "中山", "江门", "佛山", "阳江", "湛江", "茂名", "肇庆", "清远", 
            "潮州", "揭阳", "云浮"
        ]
        self.valid_city_set = set(self.valid_cities)
        self.city_fonts = ['Dengb.ttf', 'simsun.ttc']
        self.city_font_set = set(self.city_fonts)
        self.city_locate = (0.33, 0.15)
        self.city_size = (0.34, 0.22)

        self.valid_chars = list("ABCDEFGHJKLMNPQRSTUVWXYZ")
        self.valid_numbers = list("0123456789")
        self.valid_char_set = set(self.valid_chars) | set(self.valid_numbers)
        self.code_fonts = ['msgothic.ttc']
        self.code_font_set = set(self.code_fonts)
        self.code_locate = (0.065, 0.48)
        self.code_width = 0.87

        self.qr_styles = {
            'left': dict(locate=(0.85, 0.15), width=0.1, height=0.2),
            'right': dict(locate=(0.05, 0.15), width=0.1, height=0.2)
        }
        
    def create_city(self, draw, city_name=None, font=None):
        """
        1st char: left=0.33 top=0.15 width=0.11 height=0.22
        2nd char: left=0.57 top=0.15 width=0.11 height=0.22
        """
        if city_name is not None:
            assert city_name in self.valid_city_set, \
                f"Only support cities: {self.valid_cities}"
        else:
            city_name = random.choice(self.valid_cities)

        if font is not None:
            assert font in self.city_font_set, \
                f"Only support this fonts for city name: {self.city_fonts}"
        else:
            font = random.choice(self.city_fonts)

        font = ImageFont.truetype(font, 80)

        y_start = round(self.height * self.city_locate[1])
        y_start -= font.font.getsize(city_name)[1][1]

        c1, c2 = city_name
        c1_x_start = round(self.width * self.city_locate[0])
        c1_x_start -= font.font.getsize(c1)[1][0]
        draw.text((c1_x_start, y_start), text=c1, font=font, fill=(0, 0, 0, 0))

        c2_width = font.font.getsize(c2)[0][0] - font.font.getsize(c2)[1][0]
        c2_x_start = round(
            self.width * (self.city_locate[0] + self.city_size[0])) - c2_width
        draw.text((c2_x_start, y_start), text=c2, font=font, fill=(0, 0, 0, 0))

        return city_name
        
    def create_qr(self, img, content, style=None):
        """
        style1: left=0.85 top=0.15 width=0.1 height=0.2
        style1: left=0.05 top=0.15 width=0.1 height=0.2
        """
        qr = qrcode.QRCode(
            error_correction=qrcode.constants.ERROR_CORRECT_H,
            version=4.5,
            box_size=2,
            border=0,
        )
        qr.add_data(content)
        qr.make(fit=True)
        qr_img = qr.make_image(
            image_factory=StyledPilImage, module_drawer=SquareModuleDrawer())
        qr_img = np.array(qr_img.get_image())
        cv2.copyMakeBorder(
            qr_img, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        qr_width, qr_height = qr_img.shape[:2]

        if style is None:
            style = random.choice(list(self.qr_styles))
        else:
            assert style in self.qr_styles
        x_start = round(self.width * self.qr_styles[style]['locate'][0])
        y_start = round(self.height * self.qr_styles[style]['locate'][1])
        img[y_start:y_start+qr_height, x_start:x_start+qr_width, :] = qr_img

        return img

    def create_code(self, draw, code=None, font=None):
        """
        left=0.065 top=0.48 width=0.87 height=0.4
        """
        if code is not None:
            assert len(code) == 6
            code_set = set(code)
            assert code_set.issubset(self.valid_char_set), \
                f"code include invalid characters: {code_set - self.valid_char_set}"
        else:
            num_char = random.choice([1, 2])
            code = random.sample(self.valid_chars, num_char)
            code += random.sample(self.valid_numbers, 6 - num_char)
            random.shuffle(code)
            code = ''.join(code)
        if font is not None:
            assert font in self.code_font_set, \
                f"Only support this fonts for code: {self.code_fonts}"
        else:
            font = random.choice(self.code_fonts)
        
        font = ImageFont.truetype(font, 168)
        (text_width, text_height), (offset_x, offset_y) = font.font.getsize(code)
        span = max(0, (self.width * self.code_width - text_width) / 5)
        x_start = round(self.width * self.code_locate[0])
        y_start = round(self.height * self.code_locate[1]) - offset_y
        for i, c in enumerate(code):
            draw.text((x_start, y_start), text=c, font=font, fill=(0, 0, 0, 0))
            x_start += font.font.getsize(c)[0][0] + span
        
        return code

    def create_border(self, draw, mask, inner=None):
        """
        outer border: width=0.0075/0.015 start=0
        inner border: width=0.0125/0.02  start=0.025/0.05
        """
        # outer border
        draw.rounded_rectangle(
            (0, 0, self.width-1, self.height-1), fill=(255, 255, 255), 
            outline=(193, 189, 186), width=int(self.width*0.0075), radius=17
        )
        mask.rounded_rectangle(
            (0, 0, self.width-1, self.height-1), fill=255, 
            outline=255, width=1, radius=17
        )
        draw.rounded_rectangle(
            (1, 1, self.width-2, self.height-2), fill=(255, 255, 255), 
            outline=(139, 132, 126), width=1, radius=17
        )
        draw.rounded_rectangle(
            (2, 2, self.width-3, self.height-3), fill=(255, 255, 255), 
            outline=(139, 132, 126), width=1, radius=17
        )
        
        if inner is None:
            inner = random.choice([True, False])
        if not inner:
            return

        # inner border
        l = round(0.025 * self.width)
        r = self.width - l
        t = round(0.05 * self.height)
        b = self.height - t
        draw.rounded_rectangle(
            (l, t, r, b), fill=(255, 255, 255), outline=(139, 132, 126), 
            width=int(self.width*0.0125), radius=17)        
        draw.rounded_rectangle(
            (l+1, t+1, r-1, b-1), fill=(255, 255, 255), outline=(0, 0, 0), 
            width=int(self.width*0.0125/2), radius=17) 
    
    def create_image(self, city_name=None, city_font=None, code=None, 
                     code_font=None, inner_border=True, save_path='output.jpg'):
         img = Image.fromarray(
            np.full((self.height, self.width, 3), 0, dtype=np.uint8))
         mask = Image.fromarray(
            np.full((self.height, self.width), 0, dtype=np.uint8))
         draw = ImageDraw.Draw(img)
         mask_ = ImageDraw.Draw(mask)
         
         self.create_border(draw, mask_)
         city = self.create_city(draw, city_name, city_font)
         code = self.create_code(draw, code, code_font)
         
         img = np.array(img)
         content = f'{city}·{code}'
         img = self.create_qr(img, content)
         
         cv2.imwrite(save_path, img)
         cv2.imwrite(save_path.replace('.jpg', '-mask.jpg'), np.array(mask))
         with open(save_path.replace('.jpg', '.txt'), 'w') as f:
            f.write(content)


class ImageGenerator:
    def __init__(self, image_dir, plate_dir):
        self.images = [
            osp.join(image_dir, file_name) 
            for file_name in os.listdir(image_dir)
        ]
        self.plates = [
            osp.join(plate_dir, file_name) for file_name in os.listdir(plate_dir) 
            if file_name.endswith('.jpg') and not file_name.endswith('-mask.jpg')
        ]
    
    def plate_perspective(self, src, mask=None, chage_range=0.05):
        h, w, c = src.shape
        
        # 透视前后的四个角点：左上、右上、左下、右下
        pts1 = np.float32([[0, 0], [w-1, 0], [0, h-1], [w-1, h-1]])
        offset = np.hstack([
            np.random.randint(-w * chage_range, w * chage_range, (4, 1)),
            np.random.randint(-h * chage_range, h * chage_range, (4, 1))
        ])
        pts2 = pts1 + offset
        min_x = pts2[:, 0].min()
        min_y = pts2[:, 1].min()
        pts2[:, 0] -= min_x
        pts2[:, 1] -= min_y
        pts2 = pts2.astype(np.float32)
        width, height = pts2.max(axis=0).astype(np.int64)

        M = cv2.getPerspectiveTransform(pts1, pts2)
        dst = cv2.warpPerspective(src, M, (width, height))
        if mask is not None:
            mask = cv2.warpPerspective(mask, M, (width, height))

        return dst, mask
    
    def plate_resize(self, src, mask=None, scale=None, interpolation=None):
        if scale is None:
            scale = np.random.uniform(low=0.1, high=0.6, size=None)
        if interpolation is None:
            # interpolation = cv2.INTER_NEAREST   # 精度较差，速度快
            # interpolation = cv2.INTER_LINEAR    # 避免锯齿现象
            interpolation = cv2.INTER_AREA      # 推荐图片缩小时使用
            # interpolation = cv2.INTER_CUBIC     # 推荐图片放大时使用
            # interpolation = cv2.INTER_LANCZOS4  

        dst = cv2.resize(src, None, fx=scale, fy=scale, 
                         interpolation=interpolation)
        if mask is not None:
            mask = cv2.resize(mask, None, fx=scale, fy=scale, 
                              interpolation=interpolation)
        return dst, mask

    def create_background(self, src):
        wh_ratio = 16 / 9
        src_h, src_w = src.shape[:2]
        try:
            if math.isclose(src_w / src_h, wh_ratio, rel_tol=1e-6):
                # print('A')
                crop_ratio = np.random.uniform(low=0.7, high=1, size=None)
                dst_w = round(src_w * crop_ratio)
                dst_h = round(dst_w * 9 / 16)
                start_x = np.random.randint(0, src_w - dst_w)
                start_y = np.random.randint(0, src_h - dst_h)

            elif src_w / src_h > wh_ratio:
                # print('B')
                dst_w = round(src_h * wh_ratio)
                dst_h = src_h
                start_x = np.random.randint(0, src_w - dst_w)
                start_y = 0
                
            else:
                # print('C')
                dst_w = src_w
                dst_h = round(src_w * 9 / 16)
                start_x = 0
                start_y = np.random.randint(0, src_h - dst_h)
        except Exception as e:
            print(e)
            print(f"src_w={src_w} src_h={src_h} dst_w={dst_w} dst_h={dst_h}")
            exit()

        dst = src[start_y:start_y+dst_h, start_x:start_x+dst_w, :]
        # print(f"src=({src.shape}) dst=({dst.shape} dst_w={dst_w} dst_h={dst_h} x={start_x} y={start_y}")
        return cv2.resize(dst, (1280, 720), interpolation=cv2.INTER_CUBIC)
    
    def adjust_brightness(self, src, alpha=None):
        if alpha is None:
            alpha = crop_ratio = np.random.uniform(low=0.5, high=1.5, size=None)
        dst = np.clip(src * alpha, 0, 255)
        return dst
    
    def plate_paste(self, src, plate, mask, loc=None):
        h, w, c = plate.shape
        if loc is None:
            x = np.random.randint(0, src.shape[1] - w)
            y = np.random.randint(0, src.shape[0] - h)
        else:
            x, y = loc

        dst = deepcopy(src)

        # create mask
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        _, fg_mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
        _, bg_mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY_INV)

        # image fusion
        roi = dst[y:y+h, x:x+w, :]
        fg = cv2.bitwise_and(plate, plate, mask=fg_mask)
        bg = cv2.bitwise_and(roi, roi, mask=bg_mask)
        dst[y:y+h, x:x+w, :] = cv2.add(fg, bg)

        return dst

    def create_data(self, image_path=None, plate_path=None, crop_ratio=None, 
                    save_path=None):
        if not image_path:
            image_path = random.choice(self.images)
        img = cv2.imread(image_path)
        img = self.create_background(img)

        if not plate_path:
            plate_path = random.choice(self.plates)
        plate = cv2.imread(plate_path)
        mask = cv2.imread(plate_path.replace('.jpg', '-mask.jpg'))
        plate, mask = self.plate_perspective(plate, mask)
        plate, mask = self.plate_resize(plate, mask)
        img = self.plate_paste(img, plate, mask)

        img = self.adjust_brightness(img)
        cv2.imwrite(save_path, img)


def main():
    # pg = PlateGenerator()
    # for i in tqdm(range(10000)):
    #     save_path=f'data/plates/{i:0>6d}.jpg'
    #     pg.create_image(save_path=save_path)
    
    ig = ImageGenerator('data/backgrounds', 'data/plates')
    for i in tqdm(range(100)):
        save_path=f'data/images/{i:0>8d}.jpg'
        ig.create_data(save_path=save_path)
    

if __name__ == "__main__":
    main()
