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
    def __init__(self, width=680, height=340, nail_dir='./guding'):
        self.width = width
        self.height = height

        # city name
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

        # plate code
        self.valid_chars = list("ABCDEFGHJKLMNPQRSTUVWXYZ")
        self.valid_numbers = list("0123456789")
        self.valid_char_set = set(self.valid_chars) | set(self.valid_numbers)
        self.code_fonts = ['msgothic.ttc']
        self.code_font_set = set(self.code_fonts)
        self.code_locate = (0.065, 0.48)
        self.code_width = 0.87

        # qr
        self.qr_styles = {
            'left': dict(locate=(0.85, 0.15), width=0.1, height=0.2),
            'right': dict(locate=(0.05, 0.15), width=0.1, height=0.2)
        }

        # fixed nail
        self.nails = []
        for name in os.listdir(nail_dir):
            if 'mask' in name:
                continue
            img = cv2.imread(osp.join(nail_dir, name))
            mask = cv2.imread(
                osp.join(nail_dir, name.replace('.png', '_mask1.png')), 0)
            self.nails.append((img, mask))

        
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

    def create_mounting_hole(self, draw, mask):
        """
        1st mounting hole: cx=0.237 cy=0.114 width=0.06~0.16 height=0.07
        2nd mounting hole: cx=0.763 cy=0.114 width=0.06~0.16 height=0.07
        """
        # outer border

        w = random.choice(
            range(round(self.width * 0.06), round(self.width * 0.16 + 1)))
        h = round(self.height * 0.07)
        cx1 = round(self.width * 0.237)
        cx2 = self.width - cx1
        cy = round(self.height * 0.114)
        start_x1 = cx1 - w // 2
        start_x2 = cx2 - w // 2
        start_y = cy - h // 2
        radius = h // 2
        
        draw.rounded_rectangle(
            (start_x1, start_y, start_x1+w, start_y+h), fill=(0, 0, 0), 
            outline=(193, 189, 186), width=round(self.width*0.003), radius=radius
        )
        mask.rounded_rectangle(
            (start_x1, start_y, start_x1+w, start_y+h), fill=0, 
            outline=255, width=1, radius=radius
        )
        draw.rounded_rectangle(
            (start_x2, start_y, start_x2+w, start_y+h), fill=(0, 0, 0), 
            outline=(193, 189, 186), width=round(self.width*0.003), radius=radius
        )
        mask.rounded_rectangle(
            (start_x2, start_y, start_x2+w, start_y+h), fill=0, 
            outline=255, width=1, radius=radius
        )
        # print(f"w={w} h={h} sx1={start_x1} sx2={start_x2} sy={start_y} r={radius}")

        hole1 = [start_x1, start_y, w, h]
        hole2 = [start_x2, start_y, w, h]
        return hole1, hole2

    def create_fixed_nail(self, img, mask, hole, nail=None):
        """
        size = 2 * holw_h ~ 3 * hole_h
        """
        hole_lx, hole_ty, hole_w, hole_h = hole
        ds = 3 * hole_h
        
        if nail is None:
            nail_img, nail_mask = random.choice(self.nails)

            # 随机选取角度，随中心点旋转
            angle = random.randint(0, 360)
            src_h, src_w = nail_img.shape[:2]
            M = cv2.getRotationMatrix2D((src_w / 2, src_h / 2), angle, 1)
            cos = np.abs(M[0, 0])
            sin = np.abs(M[0, 1])
            dst_w = src_h * sin + src_w * cos
            dst_h = src_h * cos + src_w * sin
            M[0, 2] += (dst_w - src_w) * 0.5
            M[1, 2] += (dst_h - src_h) * 0.5
            nail_img = cv2.warpAffine(nail_img, M, (round(dst_w), round(dst_h)))
            nail_mask = cv2.warpAffine(nail_mask, M, (round(dst_w), round(dst_h)))

            # 缩放至目标尺寸
            nail_img = cv2.resize(nail_img, (ds, ds))
            nail_mask = cv2.resize(nail_mask, (ds, ds))

            nail_mask = np.expand_dims((nail_mask / 255), 2).repeat(3, axis=2)
        else:
            nail_img, nail_mask = nail

        # 将固定钉图片贴到车牌图上
        cx = random.randint(
            hole_lx + hole_h // 2, hole_lx + hole_w - hole_h // 2)
        cy = hole_ty + hole_h // 2
        lx = cx - ds // 2
        ty = cy - ds // 2
        img[ty:ty+ds, lx:lx+ds, :] = (img[ty:ty+ds, lx:lx+ds, :] * \
            (1 - nail_mask) + nail_img * nail_mask).astype(np.uint8)
        
        # 更新车牌 mask 上固定钉位置的值
        mask[ty:ty+ds, lx:lx+ds] = (mask[ty:ty+ds, lx:lx+ds] * \
            (1 - nail_mask[:,:,0]) + 255 * nail_mask[:,:,0]).astype(np.uint8)
        
        return img, mask, (nail_img, nail_mask)


    def create_image(self, city_name=None, city_font=None, code=None, 
                     code_font=None, inner_border=True, save_path='output.jpg'):
        plate_img = Image.fromarray(
            np.full((self.height, self.width, 3), 0, dtype=np.uint8))
        plate_mask = Image.fromarray(
            np.full((self.height, self.width), 0, dtype=np.uint8))
        plate_draw = ImageDraw.Draw(plate_img)
        mask_draw = ImageDraw.Draw(plate_mask)
         
        self.create_border(plate_draw, mask_draw)
        hole1, hole2 = self.create_mounting_hole(plate_draw, mask_draw)
        city = self.create_city(plate_draw, city_name, city_font)
        code = self.create_code(plate_draw, code, code_font)
    
        plate_img = np.array(plate_img)
        plate_mask = np.array(plate_mask)
        plate_img, plate_mask, nail = self.create_fixed_nail(
            plate_img, plate_mask, hole1)
        plate_img, plate_mask, nail = self.create_fixed_nail(
            plate_img, plate_mask, hole2, nail)
        content = f'{city}·{code}'
        plate_img = self.create_qr(plate_img, content)
         
        cv2.imwrite(save_path, plate_img)
        cv2.imwrite(save_path.replace('.jpg', '-mask.jpg'), plate_mask)
        # with open(save_path.replace('.jpg', '.txt'), 'w') as f:
        #    f.write(content)


def main():
    pg = PlateGenerator()
    for i in tqdm(range(200)):
        save_path=f'test_out/{i:0>6d}.jpg'
        pg.create_image(save_path=save_path)
    

if __name__ == "__main__":
    main()
