from collections import defaultdict

import cv2
import paddleocr
import linora as la

__all__ = ['OCRHouseholdCard']


class OCRHouseholdCard():
    def __init__(self, ocr=None):
        self.ocr = paddleocr.PaddleOCR(show_log=False) if ocr is None else ocr
        self._keys = ['household_type', 'household_name', 'household_id', 
                      'household_address']
        self._char_household_type = ['户别']
        self._char_household_name = ['户主姓名']
        self._char_household_id = ['户号']
        self._char_household_address = ['地址']
#         self._char_number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        
    def predict(self, image):
        self._axis = None
        self._error = 'ok'
        self._angle = -1
        self._mode = ''
        if isinstance(image, str):
            self._image = cv2.imread(image)
            self._image = cv2.cvtColor(self._image, cv2.COLOR_BGR2RGB)
            self._image = la.image.array_to_image(self._image)
#             image = la.image.read_image(image)
#             self._image = la.image.color_convert(image)
        else:
            self._image = image
        self._direction_transform(self._image)
        if isinstance(self._info, str):
            self._direction_transform(la.image.enhance_brightness(self._image, 0.8))
        if isinstance(self._info, str):
            return {'data':self._info, 'axis':[], 'angle':0, 'error':self._error}
        self._axis_transform_up()
        for i in self._info:
            if '图片模糊' in self._info[i]:
                self._temp_info = self._info.copy()
                self._temp_axis = self._axis.copy()
                self._direction_transform(la.image.enhance_brightness(self._image, 0.6))
                self._axis_transform_up()
                if isinstance(self._info, str):
                    self._info = self._temp_info.copy()
                    self._axis = self._temp_axis.copy()
                else:
                    for j in self._temp_info:
                        if '图片模糊' not in self._temp_info[j]:
                            self._info[j] = self._temp_info[j]
                    for j in self._temp_axis:
                        if j not in self._axis:
                            self._axis[j] = self._temp_axis[j]
                break
        
        self._error = 'ok'
        angle = 0 if self._angle==-1 else self._angle
        for i in self._info:
            if '图片模糊' in self._info[i]:
                self._error = self._info[i]
                break
        return {'data':self._info, 'axis':self._axis, 'angle':angle, 'error':self._error}
        
    def _direction_transform(self, image):
        if self._angle!=-1:
            image1 = la.image.rotate(image, self._angle, expand=True)
            image1 = la.image.image_to_array(image1)
            self._result = self.ocr.ocr(image1, cls=False)
        else:
            self._result = []
            for angle in [0, 90, 180, 270]:
                if angle>0:
                    image1 = la.image.rotate(image, angle, expand=True)
                    image1 = la.image.image_to_array(image1)
                else:
                    image1 = la.image.image_to_array(image)
                result = self.ocr.ocr(image1, cls=False)
                rank = [0,0,0,0,0]
                for r, i in enumerate(result[0], start=1):
                    if '注意事项' in i[1][0]:
                        rank[0] = r
                    elif '户别' in i[1][0] or '户主' in i[1][0]:
                        rank[1] = r
                    elif '户号' in i[1][0] or '住址' in i[1][0]:
                        rank[2] = r
                    elif '户口专用' in i[1][0]:
                        rank[3] = r
                    elif '承办人' in i[1][0] or '签章' in i[1][0]:
                        rank[4] = r
                rank = [i for i in rank if i>0]
                if rank==sorted(rank) and len(rank)>1:
                    self._result = result.copy()
                    self._angle = angle
                    self._mode = 'shouye'
                    break
        
        self._info = {}
        if self._mode == 'shouye':
            self._info['household_type'] = '图片模糊:未识别出户别'
            self._info['household_name'] = '图片模糊:未识别出户主'
            self._info['household_id'] = '图片模糊:未识别出户号'
            self._info['household_address'] = '图片模糊:未识别出住址'
        else:
            self._info = '图片模糊:未识别出有效信息'
            self._error = '图片模糊:未识别出有效信息'
    
    def _axis_transform_up(self):
        if self._mode == 'shouye':
            self._axis_transform_shouye()
    
    def _axis_transform_shouye(self):
        if len(self._result)==0:
            return 0
        fix_x = []
        axis_true = defaultdict(list)
        axis_dict = defaultdict(list)

        step = 0
        for i in self._result[0]:
            h = (i[0][3][1]+i[0][2][1]-i[0][1][1]-i[0][0][1])/2
            w = (i[0][1][0]+i[0][2][0]-i[0][0][0]-i[0][3][0])/2
            x = min(i[0][0][0], i[0][3][0])
            y = min(i[0][0][1], i[0][1][1])
            if 'household_type' not in axis_true:
                for char in self._char_household_type:
                    if char in i[1][0]:
                        if len(i[1][0][i[1][0].find(char)+len(char):])>1:
                            w = w/(len(i[1][0])+2)
                            axis_true['household_type'] = [x+w*3, y]+i[0][2]
                        else:
                            w = w/(len(i[1][0])+1.5)
                            axis_true['household_type'] = [x+w*3, y, x+w*10, y+h]
                        axis_dict['household_name'].append(([x+w*14, y, x+w*24, y+h], 0.8))
                        axis_dict['household_id'].append(([x+w*3, y+h, x+w*8, y+h*3], 0.8))
                        axis_dict['household_address'].append(([x+w*12, y+h, x+w*24, y+h*3], 0.6))
                        break
                if 'household_type' in axis_true:
                    continue
            if 'household_name' not in axis_true:
                for char in self._char_household_name:
                    if char in i[1][0]:
                        if len(i[1][0][i[1][0].find(char)+len(char):])>1:
                            w = w/(len(i[1][0])+2)
                            axis_true['household_name'] = [x+w*4, y]+i[0][2]
                        else:
                            w = w/(len(i[1][0]))
                            axis_true['household_name'] = [x+w*4, y, x+w*10, y+h]
                        axis_dict['household_type'].append(([x-w*8, y, x, y+h], 0.8))
                        axis_dict['household_id'].append(([x-w*8, y+h, x-w*2, y+h*3], 0.6))
                        axis_dict['household_address'].append(([x+w*2, y+h, x+w*16, y+h*3], 0.8))
                        break
                if 'household_name' in axis_true:
                    continue
            if 'household_id' not in axis_true:
                for char in self._char_household_id:
                    if char in i[1][0]:
                        if len(i[1][0][i[1][0].find(char)+len(char):])>1:
                            w = w/(len(i[1][0])+2)
                            axis_true['household_id'] = [x+w*3, y]+i[0][2]
                        else:
                            w = w/(len(i[1][0])+1.5)
                            axis_true['household_id'] = [x+w*3, y, x+w*9, y+h]
                        axis_dict['household_type'].append(([x+w*3, y-h*2, x+w*10, y], 0.8))
                        axis_dict['household_name'].append(([x+w*14, y+h*2, x+w*24, y], 0.6))
                        axis_dict['household_address'].append(([x+w*12, y, x+w*24, y+h], 0.8))
                        break
                if 'household_id' in axis_true:
                    continue
            if 'household_address' not in axis_true:
                for char in self._char_household_address:
                    if char in i[1][0]:
                        if len(i[1][0][i[1][0].find(char)+len(char):])>1:
                            w = w/(len(i[1][0])+2)
                            axis_true['household_address'] = [x+w*3, y]+i[0][2]
                        else:
                            w = w/(len(i[1][0])+1)
                            axis_true['household_address'] = [x+w*3, y, x+w*18, y+h]
                        axis_dict['household_type'].append(([x-w*6, y-h*2, x+w, y], 0.6))
                        axis_dict['household_name'].append(([x+w*5, y-h*2, x+w*14, y], 0.8))
                        axis_dict['household_id'].append(([x-w*6, y, x-w, y+h], 0.8))
                        break
                if 'household_address' in axis_true:
                    continue

        for i in ['household_type', 'household_name', 'household_id', 'household_address']:
            if i not in axis_true:
                if i in axis_dict:
                    weight = sum([j[1] for j in axis_dict[i]])
                    axis_true[i] = [sum([j[0][0]*j[1] for j in axis_dict[i]])/weight,
                                    sum([j[0][1]*j[1] for j in axis_dict[i]])/weight,
                                    sum([j[0][2]*j[1] for j in axis_dict[i]])/weight,
                                    sum([j[0][3]*j[1] for j in axis_dict[i]])/weight]

        if self._axis is None:
            self._axis = axis_true.copy()
        for i in axis_true:
            axis_true[i] = tuple(axis_true[i])
        for i in self._result[0]:
            h = (i[0][3][1]+i[0][2][1]-i[0][1][1]-i[0][0][1])/2
            w = (i[0][1][0]+i[0][2][0]-i[0][0][0]-i[0][3][0])/2
            x = min(i[0][0][0], i[0][3][0])
            y = min(i[0][0][1], i[0][1][1])
            if '图片模糊' in self._info['household_type'] and 'household_type' in axis_true:
                h1 = min(max(i[0][3][1], i[0][2][1]), axis_true['household_type'][3])-max(min(i[0][0][1], i[0][1][1]), axis_true['household_type'][1])
                w1 = min(max(i[0][1][0], i[0][2][0]), axis_true['household_type'][2])-max(min(i[0][0][0], i[0][3][0]), axis_true['household_type'][0])            
                if h1/h>0.6 and w1/w>0.6:
                    if len(i[1][0][i[1][0].find('别')+len('别'):])>1:
                        self._info['household_type'] = i[1][0][i[1][0].find('别')+1:]
                        self._axis['household_type'] = [self._axis['household_type'][0], y]+i[0][2]
                    elif len(i[1][0])>1 and sum([1 for j in '户别' if j in i[1][0]])==0:
                        self._info['household_type'] = i[1][0]
                        self._axis['household_type'] = [x, y]+i[0][2]
                        fix_x.append(i[0][0][0])
                if '图片模糊' not in self._info['household_type']:
                    continue
            if '图片模糊' in self._info['household_name'] and 'household_name' in axis_true:
                h1 = min(max(i[0][3][1], i[0][2][1]), axis_true['household_name'][3])-max(min(i[0][0][1], i[0][1][1]), axis_true['household_name'][1])
                w1 = min(max(i[0][1][0], i[0][2][0]), axis_true['household_name'][2])-max(min(i[0][0][0], i[0][3][0]), axis_true['household_name'][0])            
                if h1/h>0.6 and w1/w>0.6:
                    if len(i[1][0][i[1][0].find('名')+len('名'):])>1:
                        self._info['household_name'] = i[1][0][i[1][0].find('名')+1:]
                        self._axis['household_name'] = [self._axis['household_name'][0], y]+i[0][2]
                    elif len(i[1][0])>1:
                        self._info['household_name'] = i[1][0]
                        self._axis['household_name'] = [x, y]+i[0][2]
                if '图片模糊' not in self._info['household_name']:
                    continue
            if '图片模糊' in self._info['household_id'] and 'household_id' in axis_true:
                h1 = min(max(i[0][3][1], i[0][2][1]), axis_true['household_id'][3])-max(min(i[0][0][1], i[0][1][1]), axis_true['household_id'][1])
                w1 = min(max(i[0][1][0], i[0][2][0]), axis_true['household_id'][2])-max(min(i[0][0][0], i[0][3][0]), axis_true['household_id'][0])            
                if h1/h>0.6 and w1/w>0.6:
                    if len(i[1][0][i[1][0].find('号')+len('号'):])>1:
                        self._info['household_id'] = i[1][0][i[1][0].find('号')+1:]
                        self._axis['household_id'] = [self._axis['household_id'][0], y]+i[0][2]
                    elif len(i[1][0])>1:
                        self._info['household_id'] = i[1][0]
                        self._axis['household_id'] = [x, y]+i[0][2]
                        fix_x.append(i[0][0][0])
                if '图片模糊' not in self._info['household_id']:
                    continue
            if '图片模糊' in self._info['household_address'] and 'household_address' in axis_true:
                h1 = min(max(i[0][3][1], i[0][2][1]), axis_true['household_address'][3])-max(min(i[0][0][1], i[0][1][1]), axis_true['household_address'][1])
                w1 = min(max(i[0][1][0], i[0][2][0]), axis_true['household_address'][2])-max(min(i[0][0][0], i[0][3][0]), axis_true['household_address'][0])            
                if h1/h>0.6 and w1/w>0.6:
                    if len(i[1][0][i[1][0].find('址')+len('址'):])>1:
                        self._info['household_address'] = i[1][0][i[1][0].find('址')+1:]
                        self._axis['household_address'] = [self._axis['household_address'][0], y]+i[0][2]
                    elif len(i[1][0])>1:
                        self._info['household_address'] = i[1][0]
                        self._axis['household_address'] = [x, y]+i[0][2]
                if '图片模糊' not in self._info['household_address']:
                    continue

        try:
            if len(fix_x)>0:
                fix_x = sum(fix_x)/len(fix_x)
                self._axis['household_type'][0] = fix_x
                self._axis['household_id'][0] = fix_x
        except:
            pass
    
        for i in self._axis:
            self._axis[i] = [int(max(0, j)) for j in self._axis[i]]
    
    
    def draw_mask(self, image=None, axis=None, box_axis='all', mask_axis=None):
        if image is None:
            image = self._image.copy()
        angle = self._angle if axis is None else axis['angle']
        axis = self._axis if axis is None else axis['axis']

        if box_axis=='all':
            box_axis = self._keys
        elif isinstance(box_axis, str):
            if box_axis in self._keys:
                box_axis = [box_axis]
            else:
                raise ValueError(f'`box_axis` must be one of {self._keys}')
        elif isinstance(box_axis, list):
            for i in box_axis:
                if i not in self._keys:
                    raise ValueError(f'`{i}` not in {self._keys}')
        else:
            raise ValueError(f'`box_axis` must be one of {self._keys}')


        if mask_axis is None:
            mask_axis = []
        elif mask_axis=='all':
            mask_axis = self._keys
        elif isinstance(mask_axis, str):
            if mask_axis in self._keys:
                mask_axis = [mask_axis]
            else:
                raise ValueError(f'`box_axis` must be one of {self._keys}')
        elif isinstance(mask_axis, list):
            for i in mask_axis:
                if i not in self._keys:
                    raise ValueError(f'`{i}` not in {self._keys}')
        else:
            raise ValueError(f'`box_axis` must be one of {self._keys}')

        try:
            if angle>0:
                image = la.image.rotate(image, angle, expand=True)
            t = [la.image.box_convert(axis[i], 'xyxy', 'axis') for i in box_axis if i not in mask_axis and i in axis]
            if len(t)>0:
                image = la.image.draw_box(image, t, width=2)
            t = [la.image.box_convert(axis[i], 'xyxy', 'axis') for i in mask_axis and i in axis]
            if len(t)>0:
                image = la.image.draw_box(image, t, fill_color=(255,255,255), width=2)
        except:
            pass
        return image
    
    def env_check(self):
        env = la.utils.pip.freeze('paddleocr')['paddleocr']
        if env>='2.6.1.3':
            return 'Environment check ok.'
        else:
            return f"Now environment dependent paddleocr>='2.6.1.3', local env paddleocr='{env}'"


# img_path = '/home/app_user_5i5j/zhaohang/11/data/hukoubenfenlei/1680845488911.jpg'
# model = OCRHouseholdCard()
# model.predict(img_path)

