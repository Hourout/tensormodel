from collections import defaultdict

import cv2
import paddleocr
import linora as la

__all__ = ['OCRMarriageCard']


class OCRMarriageCard():
    def __init__(self, ocr=None):
        self.ocr = paddleocr.PaddleOCR(show_log=False) if ocr is None else ocr
        self._keys = ['marriage_name', 'marriage_date', 'marriage_id', 
                      'user_name_up', 'user_sex_up', 'user_country_up', 'user_born_up', 'user_number_up', 
                      'user_name_down', 'user_sex_down', 'user_country_down', 'user_born_down', 'user_number_down', 
                      'user_face', 'user_card']
        self._char_marriage_name = ['持证人']
        self._char_marriage_date = ['登记日期']
        self._char_marriage_id = ['结婚证字号', '离婚证字号']
        self._char_user_name = ['姓名']
        self._char_user_country = ['国籍', '国箱', '国馨', '国精']
        self._char_number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        
    def predict(self, image, axis=False, ocr_result=None):
        self._marriage_name_prob = 0
        self._axis = None
        self._show_axis = axis
        self._error = 'ok'
        self._angle = -1
        
        if ocr_result is not None:
            self._result = ocr_result
            self._direction_transform(image, use_ocr_result=True)
            self._axis_transform_up()
        else:
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
                if self._show_axis:
                    return {'data':self._info, 'axis':[], 'angle':0, 'error':self._error}
                else:
                    return {'data':self._info, 'angle':0, 'error':self._error}
            self._axis_transform_up()
            for i in self._info:
                if '图片模糊' in self._info[i]:
                    self._temp_info = self._info.copy()
                    if self._show_axis:
                        self._temp_axis = self._axis.copy()
                    self._direction_transform(la.image.enhance_brightness(self._image, 0.6))
                    self._axis_transform_up()
                    if isinstance(self._info, str):
                        self._info = self._temp_info.copy()
                        if self._show_axis:
                            self._axis = self._temp_axis.copy()
                    else:
                        for j in self._temp_info:
                            if '图片模糊' not in self._temp_info[j]:
                                self._info[j] = self._temp_info[j]
                        if self._show_axis:
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
        if self._show_axis:
            return {'data':self._info, 'axis':self._axis, 'angle':angle, 'error':self._error}
        else:
            return {'data':self._info, 'angle':angle, 'error':self._error}
        
    def _direction_transform(self, image, use_ocr_result=False):
        if use_ocr_result:
            self._angle = 0
        elif self._angle!=-1:
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
                    if '持证人' in i[1][0]:
                        rank[0] = r
                    elif '登记日期' in i[1][0]:
                        rank[1] = r
                    elif '婚证字号'in i[1][0]:
                        rank[2] = r
                    elif '备注' in i[1][0]:
                        rank[3] = r
                    elif '身份证件号' in i[1][0]:
                        rank[4] = r
                rank = [i for i in rank if i>0]
                if rank==sorted(rank) and len(rank)>1:
                    self._result = result.copy()
                    self._angle = angle
                    break
        
        self._info = {}
        if self._angle!=-1:
            self._info['marriage_name'] = '图片模糊:未识别出持证人'
            self._info['marriage_date'] = '图片模糊:未识别出登记日期'
            self._info['marriage_id'] = '图片模糊:未识别出结婚证字号'
            self._info['user_name_up'] = '图片模糊:未识别出本人姓名'
            self._info['user_sex_up'] = '图片模糊:未识别出本人性别'
            self._info['user_country_up'] = '图片模糊:未识别出本人国籍'
            self._info['user_born_up'] = '图片模糊:未识别出本人出生日期'
            self._info['user_number_up'] = '图片模糊:未识别出本人身份证号码'
            self._info['user_name_down'] = '图片模糊:未识别出配偶姓名'
            self._info['user_sex_down'] = '图片模糊:未识别出配偶性别'
            self._info['user_country_down'] = '图片模糊:未识别出配偶国籍'
            self._info['user_born_down'] = '图片模糊:未识别出配偶出生日期'
            self._info['user_number_down'] = '图片模糊:未识别出配偶身份证号码'
        else:
            self._info = '图片模糊:未识别出有效信息'
            self._error = '图片模糊:未识别出有效信息'
    
    def _axis_transform_up(self):
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
            if 'marriage_name' not in axis_true:
                for char in self._char_marriage_name:
                    if char in i[1][0]:
                        if len(i[1][0][i[1][0].find(char)+len(char):])>1:
                            w = w/(len(i[1][0])+2)
                            axis_true['marriage_name'] = [x+w*3, y]+i[0][2]
                        else:
                            w = w/(len(i[1][0])+2)
                            axis_true['marriage_name'] = [x, y-h*0.5, x+w*12, y+h*3.5]
                        axis_dict['marriage_date'].append(([x, y+h*3, x+w*14, y+h*6.5], 0.8))
                        axis_dict['marriage_id'].append(([x, y+h*7, x+w*16, y+h*9.5], 0.6))
                        break
                if 'marriage_name' in axis_true:
                    continue
            if 'marriage_date' not in axis_true:
                for char in self._char_marriage_date:
                    if char in i[1][0]:
                        if len(i[1][0][i[1][0].find(char)+len(char):])>1:
                            self._info['marriage_date'] = (i[1][0][i[1][0].find(char)+len(char):]).strip()
                            w = w/12
                            axis_true['marriage_date'] = [x+w*5, y]+i[0][2]
                        else:
                            w = w/4
                            axis_true['marriage_date'] = [x, y-h*0.5, x+w*14, y+h*3.5]
                        axis_dict['marriage_name'].append(([x, y-h*3.5, x+w*12, y-h*0.5], 0.8))
                        axis_dict['marriage_id'].append(([x, y+h*4, x+w*16, y+h*6.5], 0.8))
                        break
                if 'marriage_date' in axis_true:
                    continue
            if 'marriage_id' not in axis_true:
                for char in self._char_marriage_id:
                    if char in i[1][0]:
                        if len(i[1][0][i[1][0].find(char)+len(char):])>1:
                            self._info['marriage_id'] = (i[1][0][i[1][0].find(char)+len(char):]).strip()
                            w = w/12
                            axis_true['marriage_id'] = [x+w*5, y]+i[0][2]
                        else:
                            w = w/4
                            axis_true['marriage_id'] = [x, y-h*0.5, x+w*16, y+h*3]
                        axis_dict['marriage_name'].append(([x, y-h*8, x+w*12, y-h*4.5], 0.6))
                        axis_dict['marriage_date'].append(([x, y-h*3.5, x+w*14, y-h*0.5], 0.8))
                        break
                if 'marriage_id' in axis_true:
                    continue
            if 'user_name_up' not in axis_true:
                for char in self._char_user_name:
                    if char in i[1][0]:
                        if len(i[1][0])>3:
                            axis_true['user_name_up'] = [x+w*0.2, y-h]+i[0][2]
                        elif len(i[1][0])>2:
                            axis_true['user_name_up'] = [x+w*0.5, y-h, x+w*2.5, y+h*2]
                        else:
                            axis_true['user_name_up'] = [x+w*1.2, y-h, x+w*4.5, y+h*2]
                        break
                if 'user_name_up' in axis_true:
                    continue
            if 'user_country_up' not in axis_true:
                for char in self._char_user_country:
                    if char in i[1][0]:
                        if len(i[1][0][i[1][0].find(char)+len(char):])>1:
                            axis_true['user_country_up'] = [x+w*0.2, y]+i[0][2]
                        else:
                            axis_true['user_country_up'] = [x+w*1.5, y-h, x+w*4.5, y+h*2]
                        break
                if 'user_country_up' in axis_true:
                    continue
            if 'user_number_up' not in axis_true and step==0:
                if sum([1 for j in i[1][0][-18:] if j in '0123456789xX'])==18:
                    axis_true['user_number_up'] = [x, y, max(i[0][1][0], i[0][2][0]), max(i[0][2][1], i[0][3][1])]
                    if len(i[1][0][:-18]):
                        axis_true['user_number_up'][0] = axis_true['user_number_up'][0]+w*len(i[1][0][:-18])/(len(i[1][0])-18+9+1.5)
                        w = (i[0][1][0]+i[0][2][0])/2-axis_true['user_number_up'][0]
                    if 'user_name_up' not in axis_true:
                        axis_true['user_name_up'] = [axis_true['user_number_up'][0]-w*0.33, axis_true['user_number_up'][1]-h*3.5, 
                                                     axis_true['user_number_up'][0]+w*5.5/13, axis_true['user_number_up'][1]-h*2.5]
                    if 'user_country_up' not in axis_true:
                        axis_true['user_country_up'] = [axis_true['user_number_up'][0]-w*0.33, axis_true['user_number_up'][1]-h*2.25, 
                                                        axis_true['user_number_up'][0]+w*3.5/13, axis_true['user_number_up'][1]-h*0.5]
                    axis_true['user_sex_up'] = [axis_true['user_number_up'][0]+w*1.2, axis_true['user_number_up'][1]-h*3, 
                                                axis_true['user_number_up'][0]+w*1.66, axis_true['user_number_up'][1]-h*1.25]
                    axis_true['user_born_up'] = [axis_true['user_number_up'][0]+w*1.5, axis_true['user_number_up'][1]-h*1.75, 
                                                 axis_true['user_number_up'][0]+w*2.3, axis_true['user_number_up'][1]-h*0.1]
                if 'user_number_up' in axis_true:
                    continue
            if 'user_name_down' not in axis_true:
                for char in self._char_user_name:
                    if char in i[1][0]:
                        if len(i[1][0])>3:
                            axis_true['user_name_down'] = [x+w*0.2, y-h]+i[0][2]
                        elif len(i[1][0])>2:
                            axis_true['user_name_down'] = [x+w*0.5, y-h, x+w*2.5, y+h*2]
                        else:
                            axis_true['user_name_down'] = [x+w*1.2, y-h, x+w*4.5, y+h*2]
                        break
                if 'user_name_down' in axis_true:
                    step = 1
                    continue
            if 'user_country_down' not in axis_true:
                for char in self._char_user_country:
                    if char in i[1][0]:
                        if len(i[1][0][i[1][0].find(char)+len(char):])>1:
                            axis_true['user_country_down'] = [x+w*0.2, y]+i[0][2]
                        else:
                            axis_true['user_country_down'] = [x+w*1.5, y-h, x+w*4.5, y+h*2]
                        break
                if 'user_country_down' in axis_true:
                    continue
            if 'user_number_down' not in axis_true:
                if sum([1 for j in i[1][0][-18:] if j in '0123456789xX'])==18:
                    axis_true['user_number_down'] = [x, y, max(i[0][1][0], i[0][2][0]), max(i[0][2][1], i[0][3][1])]
                    if len(i[1][0][:-18]):
                        axis_true['user_number_down'][0] = axis_true['user_number_down'][0]+w*len(i[1][0][:-18])/(len(i[1][0])-18+9+1.5)
                        w = (i[0][1][0]+i[0][2][0])/2-axis_true['user_number_down'][0]
                    if 'user_name_down' not in axis_true:
                        axis_true['user_name_down'] = [axis_true['user_number_down'][0]-w*0.33, axis_true['user_number_down'][1]-h*3.5, 
                                                       axis_true['user_number_down'][0]+w*5.5/13, axis_true['user_number_down'][1]-h*2.5]
                    if 'user_country_down' not in axis_true:
                        axis_true['user_country_down'] = [axis_true['user_number_down'][0]-w*0.33, axis_true['user_number_down'][1]-h*2.25, 
                                                          axis_true['user_number_down'][0]+w*3.5/13, axis_true['user_number_down'][1]-h*0.5]
                    axis_true['user_sex_down'] = [axis_true['user_number_down'][0]+w*1.2, axis_true['user_number_down'][1]-h*3, 
                                                  axis_true['user_number_down'][0]+w*1.65, axis_true['user_number_down'][1]-h*1.25]
                    axis_true['user_born_down'] = [axis_true['user_number_down'][0]+w*1.5, axis_true['user_number_down'][1]-h*1.75, 
                                                   axis_true['user_number_down'][0]+w*2.3, axis_true['user_number_down'][1]-h*0.1]


        for i in ['marriage_name', 'marriage_date', 'marriage_id', 'user_number_up', 'user_number_down']:
            if i not in axis_true:
                if i in axis_dict:
                    weight = sum([j[1] for j in axis_dict[i]])
                    axis_true[i] = [sum([j[0][0]*j[1] for j in axis_dict[i]])/weight,
                                    sum([j[0][1]*j[1] for j in axis_dict[i]])/weight,
                                    sum([j[0][2]*j[1] for j in axis_dict[i]])/weight,
                                    sum([j[0][3]*j[1] for j in axis_dict[i]])/weight]

        self._axis = axis_true.copy()
        for i in axis_true:
            axis_true[i] = tuple(axis_true[i])
        step = 0
        step_name = 0
        for i in self._result[0]:
            for j in ['国籍', '出生日期', '身份证件号']:
                if j in i[1][0]:
                    step_name = 1
            h = (i[0][3][1]+i[0][2][1]-i[0][1][1]-i[0][0][1])/2
            w = (i[0][1][0]+i[0][2][0]-i[0][0][0]-i[0][3][0])/2
            if h==0:
                h = 1
            if w==0:
                w = 1
            x = min(i[0][0][0], i[0][3][0])
            y = min(i[0][0][1], i[0][1][1])
            if '图片模糊' in self._info['marriage_name'] and 'marriage_name' in axis_true:
                h1 = min(max(i[0][3][1], i[0][2][1]), axis_true['marriage_name'][3])-max(min(i[0][0][1], i[0][1][1]), axis_true['marriage_name'][1])
                w1 = min(max(i[0][1][0], i[0][2][0]), axis_true['marriage_name'][2])-max(min(i[0][0][0], i[0][3][0]), axis_true['marriage_name'][0])            
                if h1/h>0.6 and w1/w>0.6:
                    if len(i[1][0])>4 and '人' in i[1][0]:
                        self._info['marriage_name'] = i[1][0][i[1][0].find('人')+1:]
                        self._axis['marriage_name'] = [self._axis['marriage_name'][0], y]+i[0][2]
                        self._marriage_name_prob = i[1][1]
                    elif len(i[1][0])>1 and sum([1 for j in '持证人' if j in i[1][0]])<2:
                        self._info['marriage_name'] = i[1][0]
                        self._axis['marriage_name'] = [x, y]+i[0][2]
                        fix_x.append(i[0][0][0])
                        self._marriage_name_prob = i[1][1]
                if '图片模糊' not in self._info['marriage_name']:
                    continue
            if '图片模糊' in self._info['marriage_date'] and 'marriage_date' in axis_true:
                h1 = min(max(i[0][3][1], i[0][2][1]), axis_true['marriage_date'][3])-max(min(i[0][0][1], i[0][1][1]), axis_true['marriage_date'][1])
                w1 = min(max(i[0][1][0], i[0][2][0]), axis_true['marriage_date'][2])-max(min(i[0][0][0], i[0][3][0]), axis_true['marriage_date'][0])            
                if h1/h>0.6 and w1/w>0.6:
                    if len(i[1][0])>10 and '期' in i[1][0]:
                        if sum([1 for j in i[1][0][i[1][0].find('期')+1:].replace('年','').replace('月','').replace('日','') if j not in self._char_number])==0:
                            self._info['marriage_date'] = i[1][0][i[1][0].find('期')+1:]
                            self._axis['marriage_date'] = [self._axis['marriage_date'][0], y]+i[0][2]
                    elif len(i[1][0]) in [9, 10, 11] and i[1][0].find('年')==4 and '月' in i[1][0] and i[1][0].endswith('日'):
                        if sum([1 for j in i[1][0].replace('年','').replace('月','').replace('日','') if j not in self._char_number])==0:
                            self._info['marriage_date'] = i[1][0]
                            self._axis['marriage_date'] = [x, y]+i[0][2]
                            fix_x.append(i[0][0][0])
                    elif len(i[1][0])==10 and i[1][0].find('-')==4 and i[1][0][-3]=='-':
                        self._info['marriage_date'] = f'{i[1][0][:4]}年{i[1][0][5:7]}月{i[1][0][-2:]}日'
                        self._axis['marriage_date'] = [x, y]+i[0][2]
                        fix_x.append(i[0][0][0])
                if '图片模糊' not in self._info['marriage_date']:
                    continue
            if '图片模糊' in self._info['marriage_id'] and 'marriage_id' in axis_true:
                h1 = min(max(i[0][3][1], i[0][2][1]), axis_true['marriage_id'][3])-max(min(i[0][0][1], i[0][1][1]), axis_true['marriage_id'][1])
                w1 = min(max(i[0][1][0], i[0][2][0]), axis_true['marriage_id'][2])-max(min(i[0][0][0], i[0][3][0]), axis_true['marriage_id'][0])            
                if h1/h>0.6 and w1/w>0.6:
                    if len(i[1][0])>15 and '号' in i[1][0]:
                        self._info['marriage_id'] = i[1][0][i[1][0].find('号')+1:]
                        self._axis['marriage_id'] = [self._axis['marriage_id'][0], y]+i[0][2]
                    elif len(i[1][0])>9:
                        self._info['marriage_id'] = i[1][0]
                        self._axis['marriage_id'] = [x, y]+i[0][2]
                        fix_x.append(i[0][0][0])
                if '图片模糊' not in self._info['marriage_id']:
                    continue
            if '图片模糊' in self._info['user_name_up'] and 'user_name_up' in axis_true and step_name==0:
                for char in self._char_user_name:
                    if char in i[1][0]:
                        if len(i[1][0][i[1][0].find(char)+len(char):])>1:
                            if i[1][1] >self._marriage_name_prob:
                                self._info['user_name_up'] = (i[1][0][i[1][0].find(char)+len(char):]).strip()
                                self._info['marriage_name'] = self._info['user_name_up']
                            else:
                                self._info['user_name_up'] = self._info['marriage_name']
                            self._axis['user_name_up'] = [self._axis['user_name_up'][0], y]+i[0][2]
                            break
                if '图片模糊' not in self._info['user_name_up']:
                    continue
                h1 = min(max(i[0][3][1], i[0][2][1]), axis_true['user_name_up'][3])-max(min(i[0][0][1], i[0][1][1]), axis_true['user_name_up'][1])
                w1 = min(max(i[0][1][0], i[0][2][0]), axis_true['user_name_up'][2])-max(min(i[0][0][0], i[0][3][0]), axis_true['user_name_up'][0])            
                if h1/h>0.6 and w1/w>0.6:
                    if len(i[1][0])>3 and i[1][0].find('名')==1:
                        if i[1][1] >self._marriage_name_prob:
                            self._info['user_name_up'] = i[1][0][2:]
                            self._info['marriage_name'] = self._info['user_name_up']
                        else:
                            self._info['user_name_up'] = self._info['marriage_name']
                        self._axis['user_name_up'] = [self._axis['user_name_up'][0], y]+i[0][2]
                    elif i[1][0].startswith('名') and len(i[1][0])>2:
                        if i[1][1] >self._marriage_name_prob:
                            self._info['user_name_up'] = i[1][0][1:]
                            self._info['marriage_name'] = self._info['user_name_up']
                        else:
                            self._info['user_name_up'] = self._info['marriage_name']
                        self._axis['user_name_up'] = [self._axis['user_name_up'][0], y]+i[0][2]
                    elif len(i[1][0])>1:
                        if i[1][1] >self._marriage_name_prob:
                            self._info['user_name_up'] = i[1][0]
                            self._info['marriage_name'] = self._info['user_name_up']
                        else:
                            self._info['user_name_up'] = self._info['marriage_name']
                        self._axis['user_name_up'] = [self._axis['user_name_up'][0], y]+i[0][2]
                if '图片模糊' not in self._info['user_name_up']:
                    continue
            if '图片模糊' in self._info['user_country_up'] and 'user_country_up' in axis_true:
                for char in self._char_user_country:
                    if char in i[1][0]:
                        if 10>len(i[1][0][i[1][0].find(char)+len(char):])>1:
                            self._info['user_country_up'] = (i[1][0][i[1][0].find(char)+len(char):]).strip()
                            self._axis['user_country_up'] = [self._axis['user_country_up'][0], y]+i[0][2]
                            break
                if '图片模糊' not in self._info['user_country_up']:
                    continue
                h1 = min(max(i[0][3][1], i[0][2][1]), axis_true['user_country_up'][3])-max(min(i[0][0][1], i[0][1][1]), axis_true['user_country_up'][1])
                w1 = min(max(i[0][1][0], i[0][2][0]), axis_true['user_country_up'][2])-max(min(i[0][0][0], i[0][3][0]), axis_true['user_country_up'][0])            
                if h1/h>0.6 and w1/w>0.6:
                    if 12>len(i[1][0])>3 and (i[1][0].find('籍')==1 or i[1][0].find('箱')==1):
                        self._info['user_country_up'] = i[1][0][2:]
                        self._axis['user_country_up'] = [self._axis['user_country_up'][0], y]+i[0][2]
                    elif (i[1][0].startswith('籍') or i[1][0].startswith('箱')) and 11>len(i[1][0])>2:
                        self._info['user_country_up'] = i[1][0][1:]
                        self._axis['user_country_up'] = [self._axis['user_country_up'][0], y]+i[0][2]
                    elif 10>len(i[1][0])>1:
                        self._info['user_country_up'] = i[1][0]
                        self._axis['user_country_up'] = [self._axis['user_country_up'][0], y]+i[0][2]
                if '图片模糊' not in self._info['user_country_up']:
                    continue
            if '图片模糊' in self._info['user_number_up'] and step==0:
                if sum([1 for j in i[1][0][-18:] if j in '0123456789xX'])==18:
                    self._info['user_number_up'] = i[1][0][-18:]
                    self._info['user_sex_up'] =  '男' if int(i[1][0][-18:][16])%2 else '女'
                    self._info['user_born_up'] = f"{i[1][0][-18:][6:10]}年{i[1][0][-18:][10:12]}月{i[1][0][-18:][12:14]}日"
                elif sum([1 for j in i[1][0][:18] if j in '0123456789xX'])==18:
                    self._info['user_number_up'] = i[1][0][:18]
                    self._info['user_sex_up'] =  '男' if int(i[1][0][:18][16])%2 else '女'
                    self._info['user_born_up'] = f"{i[1][0][:18][6:10]}年{i[1][0][:18][10:12]}月{i[1][0][:18][12:14]}日"
                if '图片模糊' not in self._info['user_number_up']:
                    continue
            if '图片模糊' in self._info['user_name_down'] and 'user_name_down' in axis_true:
                for char in self._char_user_name:
                    if char in i[1][0]:
                        if len(i[1][0][i[1][0].find(char)+len(char):])>1:
                            self._info['user_name_down'] = (i[1][0][i[1][0].find(char)+len(char):]).strip()
                            self._axis['user_name_down'] = [self._axis['user_name_down'][0], y]+i[0][2]
                            break
                if '图片模糊' not in self._info['user_name_down']:
                    step = 1
                    continue
                h1 = min(max(i[0][3][1], i[0][2][1]), axis_true['user_name_down'][3])-max(min(i[0][0][1], i[0][1][1]), axis_true['user_name_down'][1])
                w1 = min(max(i[0][1][0], i[0][2][0]), axis_true['user_name_down'][2])-max(min(i[0][0][0], i[0][3][0]), axis_true['user_name_down'][0])            
                if h1/h>0.6 and w1/w>0.6:
                    if len(i[1][0])>3 and i[1][0].find('名')==1:
                        self._info['user_name_down'] = i[1][0][2:]
                        self._axis['user_name_down'] = [self._axis['user_name_down'][0], y]+i[0][2]
                    elif i[1][0].startswith('名') and len(i[1][0])>2:
                        self._info['user_name_down'] = i[1][0][1:]
                        self._axis['user_name_down'] = [self._axis['user_name_down'][0], y]+i[0][2]
                    elif len(i[1][0])>1 and sum([1 for char in self._char_user_name if i[1][0].startswith(char)])==0:
                        self._info['user_name_down'] = i[1][0]
                        self._axis['user_name_down'] = [self._axis['user_name_down'][0], y]+i[0][2]
                if '图片模糊' not in self._info['user_name_down']:
                    step = 1
                    continue
            if '图片模糊' in self._info['user_country_down'] and 'user_country_down' in axis_true:
                for char in self._char_user_country:
                    if char in i[1][0]:
                        if 10>len(i[1][0][i[1][0].find(char)+len(char):])>1:
                            self._info['user_country_down'] = (i[1][0][i[1][0].find(char)+len(char):]).strip()
                            self._axis['user_country_down'] = [self._axis['user_country_down'][0], y]+i[0][2]
                            break
                if '图片模糊' not in self._info['user_country_down']:
                    continue
                h1 = min(max(i[0][3][1], i[0][2][1]), axis_true['user_country_down'][3])-max(min(i[0][0][1], i[0][1][1]), axis_true['user_country_down'][1])
                w1 = min(max(i[0][1][0], i[0][2][0]), axis_true['user_country_down'][2])-max(min(i[0][0][0], i[0][3][0]), axis_true['user_country_down'][0])            
                if h1/h>0.6 and w1/w>0.6:
                    if 12>len(i[1][0])>3 and (i[1][0].find('籍')==1 or i[1][0].find('箱')==1):
                        self._info['user_country_down'] = i[1][0][2:]
                        self._axis['user_country_down'] = [self._axis['user_country_down'][0], y]+i[0][2]
                    elif (i[1][0].startswith('籍') or i[1][0].startswith('箱')) and 11>len(i[1][0])>2:
                        self._info['user_country_down'] = i[1][0][1:]
                        self._axis['user_country_down'] = [self._axis['user_country_down'][0], y]+i[0][2]
                    elif 10>len(i[1][0])>1:
                        self._info['user_country_down'] = i[1][0]
                        self._axis['user_country_down'] = [self._axis['user_country_down'][0], y]+i[0][2]
                if '图片模糊' not in self._info['user_country_down']:
                    continue
            if '图片模糊' in self._info['user_number_down']:
                if sum([1 for j in i[1][0][-18:] if j in '0123456789xX'])==18:
                    self._info['user_number_down'] = i[1][0][-18:]
                    self._info['user_sex_down'] =  '男' if int(i[1][0][-18:][16])%2 else '女'
                    self._info['user_born_down'] = f"{i[1][0][-18:][6:10]}年{i[1][0][-18:][10:12]}月{i[1][0][-18:][12:14]}日"
                elif sum([1 for j in i[1][0][:18] if j in '0123456789xX'])==18:
                    self._info['user_number_down'] = i[1][0][:18]
                    self._info['user_sex_down'] =  '男' if int(i[1][0][:18][16])%2 else '女'
                    self._info['user_born_down'] = f"{i[1][0][:18][6:10]}年{i[1][0][:18][10:12]}月{i[1][0][:18][12:14]}日"

        
        if self._info['marriage_id'][0]=='1':
            self._info['marriage_id'] = 'J'+self._info['marriage_id'][1:]
        if '图片模糊' in self._info['user_name_up'] and '图片模糊' not in self._info['marriage_name']:
            self._info['user_name_up'] = self._info['marriage_name']
        if '图片模糊' in self._info['user_country_up']:
            self._info['user_country_up'] = '中国'
        if '图片模糊' in self._info['user_country_down']:
            self._info['user_country_down'] = '中国'
        if '图片模糊' in self._info['user_sex_down'] and '图片模糊' in self._info['user_sex_up']:
            sex = [j for i in self._result[0] for j in ['男', '女'] if j in i[1][0]]
            self._info['user_sex_down'] = '男' if len(sex)<2 else sex[-1]
            self._info['user_sex_up'] = '女' if len(sex)<2 else sex[-2]
        elif '图片模糊' in self._info['user_sex_down'] and '图片模糊' not in self._info['user_sex_up']:
            self._info['user_sex_down'] = '女' if self._info['user_sex_up']=='男' else '男'
        elif '图片模糊' not in self._info['user_sex_down'] and '图片模糊' in self._info['user_sex_up']:
            self._info['user_sex_up'] = '女' if self._info['user_sex_down']=='男' else '男'
        if '图片模糊' in self._info['user_born_down'] or '图片模糊' in self._info['user_born_up']:
            date = [i[1][0][-11:] for i in self._result[0] if i[1][0][-11:].find('年')==4 and '月' in i[1][0] and i[1][0].endswith('日')][-2:]
            if len(date)==1:
                if '图片模糊' in self._info['user_born_down'] and '图片模糊' in self._info['user_born_up']:
                    self._info['user_born_up'] = date[0]
                    self._info['user_born_down'] = date[0]
                elif '图片模糊' in self._info['user_born_down'] or '图片模糊' not in self._info['user_born_up']:
                    self._info['user_born_down'] = date[0]
                elif '图片模糊' not in self._info['user_born_down'] or '图片模糊' in self._info['user_born_up']:
                    self._info['user_born_up'] = date[0]
            elif len(date)==2:
                if '图片模糊' in self._info['user_born_down'] and '图片模糊' in self._info['user_born_up']:
                    self._info['user_born_up'] = date[0]
                    self._info['user_born_down'] = date[1]
                elif '图片模糊' in self._info['user_born_down'] or '图片模糊' not in self._info['user_born_up']:
                    self._info['user_born_down'] = date[1]
                elif '图片模糊' not in self._info['user_born_down'] or '图片模糊' in self._info['user_born_up']:
                    self._info['user_born_up'] = date[0]
        
        if self._show_axis:
            try:
                if len(fix_x)>0:
                    fix_x = sum(fix_x)/len(fix_x)
                    self._axis['marriage_name'][0] = fix_x
                    self._axis['marriage_date'][0] = fix_x
                    self._axis['marriage_id'][0] = fix_x

                self._axis['user_face'] = [0,0,0,0]
                self._axis['user_face'][0] = self._axis['marriage_id'][0]+(self._axis['marriage_id'][2]-self._axis['marriage_id'][0])*0.9
                self._axis['user_face'][1] = self._axis['marriage_name'][1]-(self._axis['marriage_name'][3]-self._axis['marriage_name'][1])*0.25
                self._axis['user_face'][2] = self._axis['marriage_id'][0]+(self._axis['marriage_id'][2]-self._axis['marriage_id'][0])*2.1
                self._axis['user_face'][3] = self._axis['marriage_id'][1]-(self._axis['marriage_id'][3]-self._axis['marriage_id'][1])*0.25
                for i in self._result[0]:
                    if '持证人' in i[1][0]:
                        self._axis['user_face'][0] = i[0][0][0]+(self._axis['marriage_id'][2]-self._axis['marriage_id'][0])*1.3
                        self._axis['user_face'][1] = i[0][0][1]-(self._axis['marriage_id'][3]-self._axis['marriage_id'][1])*2.5
                        self._axis['user_face'][2] = i[0][0][0]+(self._axis['marriage_id'][2]-self._axis['marriage_id'][0])*2.8
                        self._axis['user_face'][3] = i[0][0][1]+(self._axis['marriage_id'][3]-self._axis['marriage_id'][1])*7
                        break
                    elif '登记日期' in i[1][0]:
                        self._axis['user_face'][0] = i[0][0][0]+(self._axis['marriage_id'][2]-self._axis['marriage_id'][0])*1.3
                        self._axis['user_face'][1] = i[0][0][1]-(self._axis['marriage_id'][3]-self._axis['marriage_id'][1])*6
                        self._axis['user_face'][2] = i[0][0][0]+(self._axis['marriage_id'][2]-self._axis['marriage_id'][0])*2.8
                        self._axis['user_face'][3] = i[0][0][1]+(self._axis['marriage_id'][3]-self._axis['marriage_id'][1])*2.5
                        break
                for i in self._result[0]:
                    h = (i[0][3][1]+i[0][2][1]-i[0][1][1]-i[0][0][1])/2
                    w = (i[0][1][0]+i[0][2][0]-i[0][0][0]-i[0][3][0])/2
                    x = min(i[0][0][0], i[0][3][0])
                    y = min(i[0][0][1], i[0][1][1])
                    if self._info['user_sex_up'] in i[1][0]:
                        h1 = min(max(i[0][3][1], i[0][2][1]), axis_true['user_sex_up'][3])-max(min(i[0][0][1], i[0][1][1]), axis_true['user_sex_up'][1])
                        w1 = min(max(i[0][1][0], i[0][2][0]), axis_true['user_sex_up'][2])-max(min(i[0][0][0], i[0][3][0]), axis_true['user_sex_up'][0])            
                        if h1/h>0.2 and w1/w>0.2:
                            self._axis['user_sex_up'] = [self._axis['user_sex_up'][0], y]+i[0][2]
                            continue
                    if self._info['user_born_up'] in i[1][0]:
                        self._axis['user_born_up'] = [self._axis['user_born_up'][0], y]+i[0][2]
                        continue
                    if self._info['user_sex_down'] in i[1][0]:
                        h1 = min(max(i[0][3][1], i[0][2][1]), axis_true['user_sex_down'][3])-max(min(i[0][0][1], i[0][1][1]), axis_true['user_sex_down'][1])
                        w1 = min(max(i[0][1][0], i[0][2][0]), axis_true['user_sex_down'][2])-max(min(i[0][0][0], i[0][3][0]), axis_true['user_sex_down'][0])            
                        if h1/h>0.2 and w1/w>0.2:
                            self._axis['user_sex_down'] = [self._axis['user_sex_down'][0], y]+i[0][2]
                            continue
                    if self._info['user_born_down'] in i[1][0]:
                        self._axis['user_born_down'] = [self._axis['user_born_down'][0], y]+i[0][2]
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

    def metrics(self, image_list):
        name = 0
        date = 0
        ids = 0
        name_up = 0
        sex_up = 0
        country_up = 0
        born_up = 0
        number_up = 0
        name_down = 0
        sex_down = 0
        country_down = 0
        born_down = 0
        number_down = 0

        marriage_name = 0
        marriage_date = 0
        marriage_id = 0
        user_name_up = 0
        user_sex_up = 0
        user_country_up = 0
        user_born_up = 0
        user_number_up = 0
        user_name_down = 0
        user_sex_down = 0
        user_country_down = 0
        user_born_down = 0
        user_number_down = 0

        for i in image_list:
            label = i.split('$$')[1:-1]
            t = self.predict(i)['data']
            if isinstance(t, dict):
                if t['marriage_name']==label[0]:
                    name += 1
                if t['marriage_date']==label[1]:
                    date += 1
                if t['marriage_id']==label[2]:
                    ids += 1
                if t['user_name_up']==label[3]:
                    name_up += 1
                if t['user_sex_up']==label[4]:
                    sex_up += 1
                if t['user_country_up']==label[5]:
                    country_up += 1
                if t['user_born_up']==label[6]:
                    born_up += 1
                if t['user_number_up']==label[7]:
                    number_up += 1
                if t['user_name_down']==label[8]:
                    name_down += 1
                if t['user_sex_down']==label[9]:
                    sex_down += 1
                if t['user_country_down']==label[10]:
                    country_down += 1
                if t['user_born_down']==label[11]:
                    born_down += 1
                if t['user_number_down']==label[12]:
                    number_down += 1

            marriage_name += 1
            marriage_date += 1
            marriage_id += 1
            user_name_up += 1
            user_sex_up += 1
            user_country_up += 1
            user_born_up += 1
            user_number_up += 1
            user_name_down += 1
            user_sex_down += 1
            user_country_down += 1
            user_born_down += 1
            user_number_down += 1

        ok = name+date+ids+name_up+sex_up+country_up+born_up+number_up+name_down+sex_down+country_down+born_down+number_down
        total = marriage_name+marriage_date+marriage_id+user_name_up+user_sex_up+user_country_up+user_born_up+user_number_up+user_name_down+user_sex_down+user_country_down+user_born_down+user_number_down
        result = {'marriage_name_acc':name/marriage_name, 'marriage_date_acc':date/marriage_date, 
                  'marriage_id_acc':ids/marriage_id, 'user_name_up_acc':name_up/user_name_up, 
                  'user_sex_up_acc':sex_up/user_sex_up, 'user_country_up_acc':country_up/user_country_up, 
                  'user_born_up_acc':born_up/user_born_up, 'user_number_up_acc':number_up/user_number_up, 
                  'user_name_down_acc':name_down/user_name_down, 
                  'user_sex_down_acc':sex_down/user_sex_down, 'user_country_down_acc':country_down/user_country_down, 
                  'user_born_down_acc':born_down/user_born_down, 'user_number_down_acc':number_down/user_number_down, 
                  'totalmean_acc':ok/total}
        return {i:round(result[i], 4) for i in result}

