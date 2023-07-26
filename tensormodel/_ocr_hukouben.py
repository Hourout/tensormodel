from collections import defaultdict

import cv2
import paddleocr
import linora as la

__all__ = ['OCRHuKouBen']


class OCRHuKouBen():
    def __init__(self, ocr=None):
        self.ocr = paddleocr.PaddleOCR(show_log=False) if ocr is None else ocr
        self._keys = []
        self._char_household_type = ['户别']
        self._char_household_name = ['户主姓名', '户生姓名']
        self._char_household_id = ['户号']
        self._char_household_address = ['住址']
        self._char_register_name = ['姓名']
#         self._char_number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        
    def predict(self, image, axis=False, ocr_result=None):
        self._axis = None
        self._show_axis = axis
        self._error = 'ok'
        self._angle = -1
        self._mode = ''
        if ocr_result is not None:
            self._result = ocr_result
            self._fit_direction(image, use_ocr_result=True)
            ax = self._fit_axis()
            self._fit_characters(ax)
        else:
            if isinstance(image, str):
                self._image = cv2.imread(image)
                self._image = cv2.cvtColor(self._image, cv2.COLOR_BGR2RGB)
                self._image = la.image.array_to_image(self._image)
    #             image = la.image.read_image(image)
    #             self._image = la.image.color_convert(image)
            else:
                self._image = image
            self._fit_direction(self._image)
            if isinstance(self._info, str):
                self._fit_direction(la.image.enhance_brightness(self._image, 0.8))
            if isinstance(self._info, str):
                if self._show_axis:
                    return {'data':self._info, 'axis':[], 'angle':0, 'error':self._error}
                else:
                    return {'data':self._info, 'angle':0, 'error':self._error}
            self._transform()
            for i in self._info:
                if '图片模糊' in self._info[i]:
                    self._temp_info = self._info.copy()
                    if self._show_axis:
                        self._temp_axis = self._axis.copy()
                    self._fit_direction(la.image.enhance_brightness(self._image, 0.6))
                    self._transform()
                    if isinstance(self._info, str):
                        self._info = self._temp_info.copy()
                        if self._show_axis:
                            self._temp_axis = self._axis.copy()
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
    
    def _fit_direction(self, image, use_ocr_result=False):
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
                    if '登记事项变更' in i[1][0] or '更正记载' in i[1][0]:
                        break
                    elif '常住人口登记卡' in i[1][0]:
                        break
                    elif '注意事项' in i[1][0]:
                        rank[0] = r
                    elif '户别' in i[1][0] or '家庭户' in i[1][0]:
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
                    self._keys = ['household_type', 'household_name', 'household_id', 'household_address', 'household_date']
                    break
                    
                rank = [0,0,0,0,0,0,0,0,0,0]
                for r, i in enumerate(result[0], start=1):
                    if '登记事项变更' in i[1][0] or '更正记载' in i[1][0]:
                        rank[0] = r
                    elif '常住人口登记卡' in i[1][0]:
                        rank[1] = r
                    elif '姓名' in i[1][0]:
                        rank[2] = r
                    elif '曾用名' in i[1][0] or '性别' in i[1][0]:
                        rank[3] = r
                    elif '出生地' in i[1][0] or '民族' in i[1][0]:
                        rank[4] = r
                    elif '籍贯' in i[1][0] or '出生日期' in i[1][0]:
                        rank[5] = r
                    elif '住址' in i[1][0] or '宗教信仰' in i[1][0]:
                        rank[6] = r
                    elif '身高' in i[1][0] or '血型' in i[1][0]:
                        rank[7] = r
                    elif '文化程度' in i[1][0] or '婚姻状况' in i[1][0]:
                        rank[8] = r
                    elif '服务处所' in i[1][0]:
                        rank[9] = r
                rank = [i for i in rank if i>0]
                if rank==sorted(rank) and len(rank)>1:
                    self._result = result.copy()
                    self._angle = angle
                    self._mode = 'neirong'
                    self._keys = ['register_name', 'register_previous_name', 'register_birthplace', 
                                  'register_nativeplace', 'register_relation', 'register_sex', 
                                  'register_nation', 'register_born', 'register_number',
                                  'register_education', 'register_service_office', 'register_belief', 
                                  'register_height', 'register_blood',  'register_career', 'register_city', 'register_address',
                                  'register_marriage', 'register_military', 'register_date', 'register_content']
                    break
        
        self._info = {}
        if self._mode == 'shouye':
            self._info['household_type'] = '图片模糊:未识别出户别'
            self._info['household_name'] = '图片模糊:未识别出户主'
            self._info['household_id'] = '图片模糊:未识别出户号'
            self._info['household_address'] = '图片模糊:未识别出住址'
            self._info['household_date'] = '图片模糊:未识别出签发日期'
        elif self._mode == 'neirong':
            self._info['register_name'] = '图片模糊:未识别出姓名'
            self._info['register_relation'] = '图片模糊:未识别出与户主关系'
            self._info['register_previous_name'] = '图片模糊:未识别出曾用名'
            self._info['register_sex'] = '图片模糊:未识别出性别'
            self._info['register_birthplace'] = '图片模糊:未识别出出生地'
            self._info['register_nation'] = '图片模糊:未识别出民族'
            self._info['register_nativeplace'] = '图片模糊:未识别出籍贯'
            self._info['register_born'] = '图片模糊:未识别出出生日期'
            self._info['register_belief'] = '图片模糊:未识别出宗教信仰'
            self._info['register_number'] = '图片模糊:未识别出身份证号'
            self._info['register_height'] = '图片模糊:未识别出身高'
            self._info['register_blood'] = '图片模糊:未识别出血型'
            self._info['register_education'] = '图片模糊:未识别出文化程度'
            self._info['register_marriage'] = '图片模糊:未识别出婚姻状况'
            self._info['register_military'] = '图片模糊:未识别出兵役状况'
            self._info['register_service_office'] = '图片模糊:未识别出服务处所'
            self._info['register_career'] = '图片模糊:未识别出职业'
            self._info['register_city'] = '图片模糊:未识别出何时迁入本市县'
            self._info['register_address'] = '图片模糊:未识别出何时迁入本住址'
            self._info['register_date'] = '图片模糊:未识别出登记日期'
            self._info['register_content'] = '图片模糊:未识别出变更内容'
        else:
            self._info = '图片模糊:未识别出有效信息'
            self._error = '图片模糊:未识别出有效信息'
    
    def _transform(self):
        if self._mode == 'shouye':
            axis = self._fit_axis_shouye()
            self._fit_characters_shouye(axis)
        elif self._mode == 'neirong':
            axis = self._fit_axis_neirong()
            self._fit_characters_neirong(axis)
    
    def _fit_axis_neirong(self):
        if len(self._result)==0:
            return 0
        fix_x = []
        axis_true = defaultdict(list)
        axis_dict = defaultdict(list)
        
        axis_fix = ['宗教信仰', '身高', '血型', '婚姻状况']
        axis_fix_h = [i[0][3][1]-i[0][0][1] for i in self._result[0] for j in axis_fix if j in i[1][0]]
        axis_fix_h = sum(axis_fix_h)/len(axis_fix_h) if axis_fix_h else 0
        
        for i in self._result[0]:
            h = (i[0][3][1]+i[0][2][1]-i[0][1][1]-i[0][0][1])/2
            w = (i[0][1][0]+i[0][2][0]-i[0][0][0]-i[0][3][0])/2
            x = min(i[0][0][0], i[0][3][0])
            y = i[0][3][1]+h*0.25
            if '常住人口登记卡' == i[1][0]:
#                 h = h*1.25
                if axis_fix_h>0:
                    h = max(axis_fix_h/0.6, h*1.25)
                w = w/7
                axis_dict['register_name'].append(([x-w*2, y, x+w*4, y+h], 0.8))
                axis_dict['register_previous_name'].append(([x-w*2, y+h, x+w*4, y+h*2], 0.8))
                axis_dict['register_birthplace'].append(([x-w*2, y+h*2, x+w*4, y+h*3], 0.8))
                axis_dict['register_nativeplace'].append(([x-w*2, y+h*3, x+w*4, y+h*4], 0.8))
                axis_dict['register_relation'].append(([x+w*8, y, x+w*13, y+h], 0.8))
                axis_dict['register_sex'].append(([x+w*8, y+h, x+w*13, y+h*2], 0.8))
                axis_dict['register_nation'].append(([x+w*8, y+h*2, x+w*13, y+h*3], 0.8))
                axis_dict['register_born'].append(([x+w*8, y+h*3, x+w*13, y+h*4], 0.8))
                axis_dict['register_number'].append(([x-w*2, y+h*5, x+w*4, y+h*6], 0.6))
                axis_dict['register_education'].append(([x-w*2, y+h*6, x+w*2, y+h*7], 0.6))
                axis_dict['register_service_office'].append(([x-w*2, y+h*7, x+w*6, y+h*8], 0.6))
                axis_dict['register_belief'].append(([x+w*9, y+h*4, x+w*13, y+h*5], 0.6))
                axis_dict['register_height'].append(([x+w*7, y+h*5, x+w*9, y+h*6], 0.6))
                axis_dict['register_blood'].append(([x+w*11, y+h*5, x+w*13, y+h*6], 0.6))
                axis_dict['register_marriage'].append(([x+w*4.5, y+h*6, x+w*6, y+h*7], 0.6))
                axis_dict['register_military'].append(([x+w*9, y+h*6, x+w*13, y+h*7], 0.6))
                axis_dict['register_career'].append(([x+w*9, y+h*7, x+w*13, y+h*8], 0.6))
                axis_dict['register_city'].append(([x-w, y+h*8, x+w*13, y+h*9], 0.6))
                axis_dict['register_address'].append(([x-w, y+h*9, x+w*13, y+h*10], 0.6))
                axis_dict['register_date'].append(([x+w*8.5, y+h*10, x+w*13, y+h*11], 0.6))
                break
        
        for i in self._keys:
            if i not in axis_true:
                if i in axis_dict:
                    weight = sum([j[1] for j in axis_dict[i]])
                    axis_true[i] = [sum([j[0][0]*j[1] for j in axis_dict[i]])/weight,
                                    sum([j[0][1]*j[1] for j in axis_dict[i]])/weight,
                                    sum([j[0][2]*j[1] for j in axis_dict[i]])/weight,
                                    sum([j[0][3]*j[1] for j in axis_dict[i]])/weight]
        return axis_true
        
    def _fit_characters_neirong(self, axis):
        self._axis = axis.copy()
        axis_true = {i:tuple(axis[i]) for i in axis}
        for i in self._result[0]:
            h = (i[0][3][1]+i[0][2][1]-i[0][1][1]-i[0][0][1])/2
            w = (i[0][1][0]+i[0][2][0]-i[0][0][0]-i[0][3][0])/2
            x = min(i[0][0][0], i[0][3][0])
            y = min(i[0][0][1], i[0][1][1])
            if '图片模糊' in self._info['register_name'] and 'register_name' in axis_true:
                h1 = min(max(i[0][3][1], i[0][2][1]), axis_true['register_name'][3])-max(min(i[0][0][1], i[0][1][1]), axis_true['register_name'][1])
                w1 = min(max(i[0][1][0], i[0][2][0]), axis_true['register_name'][2])-max(min(i[0][0][0], i[0][3][0]), axis_true['register_name'][0])            
                if h1/h>0.6 and w1/w>0.6:
                    self._info['register_name'] = i[1][0]
                    self._axis['register_name'] = [x, y]+i[0][2]
                if '图片模糊' not in self._info['register_name']:
                    continue
            if '图片模糊' in self._info['register_relation'] and 'register_relation' in axis_true:
                h1 = min(max(i[0][3][1], i[0][2][1]), axis_true['register_relation'][3])-max(min(i[0][0][1], i[0][1][1]), axis_true['register_relation'][1])
                w1 = min(max(i[0][1][0], i[0][2][0]), axis_true['register_relation'][2])-max(min(i[0][0][0], i[0][3][0]), axis_true['register_relation'][0])            
                if h1/h>0.6 and w1/w>0.6:
                    self._info['register_relation'] = i[1][0]
                    self._axis['register_relation'] = [x, y]+i[0][2]
                if '图片模糊' not in self._info['register_relation']:
                    continue
            if '图片模糊' in self._info['register_previous_name'] and 'register_previous_name' in axis_true:
                h1 = min(max(i[0][3][1], i[0][2][1]), axis_true['register_previous_name'][3])-max(min(i[0][0][1], i[0][1][1]), axis_true['register_previous_name'][1])
                w1 = min(max(i[0][1][0], i[0][2][0]), axis_true['register_previous_name'][2])-max(min(i[0][0][0], i[0][3][0]), axis_true['register_previous_name'][0])            
                if h1/h>0.6 and w1/w>0.6:
                    self._info['register_previous_name'] = i[1][0]
                    self._axis['register_previous_name'] = [x, y]+i[0][2]
                if '图片模糊' not in self._info['register_previous_name']:
                    continue
            if '图片模糊' in self._info['register_sex'] and 'register_sex' in axis_true:
                h1 = min(max(i[0][3][1], i[0][2][1]), axis_true['register_sex'][3])-max(min(i[0][0][1], i[0][1][1]), axis_true['register_sex'][1])
                w1 = min(max(i[0][1][0], i[0][2][0]), axis_true['register_sex'][2])-max(min(i[0][0][0], i[0][3][0]), axis_true['register_sex'][0])            
                if h1/h>0.6 and w1/w>0.6:
                    self._info['register_sex'] = i[1][0]
                    self._axis['register_sex'] = [x, y]+i[0][2]
                if '图片模糊' not in self._info['register_sex']:
                    continue
            if '图片模糊' in self._info['register_birthplace'] and 'register_birthplace' in axis_true:
                h1 = min(max(i[0][3][1], i[0][2][1]), axis_true['register_birthplace'][3])-max(min(i[0][0][1], i[0][1][1]), axis_true['register_birthplace'][1])
                w1 = min(max(i[0][1][0], i[0][2][0]), axis_true['register_birthplace'][2])-max(min(i[0][0][0], i[0][3][0]), axis_true['register_birthplace'][0])            
                if h1/h>0.6 and w1/w>0.6:
                    self._info['register_birthplace'] = i[1][0]
                    self._axis['register_birthplace'] = [x, y]+i[0][2]
                if '图片模糊' not in self._info['register_birthplace']:
                    continue
            if '图片模糊' in self._info['register_nation'] and 'register_nation' in axis_true:
                h1 = min(max(i[0][3][1], i[0][2][1]), axis_true['register_nation'][3])-max(min(i[0][0][1], i[0][1][1]), axis_true['register_nation'][1])
                w1 = min(max(i[0][1][0], i[0][2][0]), axis_true['register_nation'][2])-max(min(i[0][0][0], i[0][3][0]), axis_true['register_nation'][0])            
                if h1/h>0.6 and w1/w>0.6:
                    self._info['register_nation'] = i[1][0]
                    self._axis['register_nation'] = [x, y]+i[0][2]
                if '图片模糊' not in self._info['register_nation']:
                    continue
            if '图片模糊' in self._info['register_nativeplace'] and 'register_nativeplace' in axis_true:
                h1 = min(max(i[0][3][1], i[0][2][1]), axis_true['register_nativeplace'][3])-max(min(i[0][0][1], i[0][1][1]), axis_true['register_nativeplace'][1])
                w1 = min(max(i[0][1][0], i[0][2][0]), axis_true['register_nativeplace'][2])-max(min(i[0][0][0], i[0][3][0]), axis_true['register_nativeplace'][0])            
                if h1/h>0.6 and w1/w>0.6:
                    self._info['register_nativeplace'] = i[1][0]
                    self._axis['register_nativeplace'] = [x, y]+i[0][2]
                if '图片模糊' not in self._info['register_nativeplace']:
                    continue
            if '图片模糊' in self._info['register_born'] and 'register_born' in axis_true:
                h1 = min(max(i[0][3][1], i[0][2][1]), axis_true['register_born'][3])-max(min(i[0][0][1], i[0][1][1]), axis_true['register_born'][1])
                w1 = min(max(i[0][1][0], i[0][2][0]), axis_true['register_born'][2])-max(min(i[0][0][0], i[0][3][0]), axis_true['register_born'][0])            
                if h1/h>0.6 and w1/w>0.6:
                    self._info['register_born'] = i[1][0]
                    self._axis['register_born'] = [x, y]+i[0][2]
                if '图片模糊' not in self._info['register_born']:
                    continue
            if '图片模糊' in self._info['register_belief'] and 'register_belief' in axis_true:
                h1 = min(max(i[0][3][1], i[0][2][1]), axis_true['register_belief'][3])-max(min(i[0][0][1], i[0][1][1]), axis_true['register_belief'][1])
                w1 = min(max(i[0][1][0], i[0][2][0]), axis_true['register_belief'][2])-max(min(i[0][0][0], i[0][3][0]), axis_true['register_belief'][0])            
                if h1/h>0.6 and w1/w>0.6:
                    self._info['register_belief'] = i[1][0]
                    self._axis['register_belief'] = [x, y]+i[0][2]
                if '图片模糊' not in self._info['register_belief']:
                    continue
            if '图片模糊' in self._info['register_number'] and 'register_number' in axis_true:
                h1 = min(max(i[0][3][1], i[0][2][1]), axis_true['register_number'][3])-max(min(i[0][0][1], i[0][1][1]), axis_true['register_number'][1])
                w1 = min(max(i[0][1][0], i[0][2][0]), axis_true['register_number'][2])-max(min(i[0][0][0], i[0][3][0]), axis_true['register_number'][0])            
                if h1/h>0.6 and w1/w>0.6:
                    self._info['register_number'] = i[1][0]
                    self._axis['register_number'] = [x, y]+i[0][2]
                if '图片模糊' not in self._info['register_number']:
                    continue
            if '图片模糊' in self._info['register_height'] and 'register_height' in axis_true:
                h1 = min(max(i[0][3][1], i[0][2][1]), axis_true['register_height'][3])-max(min(i[0][0][1], i[0][1][1]), axis_true['register_height'][1])
                w1 = min(max(i[0][1][0], i[0][2][0]), axis_true['register_height'][2])-max(min(i[0][0][0], i[0][3][0]), axis_true['register_height'][0])            
                if h1/h>0.6 and w1/w>0.6:
                    self._info['register_height'] = i[1][0]
                    self._axis['register_height'] = [x, y]+i[0][2]
                if '图片模糊' not in self._info['register_height']:
                    continue
            if '图片模糊' in self._info['register_blood'] and 'register_blood' in axis_true:
                h1 = min(max(i[0][3][1], i[0][2][1]), axis_true['register_blood'][3])-max(min(i[0][0][1], i[0][1][1]), axis_true['register_blood'][1])
                w1 = min(max(i[0][1][0], i[0][2][0]), axis_true['register_blood'][2])-max(min(i[0][0][0], i[0][3][0]), axis_true['register_blood'][0])            
                if h1/h>0.6 and w1/w>0.6:
                    self._info['register_blood'] = i[1][0].replace('0', 'o')
                    self._axis['register_blood'] = [x, y]+i[0][2]
                if '图片模糊' not in self._info['register_blood']:
                    continue
            if '图片模糊' in self._info['register_education'] and 'register_education' in axis_true:
                h1 = min(max(i[0][3][1], i[0][2][1]), axis_true['register_education'][3])-max(min(i[0][0][1], i[0][1][1]), axis_true['register_education'][1])
                w1 = min(max(i[0][1][0], i[0][2][0]), axis_true['register_education'][2])-max(min(i[0][0][0], i[0][3][0]), axis_true['register_education'][0])            
                if h1/h>0.6 and w1/w>0.6:
                    self._info['register_education'] = i[1][0]
                    self._axis['register_education'] = [x, y]+i[0][2]
                if '图片模糊' not in self._info['register_education']:
                    continue 
            if '图片模糊' in self._info['register_marriage'] and 'register_marriage' in axis_true:
                h1 = min(max(i[0][3][1], i[0][2][1]), axis_true['register_marriage'][3])-max(min(i[0][0][1], i[0][1][1]), axis_true['register_marriage'][1])
                w1 = min(max(i[0][1][0], i[0][2][0]), axis_true['register_marriage'][2])-max(min(i[0][0][0], i[0][3][0]), axis_true['register_marriage'][0])            
                if h1/h>0.6 and w1/w>0.6:
                    self._info['register_marriage'] = i[1][0]
                    self._axis['register_marriage'] = [x, y]+i[0][2]
                if '图片模糊' not in self._info['register_marriage']:
                    continue
            if '图片模糊' in self._info['register_military'] and 'register_military' in axis_true:
                h1 = min(max(i[0][3][1], i[0][2][1]), axis_true['register_military'][3])-max(min(i[0][0][1], i[0][1][1]), axis_true['register_military'][1])
                w1 = min(max(i[0][1][0], i[0][2][0]), axis_true['register_military'][2])-max(min(i[0][0][0], i[0][3][0]), axis_true['register_military'][0])            
                if h1/h>0.6 and w1/w>0.6:
                    self._info['register_military'] = i[1][0]
                    self._axis['register_military'] = [x, y]+i[0][2]
                if '图片模糊' not in self._info['register_military']:
                    continue
            if '图片模糊' in self._info['register_service_office'] and 'register_service_office' in axis_true:
                h1 = min(max(i[0][3][1], i[0][2][1]), axis_true['register_service_office'][3])-max(min(i[0][0][1], i[0][1][1]), axis_true['register_service_office'][1])
                w1 = min(max(i[0][1][0], i[0][2][0]), axis_true['register_service_office'][2])-max(min(i[0][0][0], i[0][3][0]), axis_true['register_service_office'][0])            
                if h1/h>0.6 and w1/w>0.6:
                    self._info['register_service_office'] = i[1][0]
                    self._axis['register_service_office'] = [x, y]+i[0][2]
                if '图片模糊' not in self._info['register_service_office']:
                    continue
            if '图片模糊' in self._info['register_career'] and 'register_career' in axis_true:
                h1 = min(max(i[0][3][1], i[0][2][1]), axis_true['register_career'][3])-max(min(i[0][0][1], i[0][1][1]), axis_true['register_career'][1])
                w1 = min(max(i[0][1][0], i[0][2][0]), axis_true['register_career'][2])-max(min(i[0][0][0], i[0][3][0]), axis_true['register_career'][0])            
                if h1/h>0.6 and w1/w>0.6:
                    self._info['register_career'] = i[1][0]
                    self._axis['register_career'] = [x, y]+i[0][2]
                if '图片模糊' not in self._info['register_career']:
                    continue
            if '图片模糊' in self._info['register_city'] and 'register_city' in axis_true:
                h1 = min(max(i[0][3][1], i[0][2][1]), axis_true['register_city'][3])-max(min(i[0][0][1], i[0][1][1]), axis_true['register_city'][1])
                w1 = min(max(i[0][1][0], i[0][2][0]), axis_true['register_city'][2])-max(min(i[0][0][0], i[0][3][0]), axis_true['register_city'][0])            
                if h1/h>0.6 and w1/w>0.6:
                    self._info['register_city'] = i[1][0]
                    self._axis['register_city'] = [x, y]+i[0][2]
                if '图片模糊' not in self._info['register_city']:
                    continue
            if '图片模糊' in self._info['register_address'] and 'register_address' in axis_true:
                h1 = min(max(i[0][3][1], i[0][2][1]), axis_true['register_address'][3])-max(min(i[0][0][1], i[0][1][1]), axis_true['register_address'][1])
                w1 = min(max(i[0][1][0], i[0][2][0]), axis_true['register_address'][2])-max(min(i[0][0][0], i[0][3][0]), axis_true['register_address'][0])            
                if h1/h>0.6 and w1/w>0.6:
                    self._info['register_address'] = i[1][0]
                    self._axis['register_address'] = [x, y]+i[0][2]
                if '图片模糊' not in self._info['register_address']:
                    continue
            if '图片模糊' in self._info['register_date'] and 'register_date' in axis_true:
                h1 = min(max(i[0][3][1], i[0][2][1]), axis_true['register_date'][3])-max(min(i[0][0][1], i[0][1][1]), axis_true['register_date'][1])
                w1 = min(max(i[0][1][0], i[0][2][0]), axis_true['register_date'][2])-max(min(i[0][0][0], i[0][3][0]), axis_true['register_date'][0])            
                if h1/h>0.6 and w1/w>0.6:
                    self._info['register_date'] = i[1][0][-10:]
                    self._axis['register_date'] = [x, y]+i[0][2]
                if '图片模糊' not in self._info['register_date']:
                    continue 
        
        if '图片模糊' in self._info['register_nation']:
            self._info['register_nation'] = '汉'
        if '图片模糊' in self._info['register_date']:
            if self._info['register_address'].find('年')==4 and '月' in self._info['register_address']:
                self._info['register_date'] = self._info['register_address'][:self._info['register_address'].find('日')+1]
        for i in self._info:
            if '图片模糊' in self._info[i]:
                self._info[i] = ''
            
        for i in self._axis:
            self._axis[i] = [int(max(0, j)) for j in self._axis[i]]
            
        t = ['register_name', 'register_previous_name', 'register_relation', 'register_sex', 
              'register_born', 'register_number', 'register_education', 'register_service_office', 
              'register_marriage', 'register_military', 'register_career', 'register_city', 
              'register_address', 'register_date', 'register_content']
        self._info = {i:self._info[i] for i in t}
            
    
    def _fit_axis_shouye(self):
        if len(self._result)==0:
            return 0
        fix_x = []
        axis_true = defaultdict(list)
        axis_dict = defaultdict(list)

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
                        axis_dict['household_address'].append(([x+w*12, y+h, x+w*24, y+h*4], 0.6))
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
                            axis_true['household_name'] = [x+w*4, y, x+w*14, y+h*1.5]
                        axis_dict['household_type'].append(([x-w*8, y, x, y+h*1.5], 0.8))
                        axis_dict['household_id'].append(([x-w*8, y+h, x-w*2, y+h*3], 0.6))
                        axis_dict['household_address'].append(([x+w*2, y+h, x+w*16, y+h*4], 0.8))
                        break
                if 'household_name' in axis_true:
                    continue
            if 'household_id' not in axis_true:
                for char in self._char_household_id:
                    if char in i[1][0] and '外部' not in i[1][0]:
                        if len(i[1][0][i[1][0].find(char)+len(char):])>1:
                            w = w/(len(i[1][0])+2)
                            axis_true['household_id'] = [x+w*3, y]+i[0][2]
                        else:
                            w = w/(len(i[1][0])+1.5)
                            axis_true['household_id'] = [x+w*3, y, x+w*9, y+h]
                        axis_dict['household_type'].append(([x+w*3, y-h*2, x+w*10, y], 0.8))
                        axis_dict['household_name'].append(([x+w*14, y+h*2, x+w*24, y], 0.6))
                        axis_dict['household_address'].append(([x+w*12, y, x+w*24, y+h*2], 0.8))
                        break
                if 'household_id' in axis_true:
                    continue
            if 'household_address' not in axis_true:
                for char in self._char_household_address:
                    if char in i[1][0]:
                        if len(i[1][0][i[1][0].find(char)+len(char):])>1:
                            w = w/(len(i[1][0])+2)
                            axis_true['household_address'] = [x+w*3, y, x+w*25, y+h*2]
                        else:
                            w = w/(len(i[1][0])+1)
                            axis_true['household_address'] = [x+w*3, y, x+w*25, y+h*2]
                        axis_dict['household_type'].append(([x-w*6, y-h*2, x+w, y], 0.6))
                        axis_dict['household_name'].append(([x+w*5, y-h*2, x+w*14, y], 0.8))
                        axis_dict['household_id'].append(([x-w*6, y, x-w, y+h], 0.8))
                        break
                if 'household_address' in axis_true:
                    continue
#         print(axis_true, axis_dict)
        for i in ['household_type', 'household_name', 'household_id', 'household_address']:
            if i not in axis_true:
                if i in axis_dict:
                    weight = sum([j[1] for j in axis_dict[i]])
                    axis_true[i] = [sum([j[0][0]*j[1] for j in axis_dict[i]])/weight,
                                    sum([j[0][1]*j[1] for j in axis_dict[i]])/weight,
                                    sum([j[0][2]*j[1] for j in axis_dict[i]])/weight,
                                    sum([j[0][3]*j[1] for j in axis_dict[i]])/weight]
        return axis_true

    def _fit_characters_shouye(self, axis):
        self._axis = axis.copy()
        axis_true = {i:tuple(axis[i]) for i in axis}
        address = ''
        for i in self._result[0]:
            h = (i[0][3][1]+i[0][2][1]-i[0][1][1]-i[0][0][1])/2
            w = (i[0][1][0]+i[0][2][0]-i[0][0][0]-i[0][3][0])/2
            if h==0:
                h = 1
            if w==0:
                w = 1
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
                if h1/h>0.6 and w1/w>0.6 and '家庭' not in i[1][0]:
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
                    if len(i[1][0][i[1][0].find('址')+len('址'):])>1 and '址' in i[1][0]:
                        address += i[1][0][i[1][0].find('址')+1:]
                        self._axis['household_address'] = [self._axis['household_address'][0], y]+i[0][2]
                    elif len(i[1][0])>1:
                        if address=='':
                            self._axis['household_address'] = [x, y]+i[0][2]
                        else:
                            self._axis['household_address'][3] = i[0][2][1]
                        address += i[1][0]
            if '图片模糊' in self._info['household_date']:
                if '年' in i[1][0] and '月' in i[1][0] and i[1][0].endswith('日'):
                    self._info['household_date'] = i[1][0][max(0, i[1][0].find('年')-4):]
                    continue

        if '图片模糊' in self._info['household_address'] and address!='':
            self._info['household_address'] = address
        if '图片模糊' in self._info['household_id']:
            self._info['household_id'] = ''
        if self._info['household_type'][0]=='非':
            self._info['household_type'] = '非农业家庭户'
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

    def metrics(self, data, name_list=None, debug=False):
        if la.gfile.isfile(data):
            with open(data) as f:
                data = f.read().replace('\n', '').replace('}', '}\n').strip().split('\n')
            data = [eval(i) for i in data]
        if name_list is None:
            name_list = ['household_type', 'household_name', 'household_id', 'household_address', 'household_date',
                         'register_name', 'register_previous_name', 'register_relation', 'register_sex', 
                         'register_born', 'register_number', 'register_education', 'register_service_office', 
                         'register_marriage', 'register_military', 'register_career','register_city', 
                         'register_address', 'register_date', 'register_content']

        score_a = {i:0 for i in name_list}
        score_b = {i:0.0000001 for i in name_list}
        error_list = []
        n = 0
        for i in data:
            error = {'image':i.pop('image')}
            try:
                t = self.predict(error['image'])['data']
                if isinstance(t, dict):
                    for j in name_list:
                        if j in t and j in i:
                            if t[j]==i[j]:
                                score_a[j] +=1
                            else:
                                error[j] = {'pred':t[j], 'label':i[j]}
                        score_b[j] += 1
            except:
                for j in name_list:
                    score_b[j] += 1
                error['error'] = 'program error'
            if len(error)>1:
                error_list.append(error)
            n += 1
        
        score = {f'{i}_acc':score_a[i]/score_b[i] for i in score_a}
        score['totalmean_acc'] = sum([score_a[i] for i in score_a])/sum([score_b[i] for i in score_b])
        score = {i:round(score[i], 4) for i in score}
        score['test_sample_nums'] = len(data)
        if debug:
            score['error'] = error_list
        return score
    
    def metrics1(self, image_list, label_list=None, debug=False):
        types = 0
        names = 0
        ids = 0
        addresss = 0
        dates = 0

        household_type = 0.0000001
        household_name = 0.0000001
        household_id = 0.0000001
        household_address = 0.0000001
        household_date = 0.0000001
    
        name = 0
        previous_name = 0
        relation = 0
        sex = 0
        born = 0
        number = 0
        education = 0
        service_office = 0
        marriage = 0
        military = 0
        career = 0
        city = 0
        address = 0
        date = 0
        content = 0
        
        register_name = 0.0000001
        register_previous_name = 0.0000001
        register_relation = 0.0000001
        register_sex = 0.0000001
        register_born = 0.0000001
        register_number = 0.0000001
        register_education = 0.0000001
        register_service_office = 0.0000001
        register_marriage = 0.0000001
        register_military = 0.0000001
        register_career = 0.0000001
        register_city = 0.0000001
        register_address = 0.0000001
        register_date = 0.0000001
        register_content = 0.0000001

        error_list = []
        for r, i in enumerate(image_list):
            error = {'image':i}
            if label_list is None:
                label = i.split('$$')[1:-1]
            else:
                label = label_list[r].split('$$')[1:-1]
            t = self.predict(i)['data']
            try:
                if isinstance(t, dict):
                    if len(label)==5:
                        if t['household_type']==label[0]:
                            types += 1
                        else:
                            error['household_type'] = {'pred':t['household_type'], 'label':label[0]}
                        if t['household_name']==label[1]:
                            names += 1
                        else:
                            error['household_name'] = {'pred':t['household_name'], 'label':label[1]}
                        if t['household_id']==label[2]:
                            ids += 1
                        else:
                            error['household_id'] = {'pred':t['household_id'], 'label':label[2]}
                        if t['household_address']==label[3]:
                            addresss += 1
                        else:
                            error['household_address'] = {'pred':t['household_address'], 'label':label[3]}
                        if t['household_date']==label[4]:
                            dates += 1
                        else:
                            error['household_date'] = {'pred':t['household_date'], 'label':label[4]}
                    elif len(label)==15:
                        if t['register_name']==label[0]:
                            name += 1
                        else:
                            error['register_name'] = {'pred':t['register_name'], 'label':label[0]}
                        if t['register_previous_name']==label[1]:
                            previous_name += 1
                        else:
                            error['register_previous_name'] = {'pred':t['register_previous_name'], 'label':label[1]}
                        if t['register_relation']==label[2]:
                            relation += 1
                        else:
                            error['register_relation'] = {'pred':t['register_relation'], 'label':label[2]}
                        if t['register_sex']==label[3]:
                            sex += 1
                        else:
                            error['register_sex'] = {'pred':t['register_sex'], 'label':label[3]}
                        if t['register_born']==label[4]:
                            born += 1
                        else:
                            error['register_born'] = {'pred':t['register_born'], 'label':label[4]}
                        if t['register_number']==label[5]:
                            number += 1
                        else:
                            error['register_number'] = {'pred':t['register_number'], 'label':label[5]}
                        if t['register_education']==label[6]:
                            education += 1
                        else:
                            error['register_education'] = {'pred':t['register_education'], 'label':label[6]}
                        if t['register_service_office']==label[7]:
                            service_office += 1
                        else:
                            error['register_service_office'] = {'pred':t['register_service_office'], 'label':label[7]}
                        if t['register_marriage']==label[8]:
                            marriage += 1
                        else:
                            error['register_marriage'] = {'pred':t['register_marriage'], 'label':label[8]}
                        if t['register_military']==label[9]:
                            military += 1
                        else:
                            error['register_military'] = {'pred':t['register_military'], 'label':label[9]}
                        if t['register_career']==label[10]:
                            career += 1
                        else:
                            error['register_career'] = {'pred':t['register_career'], 'label':label[10]}
                        if t['register_city']==label[11]:
                            city += 1
                        else:
                            error['register_city'] = {'pred':t['register_city'], 'label':label[11]}
                        if t['register_address']==label[12]:
                            address += 1
                        else:
                            error['register_address'] = {'pred':t['register_address'], 'label':label[12]}
                        if t['register_date']==label[13]:
                            date += 1
                        else:
                            error['register_date'] = {'pred':t['register_date'], 'label':label[13]}
                        if t['register_content']==label[14]:
                            content += 1
                        else:
                            error['register_content'] = {'pred':t['register_content'], 'label':label[14]}
            except:
                pass
            if len(label)==5:
                household_type += 1
                household_name += 1
                household_id += 1
                household_address += 1
                household_date += 1
            elif len(label)==15:
                register_name += 1
                register_previous_name += 1
                register_relation += 1
                register_sex += 1
                register_born += 1
                register_number += 1
                register_education += 1
                register_service_office += 1
                register_marriage += 1
                register_military += 1
                register_career += 1
                register_city += 1
                register_address += 1
                register_date += 1
                register_content += 1
            if len(error)>1:
                error_list.append(error)

        ok = (name+previous_name+relation+sex+born+number+education+service_office
              +marriage+military+career++city+address+date+content
              +types+names+ids+addresss+dates)
        total = (register_name+register_previous_name+register_relation+register_sex
                 +register_born+register_number+register_education+register_service_office
                 +register_marriage+register_military+register_career+register_city
                 +register_address+register_date+register_content
                 +household_type+household_name+household_id+household_address+household_date)
        result = {'household_type_acc':types/household_type, 'household_name_acc':names/household_name, 
                  'household_id_acc':ids/household_id,
                  'household_address_acc':addresss/household_address, 'household_date_acc':dates/household_date, 
                  'register_name_acc':name/register_name, 'register_previous_name_acc':previous_name/register_previous_name,
                  'register_relation_acc':relation/register_relation, 'register_sex_acc':sex/register_sex,
                  'register_born_acc':born/register_born, 'register_number_acc':number/register_number,
                  'register_education_acc':education/register_education, 
                  'register_service_office_acc':service_office/register_service_office,
                  'register_marriage_acc':marriage/register_marriage, 'register_military_acc':military/register_military,
                  'register_career_acc':career/register_career, 'register_city_acc':city/register_city, 
                  'register_address_acc':address/register_address, 'register_date_acc':date/register_date,
                  'register_content_acc':content/register_content, 'totalmean_acc':ok/total,
                  'test_sample_nums':len(image_list)
                 }
        result = {i:round(result[i], 4) for i in result}
        if debug:
            result['error'] = error_list
        return result



