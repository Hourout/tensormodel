from collections import defaultdict

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
        self._char_marriage_id = ['结婚证字号']
        self._char_user_name = ['姓名']
        self._char_user_country = ['国籍', '国箱']
        
    def predict(self, image):
        self._marriage_name_prob = 0
        self._axis = None
        self._error = 'ok'
        self._image = image
        
        self._direction_transform(self._image)
        if isinstance(self._info, str):
            self._direction_transform(self._image, 0.8)
        if isinstance(self._info, str):
            return {'data':self._info, 'axis':[], 'angle':0, 'error':self._error}
        self._axis_transform_up()
        for i in self._info:
            if '图片模糊' in self._info[i]:
                self._temp = self._info.copy()
                self._direction_transform(self._image, 0.6)
                self._axis_transform_up()
                if isinstance(self._info, str):
                    self._info = self._temp.copy()
                else:
                    for j in self._temp:
                        if '图片模糊' not in self._temp[j]:
                            self._info[j] = self._temp[j]
                break
        self._error = 'ok'
        for i in self._info:
            if '图片模糊' in self._info[i]:
                self._error = self._info[i]
                break
        return {'data':self._info, 'axis':self._axis, 'angle':self._angle, 'error':self._error}
        
    def _direction_transform(self, image, aug=0):
        state = False
        self._result = []
        for angle in [0, 90, 180, 270]:
            if angle==0 and isinstance(image, str):
                result = self.ocr.ocr(image, cls=False)
            else:
                image1 = la.image.read_image(image)
                image1 = la.image.color_convert(image1)
                if aug!=0:
                    image1 = la.image.enhance_brightness(image1, aug)
                if angle>0:
                    image1 = la.image.rotate(image1, angle, expand=True)
                image1 = la.image.image_to_array(image1)       
                result = self.ocr.ocr(image1, cls=False)
            if not state:
                rank = [0,0,0,0,0]
                for r, i in enumerate(result[0], start=1):
                    if '持证人' in i[1][0]:
                        rank[0] = r
                    elif '登记日期' in i[1][0]:
                        rank[1] = r
                    elif '结婚证字号'in i[1][0]:
                        rank[2] = r
                    elif '备注' in i[1][0]:
                        rank[3] = r
                    elif '身份证件号' in i[1][0]:
                        rank[4] = r
                rank = [i for i in rank if i>0]
                if rank==sorted(rank) and len(rank)>1:
                    state = True
                    self._result = result.copy()
                    self._angle = angle
                    break
        
        self._info = {}
        if state:
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
                            self._info['marriage_name'] = (i[1][0][i[1][0].find(char)+len(char):]).strip()
                            w = w/(len(i[1][0])+4)
                            axis_true['marriage_name'] = [x+w*5, y]+i[0][2]
                            self._marriage_name_prob = i[1][1]
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
                        if len(i[1][0])>2:
                            axis_true['user_name_up'] = [x+w*0.2, y]+i[0][2]
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
                        if len(i[1][0])>2:
                            axis_true['user_name_down'] = [x+w*0.2, y]+i[0][2]
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
#                 else:
#                     self._error = '图片模糊:未识别出有效信息'
#                     return 0
        if self._axis is None:
            self._axis = axis_true.copy()
        for i in axis_true:
            axis_true[i] = tuple(axis_true[i])
        
        step = 0
        for i in self._result[0]:
            h = (i[0][3][1]+i[0][2][1]-i[0][1][1]-i[0][0][1])/2
            w = (i[0][1][0]+i[0][2][0]-i[0][0][0]-i[0][3][0])/2
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
                        self._info['marriage_date'] = i[1][0][i[1][0].find('期')+1:]
                        self._axis['marriage_date'] = [self._axis['marriage_date'][0], y]+i[0][2]
                    elif len(i[1][0]) in [9, 10, 11] and i[1][0].find('年')==4 and '月' in i[1][0] and i[1][0].endswith('日'):
                        self._info['marriage_date'] = i[1][0]
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
            if '图片模糊' in self._info['user_name_up'] and 'user_name_up' in axis_true:
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
            if '图片模糊' in self._info['user_number_up'] and 'user_number_up' in axis_true and step==0:
                if sum([1 for j in i[1][0][-18:] if j in '0123456789xX'])==18:
                    self._info['user_number_up'] = i[1][0][-18:]
                    self._info['user_sex_up'] =  '男' if int(self._info['user_number_up'][16])%2 else '女'
                    self._info['user_born_up'] = f"{self._info['user_number_up'][6:10]}年{self._info['user_number_up'][10:12]}月{self._info['user_number_up'][12:14]}日"
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
                    elif len(i[1][0])>1:
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
            if '图片模糊' in self._info['user_number_down'] and 'user_number_down' in axis_true:
                if sum([1 for j in i[1][0][-18:] if j in '0123456789xX'])==18:
                    self._info['user_number_down'] = i[1][0][-18:]
                    self._info['user_sex_down'] =  '男' if int(self._info['user_number_down'][16])%2 else '女'
                    self._info['user_born_down'] = f"{self._info['user_number_down'][6:10]}年{self._info['user_number_down'][10:12]}月{self._info['user_number_down'][12:14]}日"

        
        if '图片模糊' in self._info['user_name_up'] and '图片模糊' not in self._info['marriage_name']:
            self._info['user_name_up'] = self._info['marriage_name']
        if '图片模糊' in self._info['user_country_up']:
            self._info['user_country_up'] = '中国'
        if '图片模糊' in self._info['user_country_down']:
            self._info['user_country_down'] = '中国'
        if '图片模糊' in self._info['user_sex_down'] and '图片模糊' in self._info['user_sex_up']:
            self._info['user_sex_down'] = '男'
            self._info['user_sex_up'] = '女'
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
                    self._info['user_born_up'] = date[0] if self._info['user_born_down']==date[1] else date[1]
        
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
            image = la.image.read_image(self._image)
            image = la.image.color_convert(image)
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



