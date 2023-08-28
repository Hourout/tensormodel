import time

import paddleocr
import linora as la

__all__ = ['OCRJieHunZheng']


class OCRJieHunZheng():
    def __init__(self, model=True, name_list=None):
        if model==True:
            self._model = paddleocr.PaddleOCR(show_log=False)
        elif model:
            self._model = model
        else:
            self._model = None
        self._keys = ['marriage_name', 'marriage_date', 'marriage_id', 'marriage_type',
                      'user_name_up', 'user_sex_up', 'user_country_up', 'user_born_up', 'user_number_up', 
                      'user_name_down', 'user_sex_down', 'user_country_down', 'user_born_down', 'user_number_down']
        if name_list is None:
            name_list = self._keys.copy()
        else:
            for i in name_list:
                if i not in self._keys:
                    raise ValueError(f'Variable name `{i}`  does not conform to the specification.')
        self._name_list = name_list
        self._char_marriage_name = ['持证', '证人']
        self._char_marriage_date = ['登记日', '记日']
        self._char_marriage_id = ['结婚证字', '离婚证字', '洁婚证字']
        self._char_user_name = ['姓名', '姓多', '姓爸', '姓吉']
        self._char_user_country = ['国籍', '国箱', '国馨', '国精']
        self._char_user_sex = ['性别']
        self._char_user_born = ['出生日期']
        self._char_user_number = ['身份证件号', '身份证件导']
        
    def predict(self, image, axis=False, model=None):
        self._axis_up_down = 0
        self._marriage_name_prob = 0
        self._show_axis = axis
        self._info = '图片模糊或非二代结离婚证图片'
        self._error = '图片模糊或非二代结离婚证图片'
        
        if isinstance(image, str):
            self._image = la.image.color_convert(la.image.read_image(image))
        else:
            self._image = image
        self._fit_direction(self._model if model is None else model)
        if isinstance(self._info, str):
            if self._show_axis:
                return {'data':self._info, 'axis':[], 'angle':0, 'error':self._error}
            else:
                return {'data':self._info, 'angle':0, 'error':self._error}
        self._fit_axis()
        self._fit_characters(self._axis, self._result)
        
        for aug in [0,1,2]:
            error_list = [i for i in self._info if '图片模糊' in self._info[i]]
            if error_list:
                self._result_crop = []
                for i in error_list:
                    if i not in self._axis:
                        continue
                    image = la.image.crop(self._image, self._axis[i])
                    if aug==0:
                        image = la.image.image_to_array(image)
                    elif aug==1:
                        image = la.image.image_to_array(la.image.color_convert(image, la.image.ColorMode.grayscale))[:,:,0]
                    elif aug==2:
                        image = la.image.image_to_array(la.image.enhance_brightness(image, 0.8))
                    t = (self._model if model is None else model).ocr(image, cls=False)
                    if t[0]:
                        for j in t[0]:
                            self._result_crop.append([[self._axis[i][:2], [self._axis[i][2], self._axis[i][1]], 
                                                       self._axis[i][2:], [self._axis[i][0], self._axis[i][3]]], j[1]])
                self._fit_characters(self._axis, [self._result_crop])

        self._error = '图片模糊' if [1 for i in self._info if '图片模糊' in self._info[i]] else 'ok'
        self._info = {i:('' if '图片模糊' in j else j) for i,j in self._info.items()}
        if self._show_axis:
            return {'data':self._info, 'axis':self._axis, 'angle':self._angle, 'error':self._error}
        else:
            return {'data':self._info, 'angle':self._angle, 'error':self._error}
        
    def _fit_direction(self, model):
        for angle in [0, 90, 270, 180]:
            image = la.image.rotate(self._image, angle, expand=True)
            self._result = model.ocr(la.image.image_to_array(image), cls=False)
            logic = 0
            logic_r = 0
            number_n = 0
            number_r = 0
            rank = [0,0,0,0,0]
            for r, i in enumerate(self._result[0], start=1):
                if '持证' in i[1][0] or '证人' in i[1][0]:
                    if rank[0]==0:
                        rank[0] = r
                elif '登记日期' in i[1][0]:
                    if rank[1]==0:
                        rank[1] = r
                elif '婚证字号'in i[1][0]:
                    if rank[2]==0:
                        rank[2] = r
                elif '备注' in i[1][0]:
                    if rank[3]==0:
                        rank[3] = r
                elif '国籍' in i[1][0] or '姓名' in i[1][0]:
                    if logic==0:
                        logic = r
                elif '出生日期' in i[1][0] or '性别' in i[1][0]:
                    logic_r = r
                elif '身份证件号' in i[1][0]:
                    number_n += 1
                    number_r = r
                    if rank[4]==0:
                        rank[4] = r
            if 0<logic<max(rank[:4]):
                continue
            if number_n==2 and number_r<logic_r:
                continue
            rank = [i for i in rank if i>0]
            if rank==sorted(rank) and len(rank)>1:
                self._image = image
                self._angle = angle
                self._info = {i:'图片模糊' for i in self._name_list}
                break
    
    def _fit_axis(self):
        if len(self._result)==0:
            return 0

        axis_true = dict()
        axis_dict = {i:[] for i in self._keys}
        
        axis_up_down = []
        for var, h in [(self._char_user_name, 5), (self._char_user_country, 3), (self._char_user_sex, 5), 
                         (self._char_user_born, 3), (self._char_user_number, 1)]:
            temp = [[i[0][0][1]+(i[0][2][1]-i[0][1][1])*h for char in var if char in i[1][0]] for i in self._result[0]]
            temp = [i[0] for i in temp if i]
            if len(temp)>1:
                axis_up_down.append(temp[0])
        self._axis_up_down = sum(axis_up_down)/len(axis_up_down) if axis_up_down else 0
        
        for i in self._result[0]:
            h = max((i[0][3][1]+i[0][2][1]-i[0][1][1]-i[0][0][1])/2, 1)
            w = max((i[0][1][0]+i[0][2][0]-i[0][0][0]-i[0][3][0])/2, 1)
            x = min(i[0][0][0], i[0][3][0])
            y = min(i[0][0][1], i[0][1][1])
            if 'marriage_name' not in axis_true:
                temp = [char for char in self._char_marriage_name if char in i[1][0]]
                if temp:
                    if len(i[1][0][i[1][0].find(temp[0])+len(temp[0]):])>1:
                        w = w*3/len(i[1][0])
                    axis_true['marriage_name'] = [x, y-h*2, x+w*3, y+h*5.5]
                    axis_dict['marriage_date'].append(([x, y+h*3, x+w*3, y+h*6.5], 0.8))
                    axis_dict['marriage_id'].append(([x, y+h*6, x+w*4.5, y+h*9.5], 0.6))
                    continue
            if 'marriage_date' not in axis_true:
                temp = [char for char in self._char_marriage_date if char in i[1][0]]
                if temp:
                    if len(i[1][0][i[1][0].find(temp[0])+len(temp[0]):])>1:
                        w = w*0.3
                    axis_true['marriage_date'] = [x, y-h*2, x+w*3, y+h*5.5]
                    axis_dict['marriage_name'].append(([x, y-h*3.5, x+w*3, y-h*0.5], 0.8))
                    axis_dict['marriage_id'].append(([x, y+h*3, x+w*4.5, y+h*6.5], 0.8))
                    continue
            if 'marriage_id' not in axis_true:
                temp = [char for char in self._char_marriage_id if char in i[1][0]]
                if temp:
                    if len(i[1][0][i[1][0].find(temp[0])+len(temp[0]):])>1:
                        w = w*0.3
                    axis_true['marriage_id'] = [x, y-h*2, x+w*4.5, y+h*6]
                    axis_dict['marriage_name'].append(([x, y-h*8, x+w*2.5, y-h*4.5], 0.6))
                    axis_dict['marriage_date'].append(([x, y-h*3.5, x+w*2.5, y-h*0.5], 0.8))
                    continue
            if 'user_name_up' not in axis_true and i[0][0][1]<self._axis_up_down:
                temp = [char for char in self._char_user_name if char in i[1][0]]
                if temp:
                    if len(i[1][0])>2:
                        w = w*2/(len(i[1][0])+1)
                    axis_true['user_name_up'] = [x+w, y-h*2, x+w*4.5, y+h*2]
                    axis_dict['user_country_up'].append(([None, y+h, None, y+h*4], 0.8))
                    axis_dict['user_number_up'].append(([None, y+h*2, None, y+h*7], 0.6))
                    axis_dict['user_name_down'].append(([None, y+h*6, None, y+h*10], 0.4))
                    axis_dict['user_country_up'].append(([x+w, None, x+w*4.5, None], 100))
                    axis_dict['user_number_up'].append(([x+w, None, x+w*9, None], 100))
                    axis_dict['user_name_down'].append(([x+w, None, x+w*4.5, None], 100))
                    axis_dict['user_country_down'].append(([x+w, None, x+w*4.5, None], 100))
                    axis_dict['user_number_down'].append(([x+w, None, x+w*9, None], 100))
                    continue
            if 'user_country_up' not in axis_true and i[0][0][1]<self._axis_up_down:
                temp = [char for char in self._char_user_country if char in i[1][0]]
                if temp:
                    if len(i[1][0][i[1][0].find(temp[0])+len(temp[0]):])>1:
                        w = w*2/(len(i[1][0])+1)
                    axis_true['user_country_up'] = [x+w, y-h, x+w*4.5, y+h*2]
                    axis_dict['user_name_up'].append(([None, y-h*4, None, y], 0.8))
                    axis_dict['user_number_up'].append(([None, y, None, y+h*5], 0.8))
                    axis_dict['user_name_down'].append(([None, y+h*4, None, y+h*8], 0.4))
                    axis_dict['user_name_up'].append(([x+w, None, x+w*4.5, None], 100))
                    axis_dict['user_number_up'].append(([x+w, None, x+w*9, None], 100))
                    axis_dict['user_name_down'].append(([x+w, None, x+w*4.5, None], 100))
                    axis_dict['user_country_down'].append(([x+w, None, x+w*4.5, None], 100))
                    axis_dict['user_number_down'].append(([x+w, None, x+w*9, None], 100))
                    continue
            if 'user_number_up' not in axis_true and i[0][0][1]<self._axis_up_down:
                if sum([1 for j in i[1][0][-18:] if j in '0123456789xX'])==18:
                    axis_true['user_number_up'] = [x, y, max(i[0][1][0], i[0][2][0]), max(i[0][2][1], i[0][3][1])]
                    if i[1][0][:-18]:
                        axis_true['user_number_up'][0] = axis_true['user_number_up'][0]+w*len(i[1][0][:-18])/(len(i[1][0])-18+9+1.5)
                        w = (i[0][1][0]+i[0][2][0])/2-axis_true['user_number_up'][0]
                    axis_dict['user_name_up'].append(([axis_true['user_number_up'][0]-w*0.33, axis_true['user_number_up'][1]-h*6, 
                                                       axis_true['user_number_up'][0]+w*5.5/13, axis_true['user_number_up'][1]-h*2.5], 0.6))
                    axis_dict['user_country_up'].append(([axis_true['user_number_up'][0]-w*0.33, axis_true['user_number_up'][1]-h*2.25, 
                                                          axis_true['user_number_up'][0]+w*3.5/13, axis_true['user_number_up'][1]-h*0.5], 0.6))
                    axis_dict['user_sex_up'].append(([axis_true['user_number_up'][0]+w*1.2, axis_true['user_number_up'][1]-h*3, 
                                                      axis_true['user_number_up'][0]+w*1.66, axis_true['user_number_up'][1]-h*1.25], 0.6))
                    axis_dict['user_born_up'].append(([axis_true['user_number_up'][0]+w*1.5, axis_true['user_number_up'][1]-h*1.75, 
                                                       axis_true['user_number_up'][0]+w*2.3, axis_true['user_number_up'][1]-h*0.1], 0.6))
                    continue
                temp = [char for char in self._char_user_number if char in i[1][0]]
                if temp:
                    if len(i[1][0][i[1][0].find(temp[0])+len(temp[0]):])>1:
                        w = w*0.3
                    axis_true['user_number_up'] = [x+w*0.5, y-h*2, x+w*4, y+h*2]
                    axis_dict['user_name_up'].append(([None, y-h*5, None, y-h*2], 0.6))
                    axis_dict['user_country_up'].append(([None, y-h*3, None, y], 0.8))
                    axis_dict['user_name_down'].append(([None, y+h*2, None, y+h*5], 0.4))
                    axis_dict['user_name_up'].append(([x+w*0.4, None, x+w*3, None], 100))
                    axis_dict['user_country_up'].append(([x+w*0.4, None, x+w*3, None], 100))
                    axis_dict['user_name_down'].append(([x+w*0.4, None, x+w*3, None], 100))
                    axis_dict['user_country_down'].append(([x+w*0.4, None, x+w*3, None], 100))
                    axis_dict['user_number_down'].append(([x+w*0.5, None, x+w*4, None], 100))
                    continue
            if 'user_name_down' not in axis_true and i[0][0][1]>self._axis_up_down:
                temp = [char for char in self._char_user_name if char in i[1][0]]
                if temp:
                    if len(i[1][0])>2:
                        w = w*2/(len(i[1][0])+1)
                    axis_true['user_name_down'] = [x+w, y-h*1.5, x+w*4.5, y+h*2]
                    axis_dict['user_number_up'].append(([None, y-h*5, None, y-h], 0.8))
                    axis_dict['user_country_down'].append(([None, y+h, None, y+h*4], 0.8))
                    axis_dict['user_number_down'].append(([None, y+h*2, None, y+h*7], 0.4))
                    axis_dict['user_name_up'].append(([x+w, None, x+w*4.5, None], 100))
                    axis_dict['user_country_up'].append(([x+w, None, x+w*4.5, None], 100))
                    axis_dict['user_number_up'].append(([x+w, None, x+w*9, None], 100))
                    axis_dict['user_country_down'].append(([x+w, None, x+w*4.5, None], 100))
                    axis_dict['user_number_down'].append(([x+w, None, x+w*9, None], 100))
                    continue
            if 'user_country_down' not in axis_true and i[0][0][1]>self._axis_up_down:
                temp = [char for char in self._char_user_country if char in i[1][0]]
                if temp:
                    if len(i[1][0][i[1][0].find(temp[0])+len(temp[0]):])>1:
                        w = w*2/(len(i[1][0])+1)
                    axis_true['user_country_down'] = [x+w, y-h, x+w*4.5, y+h*2]
                    axis_dict['user_number_up'].append(([None, y-h*7, None, y-h*3], 0.4))
                    axis_dict['user_name_down'].append(([None, y-h*3, None, y], 0.8))
                    axis_dict['user_number_down'].append(([None, y, None, y+h*5], 0.8))
                    axis_dict['user_name_up'].append(([x+w, None, x+w*4.5, None], 100))
                    axis_dict['user_country_up'].append(([x+w, None, x+w*4.5, None], 100))
                    axis_dict['user_number_up'].append(([x+w, None, x+w*9, None], 100))
                    axis_dict['user_name_down'].append(([x+w, None, x+w*4.5, None], 100))
                    axis_dict['user_number_down'].append(([x+w, None, x+w*9, None], 100))
                    continue
            if 'user_number_down' not in axis_true and i[0][0][1]>self._axis_up_down:
                if sum([1 for j in i[1][0][-18:] if j in '0123456789xX'])==18:
                    axis_true['user_number_down'] = [x, y, max(i[0][1][0], i[0][2][0]), max(i[0][2][1], i[0][3][1])]
                    if len(i[1][0][:-18]):
                        axis_true['user_number_down'][0] = axis_true['user_number_down'][0]+w*len(i[1][0][:-18])/(len(i[1][0])-18+9+1.5)
                        w = (i[0][1][0]+i[0][2][0])/2-axis_true['user_number_down'][0]
                    axis_dict['user_name_down'].append(([axis_true['user_number_down'][0]-w*0.33, axis_true['user_number_down'][1]-h*6, 
                                                         axis_true['user_number_down'][0]+w*5.5/13, axis_true['user_number_down'][1]-h*2.5], 0.6))
                    axis_dict['user_country_down'].append(([axis_true['user_number_down'][0]-w*0.33, axis_true['user_number_down'][1]-h*2.25, 
                                                            axis_true['user_number_down'][0]+w*3.5/13, axis_true['user_number_down'][1]-h*0.5], 0.6))
                    axis_dict['user_sex_down'].append(([axis_true['user_number_down'][0]+w*1.2, axis_true['user_number_down'][1]-h*3, 
                                                        axis_true['user_number_down'][0]+w*1.65, axis_true['user_number_down'][1]-h*1.25], 0.6))
                    axis_dict['user_born_down'].append(([axis_true['user_number_down'][0]+w*1.5, axis_true['user_number_down'][1]-h*1.75, 
                                                         axis_true['user_number_down'][0]+w*2.3, axis_true['user_number_down'][1]-h*0.1], 0.6))
                    continue
                temp = [char for char in self._char_user_number if char in i[1][0]]
                if temp:
                    if len(i[1][0][i[1][0].find(temp[0])+len(temp[0]):])>1:
                        w = w*0.3
                    axis_true['user_number_down'] = [x+w*0.5, y-h*2, x+w*4, y+h*2]
                    axis_dict['user_number_up'].append(([None, y-h*8, None, y-h*5], 0.4))
                    axis_dict['user_name_down'].append(([None, y-h*5, None, y-h*2], 0.6))
                    axis_dict['user_country_down'].append(([None, y-h*3, None, y], 0.8))
                    axis_dict['user_name_up'].append(([x+w*0.4, None, x+w*3, None], 100))
                    axis_dict['user_country_up'].append(([x+w*0.4, None, x+w*3, None], 100))
                    axis_dict['user_number_up'].append(([x+w*0.5, None, x+w*4, None], 100))
                    axis_dict['user_name_down'].append(([x+w*0.4, None, x+w*3, None], 100))
                    axis_dict['user_country_down'].append(([x+w*0.4, None, x+w*3, None], 100))
                    continue
#         print(axis_true, '\n', axis_dict)
        for i in self._keys:
            if i not in axis_true:
                if axis_dict[i]:
                    try:
                        axis_true[i] = [
                            sum([j[0][0]*j[1] for j in axis_dict[i] if j[0][0] is not None])/sum([j[1] for j in axis_dict[i] if j[0][0] is not None]),
                            sum([j[0][1]*j[1] for j in axis_dict[i] if j[0][1] is not None])/sum([j[1] for j in axis_dict[i] if j[0][1] is not None]),
                            sum([j[0][2]*j[1] for j in axis_dict[i] if j[0][2] is not None])/sum([j[1] for j in axis_dict[i] if j[0][2] is not None]),
                            sum([j[0][3]*j[1] for j in axis_dict[i] if j[0][3] is not None])/sum([j[1] for j in axis_dict[i] if j[0][3] is not None])
                        ]
                    except:
                        pass
        self._axis = axis_true         
        
    def _fit_characters(self, axis, result):
        if len(result)==0:
            return 0

        axis_true = {i:tuple(axis[i]) for i in axis}
        fix_x = []
        marriage_id = ''
        for i in result[0]:
            if sum([1 for char in i[1][0] if char in '仅用于使'])>2:
                continue
            for j in ['国籍', '出生日期', '身份证件号']:
                if j in i[1][0]:
                    step_name = False
            h = (i[0][3][1]+i[0][2][1]-i[0][1][1]-i[0][0][1])/2
            w = (i[0][1][0]+i[0][2][0]-i[0][0][0]-i[0][3][0])/2
            if h==0:
                h = 1
            if w==0:
                w = 1
            x = min(i[0][0][0], i[0][3][0])
            y = min(i[0][0][1], i[0][1][1])
            if '图片模糊' in self._info.get('marriage_name', '') and 'marriage_name' in axis_true:
                h1 = min(max(i[0][3][1], i[0][2][1]), axis_true['marriage_name'][3])-max(min(i[0][0][1], i[0][1][1]), axis_true['marriage_name'][1])
                w1 = min(max(i[0][1][0], i[0][2][0]), axis_true['marriage_name'][2])-max(min(i[0][0][0], i[0][3][0]), axis_true['marriage_name'][0])            
                if h1/h>0.6 and w1/w>0.6 and sum([1 for char in i[1][0] if char in '年月日登记期0123456789'])<3:
                    if '人' in i[1][0] and len(i[1][0][i[1][0].find('人')+1:])>1:
                        self._info['marriage_name'] = i[1][0][i[1][0].find('人')+1:]
                        self._axis['marriage_name'] = [self._axis['marriage_name'][0], y]+i[0][2]
                        self._marriage_name_prob = i[1][1]
                    elif i[1][0].startswith('持') and sum([1 for j in '持证人' if j in i[1][0]])<2:
                        self._info['marriage_name'] = i[1][0][i[1][0].find('持')+1:]
                        self._axis['marriage_name'] = [self._axis['marriage_name'][0], y]+i[0][2]
                        self._marriage_name_prob = i[1][1]
                    elif len(la.text.sequence_preprocess(i[1][0]))>1 and sum([1 for j in '持证人' if j in i[1][0]])<2:
                        self._info['marriage_name'] = i[1][0]
                        self._axis['marriage_name'] = [x, y]+i[0][2]
                        fix_x.append(i[0][0][0])
                        self._marriage_name_prob = i[1][1]
                if '图片模糊' not in self._info['marriage_name']:
                    continue
            if '图片模糊' in self._info.get('marriage_date', '') and 'marriage_date' in axis_true:
                h1 = min(max(i[0][3][1], i[0][2][1]), axis_true['marriage_date'][3])-max(min(i[0][0][1], i[0][1][1]), axis_true['marriage_date'][1])
                w1 = min(max(i[0][1][0], i[0][2][0]), axis_true['marriage_date'][2])-max(min(i[0][0][0], i[0][3][0]), axis_true['marriage_date'][0])
                temp = self._analysis_date(i[1][0])
                if h1/h>0.6 and w1/w>0.6 and len(temp)==11 and temp.find('年')==4:
                    self._info['marriage_date'] = temp
                    self._axis['marriage_date'] = [x, y]+i[0][2]
                    continue
            if '图片模糊' in self._info.get('marriage_type', ''):
                if i[1][0].startswith('结婚证'):
                    self._info['marriage_type'] = '结婚证'
                elif i[1][0].startswith('离婚证'):
                    self._info['marriage_type'] = '离婚证'
            if '图片模糊' in self._info.get('marriage_id', '') and 'marriage_id' in axis_true:
                h1 = min(max(i[0][3][1], i[0][2][1]), axis_true['marriage_id'][3])-max(min(i[0][0][1], i[0][1][1]), axis_true['marriage_id'][1])
                w1 = min(max(i[0][1][0], i[0][2][0]), axis_true['marriage_id'][2])-max(min(i[0][0][0], i[0][3][0]), axis_true['marriage_id'][0])            
                temp = i[1][0]
                for n,m in [('T', 'J'), ('--', '-'), (' ', ''), ('结宇', '结字'), ('.', ''), ('（', '('), ('）', ')'),
                            ('Q', '0'), ('f', '1'), ('I', '1')]:
                    temp = temp.replace(n, m)
                for char in ['号', '婚证字', '备', '注']:
                    if char in temp:
                        temp = temp[temp.find(char)+len(char):]
                if h1/h>0.6 and w1/w>0.6 and sum([1 for char in temp if char in '年月日其他，补发此证专用章仅限用于'])<2:
                    if len(temp)>9:
                        if temp[0]=='B' and temp[1] in '0123456789':
                            temp = 'BJ'+temp[1:]
                        if len(temp.split('-')[0])==7 and temp[0] not in 'JL':
                            temp = 'J'+temp[1:]
                        elif len(temp.split('-')[0])==6:
                            temp = 'J1'+temp[1:] if temp[0]=='J' else 'J'+temp
                        elif len(temp.split('-')[0])==5:
                            temp = ('J3' if temp[0] in ['6', '7'] else 'J1')+temp
                        if temp[0] in 'BJL':
                            temp = ''.join([char for char in temp if char in 'BJL-0123456789'])
                        self._info['marriage_id'] = temp
                        self._axis['marriage_id'] = [x, y]+i[0][2]
                        continue
                    else:
                        marriage_id += temp
            if '图片模糊' in self._info.get('user_name_up', '') and 'user_name_up' in axis_true and i[0][0][1]<self._axis_up_down:
                temp = la.text.sequence_preprocess(i[1][0])
                if sum([1 for char in temp if char in '国籍中性别男女出生期身份证件号'])<2:
                    h1 = min(max(i[0][3][1], i[0][2][1]), axis_true['user_name_up'][3])-max(min(i[0][0][1], i[0][1][1]), axis_true['user_name_up'][1])
                    w1 = min(max(i[0][1][0], i[0][2][0]), axis_true['user_name_up'][2])-max(min(i[0][0][0], i[0][3][0]), axis_true['user_name_up'][0])            
                    for char in self._char_user_name+['姓', '名']:
                        if temp.startswith(char):
                            temp = temp[temp.find(char)+len(char):]
                    if h1/h>0.6 and w1/w>0.4 and len(temp)>1:
                        self._info['user_name_up'] = temp
                        if 'marriage_name' in self._info:
                            if i[1][1]>self._marriage_name_prob:
                                self._info['marriage_name'] = temp
                            else:
                                self._info['user_name_up'] = self._info['marriage_name']
                        self._axis['user_name_up'] = [self._axis['user_name_up'][0], y]+i[0][2]
                        continue
            if '图片模糊' in self._info.get('user_country_up', '') and 'user_country_up' in axis_true and i[0][0][1]<self._axis_up_down:
                temp = find_country(i[1][0])
                if temp:
                    h1 = min(max(i[0][3][1], i[0][2][1]), axis_true['user_country_up'][3])-max(min(i[0][0][1], i[0][1][1]), axis_true['user_country_up'][1])
                    w1 = min(max(i[0][1][0], i[0][2][0]), axis_true['user_country_up'][2])-max(min(i[0][0][0], i[0][3][0]), axis_true['user_country_up'][0]) 
                    if h1/h>0.6 and w1/w>0.4:
                        self._info['user_country_up'] = temp
                        self._axis['user_country_up'] = [self._axis['user_country_up'][0], y]+i[0][2]
                        continue
            if '图片模糊' in self._info.get('user_number_up', '') and self._axis_up_down-h*5<i[0][0][1]<self._axis_up_down+h*2:
                temp = find_shenfenzheng(i[1][0])
                if temp:
                    self._info['user_number_up'] = temp[0]
                    self._info['user_sex_up'] =  temp[1]
                    self._info['user_born_up'] = temp[2]
                else:
                    temp = i[1][0].replace(' ', '').replace('（', '(').replace('）', ')')
                    for char in self._char_user_number:
                        if char in temp:
                            temp = temp[temp.find(char)+len(char):]
                    if len(temp)>8:
                        if '字第' in temp and '结' not in temp:
                            self._info['user_number_up'] = temp
                        elif '护照' in temp:
                            self._info['user_number_up'] = temp
                        elif [char for char in ['香港', '澳门', '台湾'] if char in temp]:
                            self._info['user_number_up'] = temp
                if '图片模糊' not in self._info['user_number_up']:
                    continue
            if '图片模糊' in self._info.get('user_name_down', '') and 'user_name_down' in axis_true:
                temp = la.text.sequence_preprocess(i[1][0])
                if sum([1 for char in temp if char in '国籍中性别男女出生期身份证件号'])<2:
                    h1 = min(max(i[0][3][1], i[0][2][1]), axis_true['user_name_down'][3])-max(min(i[0][0][1], i[0][1][1]), axis_true['user_name_down'][1])
                    w1 = min(max(i[0][1][0], i[0][2][0]), axis_true['user_name_down'][2])-max(min(i[0][0][0], i[0][3][0]), axis_true['user_name_down'][0])            
                    for char in self._char_user_name+['姓', '名']:
                        if temp.startswith(char):
                            temp = temp[temp.find(char)+len(char):]
                    if h1/h>0.6 and w1/w>0.4 and len(temp)>1:
                        self._info['user_name_down'] = temp
                        self._axis['user_name_down'] = [self._axis['user_name_down'][0], y]+i[0][2]
                        continue
            if '图片模糊' in self._info.get('user_country_down', '') and 'user_country_down' in axis_true and i[0][0][1]>self._axis_up_down:
                temp = find_country(i[1][0])
                if temp:
                    h1 = min(max(i[0][3][1], i[0][2][1]), axis_true['user_country_down'][3])-max(min(i[0][0][1], i[0][1][1]), axis_true['user_country_down'][1])
                    w1 = min(max(i[0][1][0], i[0][2][0]), axis_true['user_country_down'][2])-max(min(i[0][0][0], i[0][3][0]), axis_true['user_country_down'][0]) 
                    if h1/h>0.6 and w1/w>0.4:
                        self._info['user_country_down'] = temp
                        self._axis['user_country_down'] = [self._axis['user_country_down'][0], y]+i[0][2]
                        continue
            if '图片模糊' in self._info.get('user_number_down', '') and i[0][0][1]>self._axis_up_down:
                temp = find_shenfenzheng(i[1][0])
                if temp:
                    self._info['user_number_down'] = temp[0]
                    self._info['user_sex_down'] =  temp[1]
                    self._info['user_born_down'] = temp[2]
                else:
                    temp = i[1][0].replace(' ', '').replace('（', '(').replace('）', ')')
                    for char in self._char_user_number:
                        if char in temp:
                            temp = temp[temp.find(char)+len(char):]
                    if len(temp)>8:
                        if '字第' in temp and '结' not in temp:
                            self._info['user_number_down'] = temp
                        elif '护照' in temp:
                            self._info['user_number_down'] = temp
                        elif [char for char in ['香港', '澳门', '台湾'] if char in temp]:
                            self._info['user_number_down'] = temp
                if '图片模糊' not in self._info['user_number_down']:
                    continue

        if '图片模糊' in self._info.get('marriage_id', '') and marriage_id:
            self._info['marriage_id'] = marriage_id
        if 'marriage_type' in self._info:
            if [1 for i in ['J', '结字', '结补字', '结-'] if i in self._info.get('marriage_id', '')]:
                self._info['marriage_type'] = '结婚证'
            elif [1 for i in ['L', '离字'] if i in self._info.get('marriage_id', '')]:
                self._info['marriage_type'] = '离婚证'
            elif '图片模糊' in self._info['marriage_type']:
                self._info['marriage_type'] = '结婚证'
        if '图片模糊' in self._info.get('user_name_up', '') and '图片模糊' not in self._info.get('marriage_name', ''):
            self._info['user_name_up'] = self._info['marriage_name']
        if '图片模糊' in self._info.get('user_country_up', ''):
            self._info['user_country_up'] = '中国'
        if '图片模糊' in self._info.get('user_country_down', ''):
            self._info['user_country_down'] = '中国'
        if '图片模糊' in self._info.get('user_sex_down', '') and '图片模糊' in self._info.get('user_sex_up', ''):
            sex = [j for i in self._result[0] for j in ['男', '女'] if j in i[1][0]]
            self._info['user_sex_down'] = '男' if len(sex)<2 else sex[-1]
            self._info['user_sex_up'] = '女' if len(sex)<2 else sex[-2]
        elif '图片模糊' in self._info.get('user_sex_down', '') and '图片模糊' not in self._info.get('user_sex_up', ''):
            self._info['user_sex_down'] = '女' if self._info['user_sex_up']=='男' else '男'
        elif '图片模糊' not in self._info.get('user_sex_down', '') and '图片模糊' in self._info.get('user_sex_up', ''):
            self._info['user_sex_up'] = '女' if self._info['user_sex_down']=='男' else '男'
        if '图片模糊' in self._info.get('user_born_down', '') or '图片模糊' in self._info.get('user_born_up', ''):
            date = []
            for i in self._result[0]:
                h = (i[0][3][1]+i[0][2][1]-i[0][1][1]-i[0][0][1])/2
                temp = self._analysis_date(i[1][0])
                if temp.find('年')==4 and '月' in temp and i[0][0][1]>self._axis_up_down-h*5:
                    date.append(temp)
            date = [i for i in date if i!=self._info.get('marriage_date', '')]
            if len(date)==1:
                if '图片模糊' in self._info.get('user_born_down', '') and '图片模糊' in self._info.get('user_born_up', ''):
                    self._info['user_born_up'] = date[0]
                    self._info['user_born_down'] = date[0]
                elif '图片模糊' in self._info.get('user_born_down', '') or '图片模糊' not in self._info.get('user_born_up', ''):
                    self._info['user_born_down'] = date[0]
                elif '图片模糊' not in self._info.get('user_born_down', '') or '图片模糊' in self._info.get('user_born_up', ''):
                    self._info['user_born_up'] = date[0]
            elif len(date)==2:
                if '图片模糊' in self._info.get('user_born_down', '') and '图片模糊' in self._info.get('user_born_up', ''):
                    self._info['user_born_up'] = date[0]
                    self._info['user_born_down'] = date[1]
                elif '图片模糊' in self._info.get('user_born_down', '') or '图片模糊' not in self._info.get('user_born_up', ''):
                    self._info['user_born_down'] = date[1]
                elif '图片模糊' not in self._info.get('user_born_down', '') or '图片模糊' in self._info.get('user_born_up', ''):
                    self._info['user_born_up'] = date[0]
        
#         if self._show_axis:
#             try:
#                 if len(fix_x)>0:
#                     fix_x = sum(fix_x)/len(fix_x)
#                     self._axis['marriage_name'][0] = fix_x
#                     self._axis['marriage_date'][0] = fix_x
#                     self._axis['marriage_id'][0] = fix_x

#                 for i in result[0]:
#                     h = (i[0][3][1]+i[0][2][1]-i[0][1][1]-i[0][0][1])/2
#                     w = (i[0][1][0]+i[0][2][0]-i[0][0][0]-i[0][3][0])/2
#                     x = min(i[0][0][0], i[0][3][0])
#                     y = min(i[0][0][1], i[0][1][1])
#                     if self._info['user_sex_up'] in i[1][0]:
#                         h1 = min(max(i[0][3][1], i[0][2][1]), axis_true['user_sex_up'][3])-max(min(i[0][0][1], i[0][1][1]), axis_true['user_sex_up'][1])
#                         w1 = min(max(i[0][1][0], i[0][2][0]), axis_true['user_sex_up'][2])-max(min(i[0][0][0], i[0][3][0]), axis_true['user_sex_up'][0])            
#                         if h1/h>0.2 and w1/w>0.2:
#                             self._axis['user_sex_up'] = [self._axis['user_sex_up'][0], y]+i[0][2]
#                             continue
#                     if self._info['user_born_up'] in i[1][0]:
#                         self._axis['user_born_up'] = [self._axis['user_born_up'][0], y]+i[0][2]
#                         continue
#                     if self._info['user_sex_down'] in i[1][0]:
#                         h1 = min(max(i[0][3][1], i[0][2][1]), axis_true['user_sex_down'][3])-max(min(i[0][0][1], i[0][1][1]), axis_true['user_sex_down'][1])
#                         w1 = min(max(i[0][1][0], i[0][2][0]), axis_true['user_sex_down'][2])-max(min(i[0][0][0], i[0][3][0]), axis_true['user_sex_down'][0])            
#                         if h1/h>0.2 and w1/w>0.2:
#                             self._axis['user_sex_down'] = [self._axis['user_sex_down'][0], y]+i[0][2]
#                             continue
#                     if self._info['user_born_down'] in i[1][0]:
#                         self._axis['user_born_down'] = [self._axis['user_born_down'][0], y]+i[0][2]
#             except:
#                 pass
    
        for i in self._axis:
            self._axis[i] = [int(max(0, j)) for j in self._axis[i]]
    
    def _analysis_date(self, data):
        temp = data.replace('日月', '11月').replace('.', '-').replace('/', '-')
        temp = ''.join([char for char in temp if char in '0123456789年月日-'])
        if temp.find('-')>0:
            temp = temp[:temp.find('-')]+'年'+temp[temp.find('-')+1:]
        if temp.find('-')>0:
            temp = temp[:temp.find('-')]+'月'+temp[temp.find('-')+1:]
        temp = (temp+'日').replace('日日', '日').replace('-', '')
        if len(temp)>6 and '年' in temp and '月' in temp:
            if temp[0] not in '0123456789':
                temp = temp[1:]
            if temp[0]=='0' and temp.find('年')==3:
                temp = '2'+temp
            elif temp[0]=='9' and temp.find('年')==3:
                temp = '1'+temp
            elif temp[0] in '01' and temp.find('年')==2:
                temp = '20'+temp
            elif temp[0] in '456789' and temp.find('年')==2:
                temp = '19'+temp
            if temp.find('月')-temp.find('年')==2:
                temp = temp[:temp.find('年')+1]+'0'+temp[temp.find('年')+1:]
            if temp.find('日')-temp.find('月')==2:
                temp = temp[:temp.find('月')+1]+'0'+temp[temp.find('月')+1:]
        else:
            temp = ''
        return temp
    
    def draw_mask(self):
        image = self._image.copy()
        try:
            t = [la.image.box_convert(self._axis[i], 'xyxy', 'axis') for i in self._axis if i in self._keys]
            if len(t)>0:
                image = la.image.draw_box(image, t, width=2)
        except:
            pass
        return image
    
    def check_env(self):
        env = la.utils.pip.freeze('paddleocr')['paddleocr']
        if env>='2.6.1.3':
            return 'Environment check ok.'
        else:
            return f"Now environment dependent paddleocr>='2.6.1.3', local env paddleocr='{env}'"
        
    def metrics(self, data, image_root, name_list=None, debug=False, test_sample_nums=None):
        if la.gfile.isfile(data):
            with open(data) as f:
                data = f.read().replace('\n', '').replace('}', '}\n').strip().split('\n')
            data = [eval(i) for i in data]
        if name_list is None:
            name_list = ['marriage_name', 'marriage_date', 'marriage_id', 'marriage_type', 
                         'user_name_up', 'user_sex_up', 'user_country_up', 'user_born_up', 'user_number_up', 
                         'user_name_down', 'user_sex_down', 'user_country_down', 'user_born_down', 'user_number_down']

        score_a = {i:0 for i in name_list}
        score_b = {i:0 for i in name_list}
        time_list = []
        error_list = []
        nums = len(data) if test_sample_nums is None else test_sample_nums
        for i in data[:nums]:
            error = {'image':i.pop('image')}
            try:
                time_start = time.time()
                t = self.predict(la.gfile.path_join(image_root, error['image']))['data']
                time_list.append({'image':error['image'], 'time':time.time()-time_start})
                if isinstance(t, dict):
                    for j in name_list:
                        if j in i:
                            if j in t:
                                if t[j]==i[j]:
                                    score_a[j] +=1
                                else:
                                    error[j] = {'pred':t[j], 'label':i[j]}
                else:
                    error['error'] = t
            except:
                error['error'] = 'program error'
            for j in name_list:
                if j in i:
                    score_b[j] += 1
            if len(error)>1:
                error_list.append(error)

        score = {f'{i}_acc':score_a[i]/score_b[i] for i in score_a if score_b[i]>0}
        score['totalmean_acc'] = sum([score_a[i] for i in score_a])/max(sum([score_b[i] for i in score_b]), 0.0000001)
        score = {i:round(score[i], 4) for i in score}
        score['test_sample_nums'] = nums
        temp = [i['time'] for i in time_list]
        score['test_sample_time'] = {'min':f'{min(temp):.3}s', 'mean':f'{sum(temp)/len(temp):.3}s', 'max':f'{max(temp):.3}s'}
        if debug:
            score['detailed'] = {i:f'{score_a[i]}/{score_b[i]}' for i in score_a}
            score['error'] = error_list
            score['time'] = time_list
        return score

country_list = ['中国', '美国', '英国', '日本', '巴西', '阿根廷', '埃及', '埃塞俄比亚', '澳大利亚', '巴拉圭', '巴拿马', '白俄罗斯', '比利时', '秘鲁', '波兰', '玻利维亚', '丹麦', '多米尼加', '厄瓜多尔', '法国', '菲律宾', '哥伦比亚', '哥斯达黎加', '古巴', '海地', '荷兰', '洪都拉斯', '加拿大', '黎巴嫩', '利比里亚', '卢森堡', '墨西哥', '尼加拉瓜', '挪威', '萨尔瓦多', '沙特阿拉伯', '俄罗斯', '塞维利亚', '土耳其', '危地马拉', '委内瑞拉', '乌克兰', '乌拉圭', '希腊', '新西兰', '叙利亚', '伊拉克', '伊朗', '印度', '智利', '阿富汗', '冰岛', '泰国', '瑞典', '巴基斯坦', '也门', '缅甸', '以色列', '印度尼西亚', '阿尔巴尼亚', '奥地利', '保加利亚', '柬埔寨', '锡兰', '芬兰', '匈牙利', '爱尔兰', '意大利', '约旦', '老挝', '利比亚', '尼泊尔', '葡萄牙', '罗马尼亚', '西班牙', '摩洛哥', '苏丹', '突尼斯', '加纳', '马来西亚', '几内亚', '喀麦隆', '中非', '乍得', '刚果', '塞浦路斯', '达荷美', '加蓬', '象牙海岸', '马达加斯加', '马里', '尼日尔', '尼日利亚', '塞内加尔', '索马里', '多哥', '上沃尔特', '里塔尼亚', '蒙古', '塞拉利昂', '坦噶尼喀', '阿尔及利亚', '布隆迪', '牙买加', '卢旺达', '特立尼达和多巴哥', '乌干达', '肯尼亚', '科威特', '桑给巴尔', '马拉维', '马耳他', '赞比亚', '冈比亚', '马尔代夫群岛', '新加坡', '巴巴多斯', '博茨瓦纳', '圭亚那', '莱索托', '也门', '赤道几内亚', '毛里求斯', '斯威士兰', '斐济', '巴林', '不丹', '阿曼', '卡塔尔', '阿拉伯联合酋长国', '巴哈马', '德国', '孟加拉国', '格林纳达', '几内亚比绍', '佛得角', '科摩罗', '莫桑比克', '巴布亚新几内亚', '圣多美和普林西比', '苏里南', '安哥拉', '萨摩亚', '塞舌尔', '吉布提', '越南', '多米尼加', '所罗门群岛', '圣卢西亚', '圣文森特和格林纳丁斯', '津巴布韦', '安提瓜和巴布达', '伯利兹', '瓦努阿图', '圣克里斯托弗和尼维斯', '文莱', '列支敦士登', '纳米比亚', '朝鲜', '爱沙尼亚', '拉脱维亚', '立陶宛', '马绍尔群岛', '密克罗尼西亚', '韩国', '亚美尼亚', '阿塞拜疆', '波黑', '克罗地亚', '格鲁吉亚', '哈萨克斯坦', '吉尔吉斯斯坦', '摩尔多瓦', '圣马力诺', '斯洛文尼亚', '塔吉克斯坦', '土库曼斯坦', '乌兹别克斯坦', '安道尔', '捷克', '厄立特里亚', '摩纳哥', '斯洛伐克', '马其顿', '帕劳', '基里巴斯', '瑙鲁', '汤加', '图瓦卢', '瑞士', '东帝汶', '黑山共和国', '南苏丹']

def find_country(data):
    t = data.replace('中华人民共和国', '中国')
    for i in country_list:
        if i in t:
            return i
    return ''

def find_shenfenzheng(data):
    temp = data.replace('x', 'X')
    year = time.localtime(time.time()).tm_year
    if sum([1 for i in temp if i in '0123456789X'])<15:
        return []
    for i in range(0,len(temp)-14):
        t = ''.join([j for j in temp[i:i+18] if j in '0123456789X'])
        if 'X' in t:
            if not (t.count('X')==1 and t.endswith('X')):
                continue
        if len(t)==17:
            t = ('3' if int(t[0])>5 else '1')+t
        if len(t)==18:
            if 1900<int(t[6:10])<year and int(t[10:12])<13 and int(t[12:14])<32:
                return [t, '男' if int(t[16])%2 else '女', f"{t[6:10]}年{t[10:12]}月{t[12:14]}日"]
        elif len(t)==15 and 'X' not in t:
            if int(t[8:10])<13 and int(t[10:12])<32:
                return [t, '男' if int(t[-1])%2 else '女', f"19{t[6:8]}年{t[8:10]}月{t[10:12]}日"]
    return []