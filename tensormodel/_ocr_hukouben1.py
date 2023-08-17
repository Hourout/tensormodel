from collections import defaultdict

import cv2
import paddleocr
import linora as la

__all__ = ['OCRHuKouBen']


class OCRHuKouBen():
    def __init__(self, ocr=None):
        self.ocr = paddleocr.PaddleOCR(show_log=False) if ocr is None else ocr
        self._keys = []
        self._char_household_type = ['农业家庭户', '非农业家庭户', '非农业家庭户口', '非农业集体', '非农业集体户口',
                                     '城市户口', '家庭户', '家庭户口']
        self._char_household_name = ['户主姓名', '户生姓名']
        self._char_household_id = ['户号']
        self._char_household_address = ['住址']
        self._char_register_name = ['姓名']
#         self._char_number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        self._aug_image = True
        
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

    def metrics(self, data, image_root, name_list=None, debug=False):
        if la.gfile.isfile(data):
            with open(data) as f:
                data = f.read().replace('\n', '').replace('}', '}\n').strip().split('\n')
            data = [eval(i) for i in data]
        if name_list is None:
            name_list = ['household_type', 'household_name', 'household_id', 'household_address',
                         'register_name', 'register_previous_name', 'register_relation', 'register_sex', 
                         'register_born', 'register_number', 'register_education', 'register_service_office', 
                         'register_marriage', 'register_military', 'register_career','register_city', 
                         'register_address']

        score_a = {i:0 for i in name_list}
        score_b = {i:0.0000001 for i in name_list}
        error_list = []
        for i in data:
            error = {'image':i.pop('image')}
            try:
                t = self.predict(la.gfile.path_join(image_root, error['image']))['data']
                if isinstance(t, dict):
                    for j in name_list:
                        if j in i:
                            if j in t:
                                if t[j]==i[j]:
                                    score_a[j] +=1
                                else:
                                    error[j] = {'pred':t[j], 'label':i[j]}
                            score_b[j] += 1
            except:
                for j in name_list:
                    if j in i:
                        score_b[j] += 1
                error['error'] = 'program error'
            if len(error)>1:
                error_list.append(error)
        
        score = {f'{i}_acc':score_a[i]/max(score_b[i], 0.0000001) for i in score_a}
        score['totalmean_acc'] = sum([score_a[i] for i in score_a])/max(sum([score_b[i] for i in score_b]), 0.0000001)
        score = {i:round(score[i], 4) for i in score}
        score['test_sample_nums'] = len(data)
        if debug:
            score['error'] = error_list
        return score
    
    def predict(self, image, axis=False, ocr_result=None):
        self._axis = None
        self._show_axis = axis
        self._error = 'ok'
        self._angle = -1
        self._mode = ''
        self._image_str = None
        self._aug_image = False
        
        if ocr_result is not None:
            self._result = ocr_result
            self._fit_direction(image, use_ocr_result=True)
            ax = self._fit_axis()
            self._fit_characters(ax)
        else:
            if isinstance(image, str):
                self._image_str = image
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
            if sum([1 for i in self._info if '图片模糊' in self._info[i]])>0:
                self._aug_image = True
                self._temp_info = self._info.copy()
                if self._show_axis:
                    self._temp_axis = self._axis.copy()
                if self._image_str is not None and self._angle==0:
                    self._fit_direction(self._image_str)
                else:
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
        if use_ocr_result:
            self._angle = 0
        elif self._angle!=-1:
            if self._image_str is not None and self._angle==0:
                self._result = self.ocr.ocr(image, cls=False)
            else:
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
                    self._keys = ['household_type', 'household_name', 'household_id', 'household_address']
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
                                  'register_height', 'register_blood',  'register_career', 'register_city', 
                                  'register_address', 'register_marriage', 'register_military']
                    break
        
        self._info = {}
        if self._mode == 'shouye':
            self._info['household_type'] = '图片模糊:未识别出户别'
            self._info['household_name'] = '图片模糊:未识别出户主'
            self._info['household_id'] = '图片模糊:未识别出户号'
            self._info['household_address'] = '图片模糊:未识别出住址'
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
    
    def _fit_axis_shouye(self):
        if len(self._result)==0:
            return 0
        
        axis_true = defaultdict(list)
        axis_dict = defaultdict(list)

        jitihu = False
        axis_address = {}
        for i in self._result[0]:
            h = (i[0][3][1]+i[0][2][1]-i[0][1][1]-i[0][0][1])/2
            w = (i[0][1][0]+i[0][2][0]-i[0][0][0]-i[0][3][0])/2
            if h==0:
                h = 1
            if w==0:
                w = 1
            x = min(i[0][0][0], i[0][3][0])
            y = min(i[0][0][1], i[0][1][1])
            if 'household_type' not in axis_true:
                for char in self._char_household_type:
                    if char==i[1][0]:
                        w = w*(1.2 if len(char)>4 else 1.5)
                        axis_true['household_type'] = [x, y]+i[0][2]
                        axis_dict['household_name'].append(([x+w*1.6, y-h*0.5, x+w*3.5, y+h*1.5], 0.6))
                        axis_dict['household_id'].append(([x-w*0.2, y+h*1.5, x+w*0.8, y+h*3.5], 0.6))
                        axis_dict['household_address'].append(([x+w*1.5, y+h*1.5, x+w*4, y+h*4], 0.6))
                        break
                if 'household_type' in axis_true:
                    continue
            if 'household_name' not in axis_true:
                if '集体户名称' in i[1][0]:
                    if len(i[1][0])>5:
                        w = w*(len(char)+0.5)/len(i[1][0])
                    axis_true['household_name'] = [x+w*1.1, y-h*0.5, x+w*3, y+h*1.5]
                    axis_dict['household_type'].append(([x-w*1.75, y-h*0.5, x-w*0.25, y+h*1.5], 0.8))
                    axis_dict['household_address'].append(([x-w*1.75, y+h*1.5, x+w*3, y+h*3.5], 0.8))
                    jitihu = True
                    continue
                for char in self._char_household_name:
                    if char in i[1][0]:
                        if char in i[1][0]:
                            if len(i[1][0])>4:
                                w = w*(len(char)+0.5)/len(i[1][0])
                            axis_true['household_name'] = [x+w, y-h*0.5, x+w*4, y+h*2]
                            axis_dict['household_type'].append(([x-w*2, y-h*0.5, x-w*0.25, y+h*1.5], 0.8))
                            axis_dict['household_id'].append(([x-w*2, y+h*1.5, x-w*0.75, y+h*3.5], 0.8))
                            axis_dict['household_address'].append(([x+w*0.5, y+h*1.5, x+w*5, y+h*4], 0.8))
                            break
                if 'household_name' in axis_true:
                    continue
            if i[1][0] in ['住'] and '住' not in axis_address:
                axis_address['住'] = [x, y]+i[0][2]
                continue
            if i[1][0] in ['址'] and '址' not in axis_address:
                axis_address['址'] = [x, y]+i[0][2]
                continue
                
        if '住' in axis_address and '址' in axis_address:
            axis = axis_address['住'][:2]+axis_address['址'][-2:]
            h = axis[3]-axis[1]
            w = axis[2]-axis[0]
            if h==0:
                h = 1
            if w==0:
                w = 1
            x = axis[0]
            y = axis[1]
            if jitihu:
                axis_dict['household_type'].append(([x+w*1.1, y-h*3, x+w*3, y-h*0.5], 0.8))
                axis_dict['household_name'].append(([x+w*4.75, y-h*3, x+w*8, y-h*0.5], 0.8))
                axis_dict['household_address'].append(([x+w, y-h*0.5, x+w*8, y+h*1.5], 0.8))
            else:
                axis_dict['household_type'].append(([x-w*2, y-h*2.5, x, y-h*0.5], 0.8))
                axis_dict['household_name'].append(([x+w*1.75, y-h*2.5, x+w*6, y-h*0.5], 0.8))
                axis_dict['household_id'].append(([x-w*2, y-h*0.5, x-w*0.5, y+h*1.5], 0.8))
                axis_dict['household_address'].append(([x+w, y-h*0.5, x+w*7, y+h*2.5], 0.8))
            
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
        fix_x = []
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
                if sum([1 for char in self._char_household_name if char in i[1][0]])>0:
                    for char in self._char_household_name:
                        if char in i[1][0] and len(i[1][0][i[1][0].find(char)+len(char):])>1:
                            self._info['household_name'] = i[1][0][i[1][0].find(char)+len(char):].strip()
                            self._axis['household_name'] = [self._axis['household_name'][0], i[0][0][1]]+i[0][2]
                            break
                    continue
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
#             if '图片模糊' in self._info['household_date']:
#                 if '年' in i[1][0] and '月' in i[1][0] and i[1][0].endswith('日'):
#                     self._info['household_date'] = i[1][0][max(0, i[1][0].find('年')-4):]
#                     continue

        if '图片模糊' in self._info['household_address'] and address!='':
            self._info['household_address'] = address
        
        if self._info['household_id'].endswith('住'):
            self._info['household_id'] = self._info['household_id'][:-1]
        if '图片模糊' in self._info['household_id'] and self._aug_image:
            self._info['household_id'] = ''
        if '图片模糊' in self._info['household_type'] and self._aug_image:
            self._info['household_type'] = '非农业家庭户口'
        else:
            types = self._info['household_type']
            if types.endswith('户'):
                types = types+'口'
            if types.endswith('集体'):
                types = types+'户口'
            if types[1] in ['农','城','家','居']:
                types = '非'+types[1:]
            types = types.replace('衣', '农')
            if not types.endswith('户口'):
                if '家庭' in types:
                    types = types[:types.find('家庭')+2]+'户口'
                elif '农业' in types and types.endswith('体'):
                    types = types[:types.find('农业')+2]+'集体户口'
            self._info['household_type'] = types
        try:
            if len(fix_x)>0:
                fix_x = sum(fix_x)/len(fix_x)
                self._axis['household_type'][0] = fix_x
                self._axis['household_id'][0] = fix_x
        except:
            pass
        
        for i in self._axis:
            self._axis[i] = [int(max(0, j)) for j in self._axis[i]]
    
    def _fit_axis_neirong(self):
        if len(self._result)==0:
            return 0
        fix_x = []
        axis_true = defaultdict(list)
        axis_dict = defaultdict(list)
        
        axis_fix = ['宗教信仰', '身高', '血型', '婚姻状况']
        axis_fix_h = [i[0][3][1]-i[0][0][1] for i in self._result[0] for j in axis_fix if j in i[1][0]]
        axis_fix_h = sum(axis_fix_h)/len(axis_fix_h) if axis_fix_h else 0
        
        h_abs = 0
        y_abs = 0
        for i in self._result[0]:
            if '常住人口登记卡'==i[1][0]:
                h_abs = (i[0][3][1]+i[0][2][1]-i[0][1][1]-i[0][0][1])/2*0.65
                y_abs = i[0][2][1]
                break
        
        register_name = {}
        register_sex = {}
        register_nation = {}
        register_nativeplace = {}
        for i in self._result[0]:
            h = (i[0][3][1]+i[0][2][1]-i[0][1][1]-i[0][0][1])/2
            w = (i[0][1][0]+i[0][2][0]-i[0][0][0]-i[0][3][0])/2
#             print(h, h_abs, i[1][0])
            if h==0:
                h = 1
            if w==0:
                w = 1
            x = min(i[0][0][0], i[0][3][0])
            y = min(i[0][0][1], i[0][1][1])
            
            if '常住人口登记卡'==i[1][0]:
                axis_dict['register_name'].append(([x-w*0.3, y+h*1.25, x+w*0.6, y+h*2.85], 0.8, 1))
                axis_dict['register_relation'].append(([x+w*1.15, y+h*1.25, x+w*1.75, y+h*2.85], 0.8, 1))
                axis_dict['register_previous_name'].append(([x-w*0.3, y+h*2.85, x+w*0.6, y+h*4.25], 0.8, 1))
                axis_dict['register_sex'].append(([x+w*1.15, y+h*2.85, x+w*1.75, y+h*4.25], 0.4, 1))
                axis_dict['register_birthplace'].append(([x-w*0.3, y+h*4.25, x+w*0.6, y+h*5.75], 0.6, 1))
                axis_dict['register_nation'].append(([x+w*1.15, y+h*4.25, x+w*1.75, y+h*5.75], 0.4, 1))
                continue
            
            if y_abs>i[0][2][1]:
                continue
            if h>h_abs and h_abs>0:
                y += (h-h_abs)/2
                h = h_abs
            h1, h2, h3, h4, h5, h6, h7 = -4.5*h, -2.5*h, -0.5*h, 1.5*h, 3.5*h, 5.5*h, 7.5*h
            
            if i[1][0] in ['姓'] and '姓' not in register_name:
                register_name['姓'] = [x, y]+i[0][2]
                continue
            if i[1][0] in ['名'] and '名' not in register_name:
                register_name['名'] = [x, y]+i[0][2]
                continue
            if i[1][0] in ['性'] and '性' not in register_sex:
                register_sex['性'] = [x, y]+i[0][2]
                continue
            if i[1][0] in ['别'] and '别' not in register_sex:
                register_sex['别'] = [x, y]+i[0][2]
                continue
            if i[1][0] in ['民'] and '民' not in register_nation:
                register_nation['民'] = [x, y]+i[0][2]
                continue
            if i[1][0] in ['族'] and '族' not in register_nation:
                register_nation['族'] = [x, y]+i[0][2]
                continue
            if i[1][0] in ['籍'] and '籍' not in register_nativeplace:
                register_nativeplace['籍'] = [x, y]+i[0][2]
                continue
            if i[1][0] in ['贯'] and '贯' not in register_nativeplace:
                register_nativeplace['贯'] = [x, y]+i[0][2]
                continue
            
            if 'register_previous_name' not in axis_true and i[1][0] in ['曾用名']:
                w1, w2, w3 = 2.75*w, 3.75*w, 5.25*w
                axis_true['register_previous_name'] = [x+w, y+h3, x+w1, y+h4]
                axis_dict['register_name'].append(([x+w, y+h2, x+w1, y+h3], 0.8, 2))
                axis_dict['register_relation'].append(([x+w2, y+h2, x+w3, y+h3], 0.6, 2))
                axis_dict['register_sex'].append(([x+w2, y+h3, x+w3, y+h4], 0.8, 2))
                axis_dict['register_birthplace'].append(([x+w, y+h4, x+w1, y+h5], 0.8, 2))
                axis_dict['register_nation'].append(([x+w2, y+h4, x+w3, y+h5], 0.6, 2))
                axis_dict['register_nativeplace'].append(([x+w, y+h5, x+w1, y+h6], 0.6, 2))
                axis_dict['register_born'].append(([x+w2, y+h5, x+w3, y+h6], 0.4, 2))
                continue
            if 'register_relation' not in axis_true and i[1][0] in ['户主或与', '户主关系']:
                y += h/(2 if i[1][0] in ['户主或与'] else -2)
                w1, w2 = -2.25*w, 2.75*w
                axis_true['register_relation'] = [x+w, y+h3, x+w2, y+h4]
                axis_dict['register_name'].append(([x+w1, y+h3, x, y+h4], 0.8, 3))
                axis_dict['register_previous_name'].append(([x+w1, y+h4, x, y+h5], 0.6, 3))
                axis_dict['register_sex'].append(([x+w, y+h4, x+w2, y+h5], 0.8, 3))
                axis_dict['register_birthplace'].append(([x+w1, y+h5, x, y+h6], 0.4, 3))
                axis_dict['register_nation'].append(([x+w, y+h5, x+w2, y+h6], 0.6, 3))
                continue
            if 'register_birthplace' not in axis_true and i[1][0] in ['出生地']:
                w1, w2, w3 = 2.75*w, 3.75*w, 5.25*w
                axis_true['register_birthplace'] = [x+w, y+h3, x+w1, y+h4]
                axis_dict['register_name'].append(([x+w, y+h1, x+w1, y+h2], 0.6, 4))
                axis_dict['register_relation'].append(([x+w2, y+h1, x+w3, y+h2], 0.4, 4))
                axis_dict['register_previous_name'].append(([x+w, y+h2, x+w1, y+h3], 0.8, 4))
                axis_dict['register_sex'].append(([x+w2, y+h2, x+w3, y+h3], 0.6, 4))
                axis_dict['register_nation'].append(([x+w2, y+h3, x+w3, y+h4], 0.8, 4))
                axis_dict['register_nativeplace'].append(([x+w, y+h4, x+w1, y+h5], 0.8, 4))
                axis_dict['register_born'].append(([x+w2, y+h4, x+w3, y+h5], 0.6, 4))
                axis_dict['register_belief'].append(([x+w*4.25, y+h5, x+w3, y+h6], 0.4, 4))
                continue
            if 'register_born' not in axis_true and i[1][0] in ['出生日期']:
                w1, w2 = -2.25*w, 2.75*w
                axis_true['register_born'] = [x+w, y+h3, x+w2, y+h4]
                axis_dict['register_previous_name'].append(([x+w1, y+h1, x, y+h2], 0.4, 5))
                axis_dict['register_sex'].append(([x+w, y+h1, x+w2, y+h2], 0.6, 5))
                axis_dict['register_birthplace'].append(([x+w1, y+h2, x, y+h3], 0.6, 5))
                axis_dict['register_nation'].append(([x+w, y+h2, x+w2, y+h3], 0.8, 5))
                axis_dict['register_nativeplace'].append(([x+w1, y+h3, x, y+h4], 0.8, 5))
                axis_dict['register_belief'].append(([x+w1, y+h2, x, y+h3], 0.8, 5))
                axis_dict['register_number'].append(([x+w1, y+h5, x, y+h6], 0.4, 5))
                axis_dict['register_height'].append(([x+w*0.5, y+h5, x+w*1.5, y+h6], 0.6, 5))
                axis_dict['register_blood'].append(([x+w*2, y+h5, x+w2, y+h6], 0.6, 5))
                continue
            if 'register_belief' not in axis_true and i[1][0] in ['宗教信仰']:
                w1, w2, w3 = -4*w, -1*w, 2.85*w
                axis_true['register_belief'] = [x+w, y+h3, x+w3, y+h4]
                axis_dict['register_birthplace'].append(([x+w1, y+h1, x+w2, y+h2], 0.4, 6))
                axis_dict['register_nation'].append(([x+w*0.5, y+h1, x+w3, y+h2], 0.6, 6))
                axis_dict['register_nativeplace'].append(([x+w1, y+h2, x+w2, y+h3], 0.6, 6))
                axis_dict['register_born'].append(([x+w*0.5, y+h2, x+w3, y+h3], 0.8, 6))
                axis_dict['register_number'].append(([x+w1, y+h4, x+w2, y+h5], 0.6, 6))
                axis_dict['register_height'].append(([x, y+h4, x+w, y+h5], 0.8, 6))
                axis_dict['register_blood'].append(([x+w*2, y+h4, x+w3, y+h5], 0.8, 6))
                axis_dict['register_education'].append(([x+w1, y+h5, x+w*-2, y+h6], 0.4, 6))
                axis_dict['register_marriage'].append(([x+w2, y+h5, x, y+h6], 0.6, 6))
                axis_dict['register_military'].append(([x+w, y+h5, x+w3, y+h6], 0.6, 6))
                continue
            if 'register_height' not in axis_true and i[1][0] in ['身高']:
                w1, w2, w3 = -4.75*w, 3*w, 5.75*w
                axis_true['register_height'] = [x+w, y+h3, x+w2, y+h4]
                axis_dict['register_nativeplace'].append(([x+w1, y+h1, x, y+h2], 0.4, 7))
                axis_dict['register_born'].append(([x+w*2.25, y+h1, x+w3, y+h2], 0.6, 7))
                axis_dict['register_number'].append(([x+w1, y+h3, x, y+h4], 0.8, 7))
                axis_dict['register_belief'].append(([x+w2, y+h2, x+w3, y+h3], 0.8, 7))
                axis_dict['register_blood'].append(([x+w*4.5, y+h3, x+w3, y+h4], 0.8, 7))
                axis_dict['register_education'].append(([x+w1, y+h4, x+w*-2, y+h5], 0.6, 7))
                axis_dict['register_marriage'].append(([x, y+h4, x+w, y+h5], 0.8, 7))
                axis_dict['register_military'].append(([x+w2, y+h4, x+w3, y+h5], 0.8, 7))
                axis_dict['register_service_office'].append(([x+w1, y+h5, x+w, y+h6], 0.4, 7))
                axis_dict['register_career'].append(([x+w2, y+h5, x+w3, y+h6], 0.6, 7))
                continue
            if 'register_blood' not in axis_true and i[1][0] in ['血型']:
                w1, w2, w3 = -7.75*w, -3*w, 2.5*w
                axis_true['register_blood'] = [x+w, y+h3, x+w3, y+h4]
                axis_dict['register_nativeplace'].append(([x+w1, y+h1, x+w2, y+h2], 0.4, 8))
                axis_dict['register_born'].append(([x-w, y+h1, x+w3, y+h2], 0.6, 8))
                axis_dict['register_belief'].append(([x, y+h2, x+w3, y+h3], 0.8, 8))
                axis_dict['register_number'].append(([x+w1, y+h3, x+w2, y+h4], 0.6, 8))
                axis_dict['register_height'].append(([x+w*-2, y+h3, x, y+h4], 0.8, 8))
                axis_dict['register_education'].append(([x+w1, y+h4, x+w*-5, y+h5], 0.6, 8))
                axis_dict['register_marriage'].append(([x+w2, y+h4, x+w*-2, y+h5], 0.6, 8))
                axis_dict['register_military'].append(([x, y+h4, x+w3, y+h5], 0.8, 8))
                axis_dict['register_service_office'].append(([x+w1, y+h5, x+w*-2, y+h6], 0.4, 8))
                axis_dict['register_career'].append(([x, y+h5, x+w3, y+h6], 0.6, 8))
                continue
            if 'register_education' not in axis_true and i[1][0] in ['文化程度']:
                w1, w2, w3 = 3*w, 4*w, 5.25*w
                axis_true['register_education'] = [x+w, y+h3, x+w*2.25, y+h4]
                axis_dict['register_number'].append(([x+w, y+h2, x+w1, y+h3], 0.8, 9))
                axis_dict['register_belief'].append(([x+w*4.25, y+h1, x+w3, y+h2], 0.4, 9))
                axis_dict['register_height'].append(([x+w*3.5, y+h2, x+w*4.25, y+h3], 0.6, 9))
                axis_dict['register_blood'].append(([x+w*5, y+h2, x+w3, y+h3], 0.4, 9))
                axis_dict['register_marriage'].append(([x+w1, y+h3, x+w*3.5, y+h4], 0.8, 9))
                axis_dict['register_military'].append(([x+w2, y+h3, x+w3, y+h4], 0.6, 9))
                axis_dict['register_service_office'].append(([x+w, y+h4, x+w*3.5, y+h5], 0.8, 9))
                axis_dict['register_career'].append(([x+w2, y+h4, x+w3, y+h5], 0.6, 9))
                axis_dict['register_city'].append(([x+w*1.3, y+h5, x+w3, y+h6], 0.6, 9))
                axis_dict['register_address'].append(([x+w*1.3, y+h6, x+w3, y+h7], 0.4, 9))
                continue
            if 'register_marriage' not in axis_true and i[1][0] in ['婚姻状况']:
                w1, w2, w3 = -2*w, 3.25*w, 5*w
                axis_true['register_marriage'] = [x+w, y+h3, x+w*2, y+h4]
                axis_dict['register_number'].append(([x+w1, y+h2, x+w, y+h3], 0.8, 10))
                axis_dict['register_belief'].append(([x+w2, y+h1, x+w3, y+h2], 0.4, 10))
                axis_dict['register_height'].append(([x+w*2, y+h2, x+w2, y+h3], 0.8, 10))
                axis_dict['register_blood'].append(([x+w*4, y+h2, x+w3, y+h3], 0.6, 10))
                axis_dict['register_education'].append(([x+w1, y+h3, x, y+h4], 0.8, 10))
                axis_dict['register_military'].append(([x+w2, y+h3, x+w3, y+h4], 0.8, 10))
                axis_dict['register_service_office'].append(([x+w1, y+h4, x+w*2, y+h5], 0.8, 10))
                axis_dict['register_career'].append(([x+w2, y+h4, x+w3, y+h5], 0.8, 10))
                axis_dict['register_city'].append(([x-w*1.5, y+h5, x+w3, y+h6], 0.4, 10))
                axis_dict['register_address'].append(([x-w*1.5, y+h6, x+w3, y+h7], 0.4, 10))
                continue
            if 'register_military' not in axis_true and i[1][0] in ['兵役状况', '兵设状况']:
                w1, w2 = -4*w, 3*w
                axis_true['register_military'] = [x+w, y+h3, x+w*3, y+h4]
                axis_dict['register_number'].append(([x+w1, y+h2, x-w, y+h3], 0.6, 11))
                axis_dict['register_belief'].append(([x+w, y+h1, x+w2, y+h2], 0.6, 11))
                axis_dict['register_height'].append(([x, y+h2, x+w, y+h3], 0.8, 11))
                axis_dict['register_blood'].append(([x+w*2, y+h2, x+w2, y+h3], 0.8, 11))
                axis_dict['register_education'].append(([x+w1, y+h3, x-w*2.25, y+h4], 0.8, 11))
                axis_dict['register_marriage'].append(([x-w, y+h3, x, y+h4], 0.8, 11))
                axis_dict['register_service_office'].append(([x+w1, y+h4, x, y+h5], 0.6, 11))
                axis_dict['register_career'].append(([x+w, y+h4, x+w2, y+h5], 0.8, 11))
                axis_dict['register_city'].append(([x-w*3.5, y+h5, x+w2, y+h6], 0.6, 11))
                axis_dict['register_address'].append(([x-w*3.5, y+h6, x+w2, y+h7], 0.4, 11))
                continue
            if 'register_service_office' not in axis_true and i[1][0] in ['服务处所']:
                w1, w2, w3 = 3.5*w, 4.25*w, 5.5*w
                axis_true['register_service_office'] = [x+w, y+h3, x+w1, y+h4]
                axis_dict['register_number'].append(([x+w, y+h1, x+w*3, y+h2], 0.6, 12))
                axis_dict['register_height'].append(([x+w1, y+h1, x+w2, y+h2], 0.6, 12))
                axis_dict['register_blood'].append(([x+w*0.75, y+h1, x+w3, y+h2], 0.6, 12))
                axis_dict['register_education'].append(([x+w, y+h2, x+w*2.25, y+h3], 0.8, 12))
                axis_dict['register_marriage'].append(([x+w*3, y+h2, x+w1, y+h3], 0.8, 12))
                axis_dict['register_military'].append(([x+w2, y+h2, x+w3, y+h3], 0.8, 12))
                axis_dict['register_career'].append(([x+w2, y+h3, x+w3, y+h4], 0.8, 12))
                axis_dict['register_city'].append(([x+w*1.5, y+h4, x+w3, y+h5], 0.8, 12))
                axis_dict['register_address'].append(([x+w*1.5, y+h5, x+w3, y+h6], 0.6, 12))
                continue
            if 'register_city' not in axis_true and i[1][0] in ['何时由何地', '迁来本市（县）']:
                y += h/(2 if i[1][0] in ['何时由何地'] else -2)
                w1, w2, w3 = 2.5*w, 3.1*w, 3.8*w
                axis_true['register_city'] = [x+w, y+h3, x+w3, y+h4]
                axis_dict['register_education'].append(([x+w*0.75, y+h1, x+w*1.5, y+h2], 0.6, 13))
                axis_dict['register_marriage'].append(([x+w*2.2, y+h1, x+w1, y+h2], 0.6, 13))
                axis_dict['register_military'].append(([x+w2, y+h1, x+w3, y+h2], 0.6, 13))
                axis_dict['register_service_office'].append(([x+w*0.75, y+h2, x+w1, y+h3], 0.8, 13))
                axis_dict['register_career'].append(([x+w2, y+h2, x+w3, y+h3], 0.8, 13))
                axis_dict['register_address'].append(([x+w, y+h4, x+w3, y+h5], 0.8, 13))
                continue
            if 'register_address' not in axis_true and '迁来本址' in i[1][0]:
                w1, w2, w3 = 2.5*w, 3.1*w, 3.8*w
                axis_true['register_address'] = [x+w, y+h3, x+w3, y+h4]
                axis_dict['register_service_office'].append(([x+w*0.75, y+h1, x+w1, y+h2], 0.6, 14))
                axis_dict['register_career'].append(([x+w2, y+h1, x+w3, y+h2], 0.4, 14))
                axis_dict['register_city'].append(([x+w, y+h2, x+w3, y+h3], 0.8, 14))
                continue
        if '姓' in register_name and '名' in register_name:
            axis = register_name['姓'][:2]+register_name['名'][-2:]
            h_diff = abs(register_name['姓'][1]-register_name['名'][1])
            h = axis[3]-axis[1]
            w = axis[2]-axis[0]
            if h==0:
                h = 1
            if w==0:
                w = 1
            x = axis[0]
            y = axis[1]
            if h_diff/h<0.25:
                if h>h_abs and h_abs>0:
                    y += (h-h_abs)/2
                    h = h_abs
                w1, w2, w3 = 2.75*w, 3.75*w, 5.25*w
                h1, h2, h3, h4, h5, h6 = -4.5*h, -2.5*h, -0.5*h, 1.5*h, 3.5*h, 5.5*h
                axis_true['register_name'] = [x+w, y+h3, x+w1, y+h4]
                axis_dict['register_relation'].append(([x+w2, y+h3, x+w3, y+h4], 0.8, 15))
                axis_dict['register_previous_name'].append(([x+w, y+h4, x+w1, y+h5], 0.8, 15))
                axis_dict['register_sex'].append(([x+w2, y+h4, x+w3, y+h5], 0.6, 15))
                axis_dict['register_birthplace'].append(([x+w, y+h5, x+w1, y+h6], 0.6, 15))
                axis_dict['register_nation'].append(([x+w2, y+h5, x+w3, y+h6], 0.4, 15))
        if '性' in register_sex and '别' in register_sex:
            axis = register_sex['性'][:2]+register_sex['别'][-2:]
            h_diff = abs(register_sex['性'][1]-register_sex['别'][1])
            h = axis[3]-axis[1]
            w = axis[2]-axis[0]
            if h==0:
                h = 1
            if w==0:
                w = 1
            x = axis[0]
            y = axis[1]
            if h_diff/h<0.25:
                if h>h_abs and h_abs>0:
                    y += (h-h_abs)/2
                    h = h_abs
                w1, w2 = -2.25*w, 2.75*w
                h1, h2, h3, h4, h5, h6 = -4.5*h, -2.5*h, -0.5*h, 1.5*h, 3.5*h, 5.5*h
                axis_true['register_sex'] = [x+w, y+h3, x+w2, y+h4]
                axis_dict['register_name'].append(([x+w1, y+h2, x, y+h3], 0.6, 16))
                axis_dict['register_relation'].append(([x+w, y+h2, x+w2, y+h3], 0.8, 16))
                axis_dict['register_previous_name'].append(([x+w1, y+h3, x, y+h4], 0.8, 16))
                axis_dict['register_birthplace'].append(([x+w1, y+h4, x, y+h5], 0.6, 16))
                axis_dict['register_nation'].append(([x+w, y+h4, x+w2, y+h5], 0.8, 16))
                axis_dict['register_nativeplace'].append(([x+w1, y+h5, x, y+h6], 0.4, 16))
                axis_dict['register_born'].append(([x+w, y+h5, x+w2, y+h6], 0.6, 16))
        if '民' in register_nation and '族' in register_nation:
            axis = register_nation['民'][:2]+register_nation['族'][-2:]
            h_diff = abs(register_nation['民'][1]-register_nation['族'][1])
            h = axis[3]-axis[1]
            w = axis[2]-axis[0]
            if h==0:
                h = 1
            if w==0:
                w = 1
            x = axis[0]
            y = axis[1]
            if h_diff/h<0.25:
                if h>h_abs and h_abs>0:
                    y += (h-h_abs)/2
                    h = h_abs
                w1, w2 = -2.25*w, 2.75*w
                h1, h2, h3, h4, h5, h6 = -4.5*h, -2.5*h, -0.5*h, 1.5*h, 3.5*h, 5.5*h
                axis_true['register_nation'] = [x+w, y+h3, x+w2, y+h4]
                axis_dict['register_name'].append(([x+w1, y+h1, x, y+h2], 0.4, 17))
                axis_dict['register_relation'].append(([x+w, y+h1, x+w2, y+h2], 0.6, 17))
                axis_dict['register_previous_name'].append(([x+w1, y+h2, x, y+h3], 0.6, 17))
                axis_dict['register_sex'].append(([x+w, y+h2, x+w2, y+h3], 0.8, 17))
                axis_dict['register_birthplace'].append(([x+w1, y+h3, x, y+h4], 0.8, 17))
                axis_dict['register_nativeplace'].append(([x+w1, y+h4, x, y+h5], 0.6, 17))
                axis_dict['register_born'].append(([x+w, y+h4, x+w2, y+h5], 0.8, 17))
                axis_dict['register_belief'].append(([x+w*1.3, y+h5, x+w2, y+h6], 0.6, 17))
        if '籍' in register_nativeplace and '贯' in register_nativeplace:
            axis = register_nativeplace['籍'][:2]+register_nativeplace['贯'][-2:]
            h_diff = abs(register_nativeplace['籍'][1]-register_nativeplace['贯'][1])
            h = axis[3]-axis[1]
            w = axis[2]-axis[0]
            if h==0:
                h = 1
            if w==0:
                w = 1
            x = axis[0]
            y = axis[1]
            if h_diff/h<0.25:
                if h>h_abs and h_abs>0:
                    y += (h-h_abs)/2
                    h = h_abs
                w1, w2, w3 = 2.75*w, 3.75*w, 5.25*w
                h1, h2, h3, h4, h5, h6 = -4.5*h, -2.5*h, -0.5*h, 1.5*h, 3.5*h, 5.5*h
                axis_true['register_nativeplace'] = [x+w, y+h3, x+w1, y+h4]
                axis_dict['register_previous_name'].append(([x+w, y+h1, x+w1, y+h2], 0.6, 18))
                axis_dict['register_sex'].append(([x+w2, y+h1, x+w3, y+h2], 0.4, 18))
                axis_dict['register_birthplace'].append(([x+w, y+h2, x+w1, y+h3], 0.8, 18))
                axis_dict['register_nation'].append(([x+w2, y+h2, x+w3, y+h3], 0.6, 18))
                axis_dict['register_born'].append(([x+w2, y+h3, x+w3, y+h4], 0.8, 18))
                axis_dict['register_belief'].append(([x+w*4.25, y+h4, x+w3, y+h5], 0.6, 18))
                axis_dict['register_number'].append(([x+w, y+h5, x+w1, y+h6], 0.6, 18))
                axis_dict['register_height'].append(([x+w*3.5, y+h5, x+w*4.25, y+h6], 0.4, 18))
                axis_dict['register_blood'].append(([x+w*4.75, y+h5, x+w3, y+h6], 0.4, 18))
            
#         print(axis_true)
#         print(axis_dict)
        for i in self._keys:
            if i not in axis_true:
                if i in axis_dict:
                    try:
                        weight = sum([j[1] for j in axis_dict[i]])
                        axis_true[i] = [sum([j[0][0]*j[1] for j in axis_dict[i]])/weight,
                                        sum([j[0][1]*j[1] for j in axis_dict[i]])/weight,
                                        sum([j[0][2]*j[1] for j in axis_dict[i]])/weight,
                                        sum([j[0][3]*j[1] for j in axis_dict[i]])/weight]
                    except:
                        print(i, axis_dict[i])
                        raise
        return axis_true
        
    def _fit_characters_neirong(self, axis):
        self._axis = axis.copy()
        axis_true = {i:tuple(axis[i]) for i in axis}
        
        register_city = ''
        register_address = ''
        register_name = ''
        for i in self._result[0]:
            h = (i[0][3][1]+i[0][2][1]-i[0][1][1]-i[0][0][1])/2
            w = (i[0][1][0]+i[0][2][0]-i[0][0][0]-i[0][3][0])/2
            x = min(i[0][0][0], i[0][3][0])
            y = min(i[0][0][1], i[0][1][1])
            if '图片模糊' in self._info['register_name'] and 'register_name' in axis_true:
                h1 = min(max(i[0][3][1], i[0][2][1]), axis_true['register_name'][3])-max(min(i[0][0][1], i[0][1][1]), axis_true['register_name'][1])
                w1 = min(max(i[0][1][0], i[0][2][0]), axis_true['register_name'][2])-max(min(i[0][0][0], i[0][3][0]), axis_true['register_name'][0])            
                if h1/h>0.6 and w1/w>0.6 and '户口' not in i[1][0]:
                    if len(i[1][0])==1:
                        register_name += i[1][0]
                        self._axis['register_name'][3] = i[0][2][1]
                    elif i[1][0].startswith('名') and len(i[1][0])>2:
                        self._info['register_name'] = i[1][0][1:]
                        self._axis['register_name'] = [x, y]+i[0][2]
                    else:
                        self._info['register_name'] = i[1][0]
                        self._axis['register_name'] = [x, y]+i[0][2]
                    continue
            if '图片模糊' in self._info['register_relation'] and 'register_relation' in axis_true:
                h1 = min(max(i[0][3][1], i[0][2][1]), axis_true['register_relation'][3])-max(min(i[0][0][1], i[0][1][1]), axis_true['register_relation'][1])
                w1 = min(max(i[0][1][0], i[0][2][0]), axis_true['register_relation'][2])-max(min(i[0][0][0], i[0][3][0]), axis_true['register_relation'][0])            
                if h1/h>0.6 and w1/w>0.6:
                    self._info['register_relation'] = i[1][0].replace('要', '妻')
                    self._axis['register_relation'] = [x, y]+i[0][2]
                    continue
            if '图片模糊' in self._info['register_previous_name'] and 'register_previous_name' in axis_true:
                h1 = min(max(i[0][3][1], i[0][2][1]), axis_true['register_previous_name'][3])-max(min(i[0][0][1], i[0][1][1]), axis_true['register_previous_name'][1])
                w1 = min(max(i[0][1][0], i[0][2][0]), axis_true['register_previous_name'][2])-max(min(i[0][0][0], i[0][3][0]), axis_true['register_previous_name'][0])            
                if h1/h>0.6 and w1/w>0.6 and len(i[1][0])>1:
                    self._info['register_previous_name'] = i[1][0]
                    self._axis['register_previous_name'] = [x, y]+i[0][2]
                    continue
            if '图片模糊' in self._info['register_sex'] and 'register_sex' in axis_true:
                h1 = min(max(i[0][3][1], i[0][2][1]), axis_true['register_sex'][3])-max(min(i[0][0][1], i[0][1][1]), axis_true['register_sex'][1])
                w1 = min(max(i[0][1][0], i[0][2][0]), axis_true['register_sex'][2])-max(min(i[0][0][0], i[0][3][0]), axis_true['register_sex'][0])            
                if h1/h>0.6 and w1/w>0.6 and i[1][0] in '男女':
                    self._info['register_sex'] = i[1][0]
                    self._axis['register_sex'] = [x, y]+i[0][2]
                    continue
            if '图片模糊' in self._info['register_birthplace'] and 'register_birthplace' in axis_true:
                h1 = min(max(i[0][3][1], i[0][2][1]), axis_true['register_birthplace'][3])-max(min(i[0][0][1], i[0][1][1]), axis_true['register_birthplace'][1])
                w1 = min(max(i[0][1][0], i[0][2][0]), axis_true['register_birthplace'][2])-max(min(i[0][0][0], i[0][3][0]), axis_true['register_birthplace'][0])            
                if h1/h>0.6 and w1/w>0.6 and sum([1 for char in i[1][0] if char not in '出生地'])>2:
                    temp = i[1][0]
                    for char in '出生地':
                        if temp.startswith(char) and len(temp[temp.find(char):])>2:
                            temp = temp[1:]
                    self._info['register_birthplace'] = temp
                    self._axis['register_birthplace'] = [x, y]+i[0][2]
                    continue
            if '图片模糊' in self._info['register_nation'] and 'register_nation' in axis_true:
                h1 = min(max(i[0][3][1], i[0][2][1]), axis_true['register_nation'][3])-max(min(i[0][0][1], i[0][1][1]), axis_true['register_nation'][1])
                w1 = min(max(i[0][1][0], i[0][2][0]), axis_true['register_nation'][2])-max(min(i[0][0][0], i[0][3][0]), axis_true['register_nation'][0])            
                if h1/h>0.6 and w1/w>0.5:
                    temp = i[1][0]
                    if temp.startswith('族') and len(temp)>1:
                        temp = temp[1:]
                    self._info['register_nation'] = temp
                    self._axis['register_nation'] = [x, y]+i[0][2]
                    continue
            if '图片模糊' in self._info['register_nativeplace'] and 'register_nativeplace' in axis_true:
                h1 = min(max(i[0][3][1], i[0][2][1]), axis_true['register_nativeplace'][3])-max(min(i[0][0][1], i[0][1][1]), axis_true['register_nativeplace'][1])
                w1 = min(max(i[0][1][0], i[0][2][0]), axis_true['register_nativeplace'][2])-max(min(i[0][0][0], i[0][3][0]), axis_true['register_nativeplace'][0])            
                if h1/h>0.6 and w1/w>0.6 and sum([1 for char in i[1][0] if char not in '籍贯'])>2:
                    temp = i[1][0]
                    for char in '籍贯':
                        if temp.startswith(char) and len(temp[temp.find(char):])>2:
                            temp = temp[1:]
                    self._info['register_nativeplace'] = temp
                    self._axis['register_nativeplace'] = [x, y]+i[0][2]
                    continue
            if '图片模糊' in self._info['register_born'] and 'register_born' in axis_true:
                h1 = min(max(i[0][3][1], i[0][2][1]), axis_true['register_born'][3])-max(min(i[0][0][1], i[0][1][1]), axis_true['register_born'][1])
                w1 = min(max(i[0][1][0], i[0][2][0]), axis_true['register_born'][2])-max(min(i[0][0][0], i[0][3][0]), axis_true['register_born'][0])            
                if '出生日期' in i[1][0] and len(i[1][0][i[1][0].find('出生日期'):])>7:
                    self._info['register_born'] = i[1][0][i[1][0].find('出生日期')+4:]
                    self._axis['register_born'][3] = i[0][2][1]
                    continue
                if h1/h>0.6 and w1/w>0.6 and sum([1 for char in i[1][0] if char not in '出生期'])>8:
                    temp = i[1][0]
                    for char in '出生期':
                        if temp.startswith(char) and len(temp[1:])>8:
                            temp = temp[1:]
#                     if i[1][0][4]!='年':
#                         i[1][0][:4]
                    self._info['register_born'] = temp
                    self._axis['register_born'] = [x, y]+i[0][2]
                    continue
            if '图片模糊' in self._info['register_belief'] and 'register_belief' in axis_true:
                h1 = min(max(i[0][3][1], i[0][2][1]), axis_true['register_belief'][3])-max(min(i[0][0][1], i[0][1][1]), axis_true['register_belief'][1])
                w1 = min(max(i[0][1][0], i[0][2][0]), axis_true['register_belief'][2])-max(min(i[0][0][0], i[0][3][0]), axis_true['register_belief'][0])            
                if h1/h>0.6 and w1/w>0.6:
                    self._info['register_belief'] = i[1][0]
                    self._axis['register_belief'] = [x, y]+i[0][2]
                    continue
            if '图片模糊' in self._info['register_number'] and 'register_number' in axis_true:
                h1 = min(max(i[0][3][1], i[0][2][1]), axis_true['register_number'][3])-max(min(i[0][0][1], i[0][1][1]), axis_true['register_number'][1])
                w1 = min(max(i[0][1][0], i[0][2][0]), axis_true['register_number'][2])-max(min(i[0][0][0], i[0][3][0]), axis_true['register_number'][0])            
                if h1/h>0.6 and w1/w>0.6 and sum([1 for char in i[1][0] if char in '0123456789xX'])>10:
                    if len(i[1][0])==17:
                        self._info['register_number'] = '1'+i[1][0]
                    else:
                        self._info['register_number'] = i[1][0]
                    self._axis['register_number'] = [x, y]+i[0][2]
                    continue
            if '图片模糊' in self._info['register_height'] and 'register_height' in axis_true:
                h1 = min(max(i[0][3][1], i[0][2][1]), axis_true['register_height'][3])-max(min(i[0][0][1], i[0][1][1]), axis_true['register_height'][1])
                w1 = min(max(i[0][1][0], i[0][2][0]), axis_true['register_height'][2])-max(min(i[0][0][0], i[0][3][0]), axis_true['register_height'][0])            
                if h1/h>0.6 and w1/w>0.6 and sum([1 for char in i[1][0] if char in '0123456789cmCM厘米'])>2:
                    self._info['register_height'] = i[1][0]
                    self._axis['register_height'] = [x, y]+i[0][2]
                    continue
            if '图片模糊' in self._info['register_blood'] and 'register_blood' in axis_true:
                h1 = min(max(i[0][3][1], i[0][2][1]), axis_true['register_blood'][3])-max(min(i[0][0][1], i[0][1][1]), axis_true['register_blood'][1])
                w1 = min(max(i[0][1][0], i[0][2][0]), axis_true['register_blood'][2])-max(min(i[0][0][0], i[0][3][0]), axis_true['register_blood'][0])            
                if h1/h>0.6 and w1/w>0.6 and sum([1 for char in i[1][0] if char not in '血型'])>0:
                    temp = i[1][0].replace('0', 'o')
                    if '血型' in i[1][0]:
                        temp = temp[temp.find('血型')+2:]
                    self._info['register_blood'] = temp
                    self._axis['register_blood'] = [x, y]+i[0][2]
                    continue
            if '图片模糊' in self._info['register_education'] and 'register_education' in axis_true:
                h1 = min(max(i[0][3][1], i[0][2][1]), axis_true['register_education'][3])-max(min(i[0][0][1], i[0][1][1]), axis_true['register_education'][1])
                w1 = min(max(i[0][1][0], i[0][2][0]), axis_true['register_education'][2])-max(min(i[0][0][0], i[0][3][0]), axis_true['register_education'][0])            
                if h1/h>0.6 and w1/w>0.6 and len(i[1][0])>1:
                    self._info['register_education'] = i[1][0].replace('天', '大')
                    self._axis['register_education'] = [x, y]+i[0][2]
                    continue 
            if '图片模糊' in self._info['register_marriage'] and 'register_marriage' in axis_true:
                h1 = min(max(i[0][3][1], i[0][2][1]), axis_true['register_marriage'][3])-max(min(i[0][0][1], i[0][1][1]), axis_true['register_marriage'][1])
                w1 = min(max(i[0][1][0], i[0][2][0]), axis_true['register_marriage'][2])-max(min(i[0][0][0], i[0][3][0]), axis_true['register_marriage'][0])            
                if '婚姻状况' in i[1][0] and len(i[1][0][i[1][0].find('婚姻状况'):])>5:
                    self._info['register_marriage'] = i[1][0][i[1][0].find('婚姻状况')+4:]
                    self._axis['register_marriage'][3] = i[0][2][1]
                    continue
                if h1/h>0.6 and w1/w>0.6:
                    self._info['register_marriage'] = i[1][0]
                    self._axis['register_marriage'] = [x, y]+i[0][2]
                    continue
            if '图片模糊' in self._info['register_military'] and 'register_military' in axis_true:
                h1 = min(max(i[0][3][1], i[0][2][1]), axis_true['register_military'][3])-max(min(i[0][0][1], i[0][1][1]), axis_true['register_military'][1])
                w1 = min(max(i[0][1][0], i[0][2][0]), axis_true['register_military'][2])-max(min(i[0][0][0], i[0][3][0]), axis_true['register_military'][0])            
                if h1/h>0.6 and w1/w>0.6:
                    self._info['register_military'] = i[1][0]
                    self._axis['register_military'] = [x, y]+i[0][2]
                    continue
            if '图片模糊' in self._info['register_service_office'] and 'register_service_office' in axis_true:
                h1 = min(max(i[0][3][1], i[0][2][1]), axis_true['register_service_office'][3])-max(min(i[0][0][1], i[0][1][1]), axis_true['register_service_office'][1])
                w1 = min(max(i[0][1][0], i[0][2][0]), axis_true['register_service_office'][2])-max(min(i[0][0][0], i[0][3][0]), axis_true['register_service_office'][0])            
                if h1/h>0.6 and w1/w>0.6 and sum([1 for char in i[1][0] if char not in '服务处所'])>1:
                    self._info['register_service_office'] = i[1][0]
                    self._axis['register_service_office'] = [x, y]+i[0][2]
                    continue
            if '图片模糊' in self._info['register_career'] and 'register_career' in axis_true:
                h1 = min(max(i[0][3][1], i[0][2][1]), axis_true['register_career'][3])-max(min(i[0][0][1], i[0][1][1]), axis_true['register_career'][1])
                w1 = min(max(i[0][1][0], i[0][2][0]), axis_true['register_career'][2])-max(min(i[0][0][0], i[0][3][0]), axis_true['register_career'][0])            
#                 print(i[1][0],h1/h, h1,h, w1/w, w1,w)
                if h1/h>0.5 and w1/w>0.6 and len(i[1][0])>1:
                    temp = i[1][0]
                    if sum([1 for char in temp if char in '学生']):
                        temp = '学生'
                    elif sum([1 for char in temp if char in '退休']):
                        temp = '退休'
                    self._info['register_career'] = temp
                    self._axis['register_career'] = [x, y]+i[0][2]
                    continue
            if '图片模糊' in self._info['register_city'] and 'register_city' in axis_true:
                h1 = min(max(i[0][3][1], i[0][2][1]), axis_true['register_city'][3])-max(min(i[0][0][1], i[0][1][1]), axis_true['register_city'][1])
                w1 = min(max(i[0][1][0], i[0][2][0]), axis_true['register_city'][2])-max(min(i[0][0][0], i[0][3][0]), axis_true['register_city'][0])            
                if h1/h>0.6 and w1/w>0.6:
                    register_city += i[1][0].replace(' ', '')
                    self._axis['register_city'] = [x, y]+i[0][2]
                    continue
            if '图片模糊' in self._info['register_address'] and 'register_address' in axis_true:
                h1 = min(max(i[0][3][1], i[0][2][1]), axis_true['register_address'][3])-max(min(i[0][0][1], i[0][1][1]), axis_true['register_address'][1])
                w1 = min(max(i[0][1][0], i[0][2][0]), axis_true['register_address'][2])-max(min(i[0][0][0], i[0][3][0]), axis_true['register_address'][0])            
                if h1/h>0.6 and w1/w>0.6:
                    register_address += i[1][0][i[1][0].find('本址')+2:].replace(' ', '') if '本址' in i[1][0] else i[1][0].replace(' ', '')
                    self._axis['register_address'] = [x, y]+i[0][2]
                    continue
        
        self._info['register_city'] = register_city
        self._info['register_address'] = register_address if '登记日期' not in register_address else ''
        if '图片模糊' in self._info['register_name']:
            self._info['register_name'] = register_name
        if '图片模糊' in self._info['register_nation']:
            self._info['register_nation'] = '汉'
        if '图片模糊' in self._info['register_relation']:
            if self._info['register_sex']=='男':
                self._info['register_relation'] = '之夫'
            else:
                self._info['register_relation'] = '之妻'
        if '图片模糊' in self._info['register_sex']:
            if len(self._info['register_number'])==18:
                self._info['register_sex'] = '男' if int(self._info['register_number'][16])%2 else '女'
            elif [1 for i in '妻女媳' if i in self._info['register_relation']]:
                self._info['register_sex'] = '女'
            else:
                self._info['register_sex'] = '男'
        if '配' in self._info['register_marriage']:
            self._info['register_marriage'] = '有配偶'
        
        for i in self._info:
            if '图片模糊' in self._info[i]:
                self._info[i] = ''
            
        for i in self._axis:
            self._axis[i] = [int(max(0, j)) for j in self._axis[i]]
            
#         t = ['register_name', 'register_previous_name', 'register_relation', 'register_sex', 
#               'register_born', 'register_number', 'register_education', 'register_service_office', 
#               'register_marriage', 'register_military', 'register_career', 'register_city', 
#               'register_address']
#         self._info = {i:self._info[i] for i in t}
            
    
    

# hukouben = OCRHuKouBen()