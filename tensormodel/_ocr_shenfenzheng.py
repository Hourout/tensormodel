import time

import paddleocr
import linora as la

__all__ = ['OCRShenFenZheng']


class OCRShenFenZheng():
    def __init__(self, model=True, name_list=None):
        if model==True:
            self._model = paddleocr.PaddleOCR(show_log=False)
        elif model:
            self._model = model
        else:
            self._model = None
        self._keys_front = ['user_name', 'user_sex', 'user_nation', 'user_born', 'user_address', 'user_number']
        self._keys_back = ['user_type', 'user_organization', 'user_validity_period']
        self._keys = self._keys_front+self._keys_back
        if name_list is None:
            name_list = self._keys.copy()
        else:
            for i in name_list:
                if i not in self._keys:
                    raise ValueError(f'Variable name `{i}`  does not conform to the specification.')
        self._name_list = name_list
        self._char_name = [i+j for i in ['姓', '娃', '妇', '性', '赵', '生'] for j in ['名', '容', '吉']]
        self._char_sex = ['性别']
        self._char_nation = ['民族', '民旅', '民康', '民旗', '民路', '昆旗']
        self._char_address = ['住址', '佳址', '主址', '住 址', '往址', '生址', '佳道']
        self._char_organization = ['签发机关', '鑫发机关', '金设机关', '签发物关', '盛发机关']
#         self._char_number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        
    def predict(self, image, axis=False, model=None, back=True):
        self._show_axis = axis
        self._back = back
        self._mode = ''
        self._axis = dict()
        self._result_front = []
        self._result_back = []
        self._info = '图片模糊或非二代身份证图片'
        self._error = '图片模糊或非二代身份证图片'
        
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
        self._fit_characters(self._axis, self._result_front)
        
        for aug in [0,1,2]:
            error_list = [i for i in self._info if '图片模糊' in self._info[i]]
            if error_list:
                self._result_crop = []
                for i in error_list:
                    if i not in self._axis:
                        continue
                    image = la.image.crop(self._image_front, self._axis[i])
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
        angle = self._angle_front if 'front' in self._mode else self._angle_back
        if self._show_axis:
            return {'data':self._info, 'axis':self._axis, 'angle':angle, 'error':self._error}
        else:
            return {'data':self._info, 'angle':angle, 'error':self._error}
        
    def _fit_direction(self, model):
        for angle in [0, 90, 270, 180]:
            image = la.image.rotate(self._image, angle, expand=True)
            result = model.ocr(la.image.image_to_array(image), cls=False)
            
            if 'front' not in self._mode:
                rank = [0,0,0,0,0]
                for r, i in enumerate(result[0], start=1):
                    if sum([1 for char in self._char_name if char in i[1][0]]):
                        rank[0] = r
                    elif sum([1 for char in self._char_sex if char in i[1][0]]):
                        rank[1] = r
                    elif sum([1 for char in self._char_nation if char in i[1][0]]):
                        rank[1] = r
                    elif '出生' in i[1][0]:
                        rank[2] = r
                    elif len(i[1][0]) in [9, 10, 11] and i[1][0].find('年')==4 and '月' in i[1][0] and i[1][0].endswith('日'):
                        rank[2] = r
                    elif sum([1 for char in self._char_address if char in i[1][0]]) or '址' in i[1][0]:
                        rank[3] = r
                    elif '号码' in i[1][0] or '公民' in i[1][0]:
                        rank[4] = r
                rank = [i for i in rank if i>0]
                if rank==sorted(rank) and len(rank)>1:
                    self._image_front = image
                    self._angle_front = angle
                    self._mode += 'front'
                    self._result_front = result.copy()
                    if isinstance(self._info, str):
                        self._info = {i:'图片模糊' for i in self._keys_front if i in self._name_list}
                    else:
                        for i in self._keys_front:
                            if i in self._name_list:
                                self._info[i] = '图片模糊'
            
            if self._back:
                if 'back' not in self._mode:
                    rank = [0,0]
                    for r, i in enumerate(result[0], start=1):
                        if '中华人民共和国' in i[1][0] or '居民身份证' in i[1][0]:
                            rank[0] = r
                        elif '机关' in i[1][0] or '有效期限' in i[1][0]:
                            rank[1] = r
                    if rank[1]>rank[0]:
                        self._image_back = image
                        self._angle_back = angle
                        self._mode += 'back'
                        self._result_back = result.copy()
                        if isinstance(self._info, str):
                            self._info = {i:'图片模糊' for i in self._keys_back if i in self._name_list}
                        else:
                            for i in self._keys_back:
                                if i in self._name_list:
                                    self._info[i] = '图片模糊'
            
            if self._back and 'back' in self._mode and 'front' in self._mode:
                break
            if not self._back and 'front' in self._mode:
                break
    
    def _fit_axis(self):
        if len(self._result_front)==0:
            return 0

        if 'front' not in self._mode:
            self._axis = {i:[] for i in self._keys_back}
            return 0
        
        axis_true = dict()
#         fix_x = []
        axis_dict = {i:[] for i in self._keys_front}
        for i in self._result_front[0]:
            h = (i[0][3][1]+i[0][2][1]-i[0][1][1]-i[0][0][1])/2
            w = (i[0][1][0]+i[0][2][0]-i[0][0][0]-i[0][3][0])/2
            if 'user_name' not in axis_true:
                for char in self._char_name:
                    if char in i[1][0]:
                        if len(i[1][0][i[1][0].find(char)+len(char):])>1:
                            self._info['user_name'] = (i[1][0][i[1][0].find(char)+len(char):]).strip()
                            n = len(self._info['user_name'])+1 if len(self._info['user_name'])==2 else len(self._info['user_name'])
                            w = w/(len(i[1][0])+2.5+n*1.5)
                            h = h*0.7
                            x = i[0][0][0]+w*i[1][0].find(char)
                            y = i[0][0][1]+h*0.3
                            axis_true['user_name'] = [x+w*3.5, i[0][0][1]]+i[0][2]
                        else:
                            w = w/(len(i[1][0])+1)
                            x = i[0][0][0]+w*i[1][0].find(char)
                            y = i[0][0][1]
                            axis_true['user_name'] = [x+w*3.5, y-h, x+w*13, y+h*1.5]
                        axis_dict['user_sex'].append(([x+w*3.5, y+h*2, x+w*6, y+h*3.75], 0.8))
                        axis_dict['user_nation'].append(([x+w*11.5, y+h*2, x+w*15, y+h*3.75], 0.8))
                        axis_dict['user_born'].append(([x+w*3.5, y+h*4.5, x+w*18, y+h*6.25], 0.6))
                        axis_dict['user_address'].append(([x+w*3.5, y+h*6.5, x+w*21.5, y+h*11], 0.4))
                        break
                if 'user_name' in axis_true:
                    continue
            if 'user_sex' not in axis_true:
                for char in self._char_sex:
                    if char in i[1][0] and len(i[1][0][i[1][0].find(char)+len(char):])<2:
                        if len(i[1][0][i[1][0].find(char)+len(char):])==1:
                            w = w/(len(i[1][0])+2.5)
                            h = h*0.75
                            x = i[0][0][0]+w*i[1][0].find(char)
                            y = i[0][0][1]
                            axis_true['user_sex'] = [x+w*3.5, y]+i[0][2]
                        else:
                            w = w/(len(i[1][0])+1)
                            x = i[0][0][0]+w*i[1][0].find(char)
                            y = i[0][0][1]
                            axis_true['user_sex'] = [x+w*3.5, y, x+w*6, y+h*1.25]
                        axis_dict['user_name'].append(([x+w*3.5, y-h*3, x+w*13, y-h], 0.8))
                        axis_dict['user_nation'].append(([x+w*11.5, y, x+w*15, y+h*1.25], 0.8))
                        axis_dict['user_born'].append(([x+w*3.5, y+h*2, x+w*17, y+h*4], 0.8))
                        axis_dict['user_address'].append(([x+w*3.5, y+h*4.5, x+w*21.5, y+h*9], 0.6))
                        break
                if 'user_sex' in axis_true:
                    continue
            if 'user_nation' not in axis_true:
                for char in self._char_nation:
                    if char in i[1][0] and '男' not in i[1][0] and '女' not in i[1][0]:
                        if len(i[1][0][i[1][0].find(char)+len(char):])>0:
                            w = w/(len(i[1][0])+1.5)
                            h = h*0.75
                        else:
                            w = w/(len(i[1][0])+1)
                        x = i[0][0][0]+w*i[1][0].find(char)
                        y = i[0][0][1]
                        axis_true['user_nation'] = [x+w*3.5, y, x+w*7, y+h*1.25]
                        axis_dict['user_name'].append(([x-w*4.5-w*i[1][0].find(char), y-h*3, x+w*4, y-h], 0.6))
                        axis_dict['user_sex'].append(([x-w*4.5-w*i[1][0].find(char), y, x-w*2.5, y+h*1.25], 0.6))
                        axis_dict['user_born'].append(([x-w*4.5-w*i[1][0].find(char), y+h*2, x+w*9, y+h*4], 0.6))
                        axis_dict['user_address'].append(([x-w*4.5-w*i[1][0].find(char), y+h*4.5, x+w*12, y+h*9], 0.4))
                        break
                if 'user_nation' in axis_true:
                    continue
            if len(i[1][0]) in [9, 10, 11] and i[1][0].find('年')==4 and '月' in i[1][0] and i[1][0].endswith('日'):
#                 fix_x.append(i[0][0][0])
                x = i[0][0][0]
                y = i[0][0][1]
                axis_true['user_born'] = [min(i[0][0][0], i[0][3][0]), min(i[0][0][1], i[0][1][1]), 
                                          max(i[0][1][0], i[0][2][0]), max(i[0][2][1], i[0][3][1])]
                axis_dict['user_name'].append(([x, y-h*5.5, x+w*0.5, y-h*3.5], 0.6))
                axis_dict['user_sex'].append(([x, y-h*2.25, x+w*0.2, y-h*0.8], 0.8))
                axis_dict['user_nation'].append(([x+w*0.65, y-h*2.25, x+w*0.85, y-h*0.8], 0.8))
                axis_dict['user_address'].append(([x, y+h*2, x+w*1.3, y+h*6.5], 0.8))
                continue
            if 'user_born' not in axis_true and '出生'==i[1][0]:
                w = w/(len(i[1][0])+1)
                x = i[0][0][0]
                y = i[0][0][1]
                axis_true['user_born'] = [x+w*3.5, y-h*0.5, x+w*18, y+h*0.5]
                axis_dict['user_name'].append(([x+w*3.5, y-h*5.5, x+w*10, y-h*3], 0.6))
                axis_dict['user_sex'].append(([x+w*3.5, y-h*3, x+w*6, y-h], 0.8))
                axis_dict['user_nation'].append(([x+w*11.5, y-h*3, x+w*15, y-h], 0.8))
                axis_dict['user_address'].append(([x+w*3.5, y+h*1.5, x+w*21.5, y+h*5.5], 0.8))
                continue
            if 'user_address' not in axis_true:
                for char in self._char_address:
                    if char in i[1][0]:
                        if len(i[1][0][i[1][0].find(char)+len(char):])>0:
                            w = w/(len(i[1][0])+2.5)
                            h = h*0.75
                        else:
                            w = w/(len(i[1][0])+1)
                        x = i[0][0][0]+w*i[1][0].find(char)
                        y = i[0][0][1]
                        axis_true['user_address'] = [x+w*3.2, y-h*0.5, x+w*21.5, y+h*5]
                        axis_dict['user_name'].append(([x+w*3.5, y-h*8, x+w*10, y-h*5.5], 0.4))
                        axis_dict['user_sex'].append(([x+w*3.5, y-h*5.25, x+w*6, y-h*3.5], 0.6))
                        axis_dict['user_nation'].append(([x+w*11.5, y-h*5.25, x+w*15, y-h*3.5], 0.4))
                        axis_dict['user_born'].append(([x+w*3.5, y-h*2.75, x+w*18, y-h], 0.8))
                        break
                if 'user_address' in axis_true:
                    continue
            if 'user_number' not in axis_true:
                if sum([1 for j in i[1][0][-18:] if j in '0123456789xX'])==18:
                    axis_true['user_number'] = [min(i[0][0][0], i[0][3][0]), min(i[0][0][1], i[0][1][1]), 
                                                max(i[0][1][0], i[0][2][0]), max(i[0][2][1], i[0][3][1])]
                    if len(i[1][0][:-18]):
                        axis_true['user_number'][0] = axis_true['user_number'][0]+w*len(i[1][0][:-18])/(len(i[1][0])-18+13+1.5)
                    axis_dict['user_address'].append(([axis_true['user_number'][0]-w*4.5/13, axis_true['user_number'][1]-h*6, 
                                                       axis_true['user_number'][0]+w*6.5/13, axis_true['user_number'][1]-h*1.5], 0.8))

        for i in self._keys_front:
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
        axis_true = {i:tuple(axis[i]) for i in axis}
        if 'front' in self._mode:
            fix_x = []
            address = ''
            rank = 0
            for i in result[0]:
                h = max((i[0][3][1]+i[0][2][1]-i[0][1][1]-i[0][0][1])/2, 1)
                w = max((i[0][1][0]+i[0][2][0]-i[0][0][0]-i[0][3][0])/2, 1)
                x = min(i[0][0][0], i[0][3][0])
                y = min(i[0][0][1], i[0][1][1])
                if '图片模糊' in self._info.get('user_name', '') and 'user_name' in axis_true:
                    h1 = min(max(i[0][3][1], i[0][2][1]), axis_true['user_name'][3])-max(min(i[0][0][1], i[0][1][1]), axis_true['user_name'][1])
                    w1 = min(max(i[0][1][0], i[0][2][0]), axis_true['user_name'][2])-max(min(i[0][0][0], i[0][3][0]), axis_true['user_name'][0])            
                    if h1/h>0.6 and w1/w>0.6 or (h1/h>0.25 and w1/w>0.25 and rank==0):
                        if len(i[1][0])>3 and i[1][0].find('名')==1:
                            self._info['user_name'] = i[1][0][2:]
                            self._axis['user_name'] = [self._axis['user_name'][0], i[0][0][1]]+i[0][2]
                            rank += 1
                        elif i[1][0].startswith('名') and len(i[1][0])>2:
                            self._info['user_name'] = i[1][0][1:]
                            self._axis['user_name'] = [self._axis['user_name'][0], i[0][0][1]]+i[0][2]
                            rank += 1
                        elif len(i[1][0])>1:
                            self._info['user_name'] = i[1][0]
                            self._axis['user_name'] = [self._axis['user_name'][0], i[0][0][1]]+i[0][2]
                            fix_x.append(i[0][0][0])
                            rank += 1
                    if '图片模糊' not in self._info['user_name']:
                        continue
                if '图片模糊' in self._info.get('user_nation', ''):
                    for char in self._char_nation:
                        if char in i[1][0]:
                            if  i[1][0].endswith('汉'):
                                self._info['user_nation'] = '汉'
                                self._axis['user_nation'] = [i[0][0][0]+(i[0][1][0]-i[0][0][0])/(len(i[1][0])+2)*(len(i[1][0])+1), 
                                                             i[0][0][1]]+i[0][2]
                                break
                            elif (i[1][0][i[1][0].find(char)+len(char):]).strip()!='':
                                self._info['user_nation'] = (i[1][0][i[1][0].find(char)+2:]).strip()
                                self._axis['user_nation'] = [i[0][0][0]+(i[0][1][0]-i[0][0][0])/(len(i[1][0])+2)*(len(i[1][0])+2-len(self._info['user_nation'])), 
                                                             i[0][0][1]]+i[0][2]
                                break
                    if '图片模糊' not in self._info['user_nation']:
                        continue
                if '图片模糊' in self._info.get('user_born', ''):
                    if '年' in i[1][0] and '月' in i[1][0] and i[1][0].endswith('日'):
                        self._info['user_born'] = i[1][0][max(0, i[1][0].find('年')-4):]
                        continue
                if '图片模糊' in self._info.get('user_number', ''):
                    temp = find_shenfenzheng(i[1][0])
                    if temp:
                        self._info['user_number'] = temp[0]
                        self._info['user_sex'] =  temp[1]
                        t = temp[2]
                        if t[5]=='0':
                            t = t[:5]+t[6:]
                        if t[-3]=='0':
                            t = t[:-3]+t[-2:]
                        self._info['user_born'] = t
                        continue
                if '图片模糊' in self._info.get('user_address', '') and 'user_address' in axis_true:
                    h1 = min(max(i[0][3][1], i[0][2][1]), axis_true['user_address'][3])-max(min(i[0][0][1], i[0][1][1]), axis_true['user_address'][1])
                    w1 = min(max(i[0][1][0], i[0][2][0]), axis_true['user_address'][2])-max(min(i[0][0][0], i[0][3][0]), axis_true['user_address'][0])
                    if h1/h>0.6 and w1/w>0.6:
                        if len(address)==0:
                            for char in self._char_address:
                                if i[1][0].startswith(char):
                                    address += (i[1][0][i[1][0].find(char)+len(char):]).strip()
                                    self._axis['user_address'][1] = min(i[0][0][1], i[0][1][1])
                                    self._axis['user_address'][2] = max(i[0][1][0], i[0][2][0])
                                    break
                            if len(address)==0:
                                address += i[1][0]
                                self._axis['user_address'][1] = min(i[0][0][1], i[0][1][1])
                                self._axis['user_address'][2] = max(i[0][1][0], i[0][2][0])
                                fix_x.append(i[0][0][0])
                        else:
                            address += i[1][0]
                            self._axis['user_address'][3] = max(i[0][2][1], i[0][3][1])
                            fix_x.append(i[0][0][0])
                        continue
            
            if '图片模糊' in self._info.get('user_address', '') and address!='':
                if '公民身份号码' in address:
                    address = address[:address.find('公民身份号码')]
                for i in ['住', '址', '佳', '主', '往', '生', '佳', '道']:
                    if address.startswith(i):
                        address = address[1:]
                        break
                self._info['user_address'] = address
            if '图片模糊' in self._info.get('user_nation', ''):
                self._info['user_nation'] = '汉'
            if '图片模糊' in self._info.get('user_sex', ''):
                for i in self._result_up[0]:
                    if '女' in i[1][0]:
                        self._info['user_sex'] = '女'
                        break
                if '图片模糊' in self._info['user_sex']:
                    self._info['user_sex'] = '男'

            if self._show_axis:
                try:
                    if len(fix_x)>0:
                        fix_x = sum(fix_x)/len(fix_x)
                        if 'user_name' in self._axis:
                            self._axis['user_name'][0] = fix_x
                        if 'user_sex' in self._axis:
                            self._axis['user_sex'][0] = fix_x
                        if 'user_born' in self._axis:
                            self._axis['user_born'][0] = fix_x
                        if 'user_address' in self._axis:
                            self._axis['user_address'][0] = fix_x

                    fix_y = []
                    for i in result[0]:
                        if '性别' in i[1][0]:
                            fix_y.append(min(i[0][0][1], i[0][1][1]))
                        elif sum([1 for char in self._char_nation if char in i[1][0]])==1:
                            fix_y.append(min(i[0][0][1], i[0][1][1]))
                    if len(fix_y)>0:
                        fix_y = sum(fix_y)/len(fix_y)
                        if 'user_sex' in self._axis:
                            self._axis['user_sex'][1] = fix_y
                        if 'user_nation' in self._axis:
                            self._axis['user_nation'][1] = fix_y

                    if 'user_sex' in self._axis and 'user_address' in self._axis:
                        self._axis['user_sex'][2] = self._axis['user_address'][0]+(self._axis['user_address'][2]-self._axis['user_address'][0])/11*1.5
                    if 'user_born' in self._axis and 'user_address' in self._axis:
                        self._axis['user_born'][2] = min(self._axis['user_born'][2], self._axis['user_address'][2])

                    if 'user_number' in self._axis and 'user_address' in self._axis and 'user_name' in self._axis:
                        self._axis['user_face'] = [0,0,0,0]
                        self._axis['user_card'] = [0,0,0,0]
                        self._axis['user_face'][0] = self._axis['user_address'][2]+(self._axis['user_number'][3]-self._axis['user_number'][1])*0.7
                        self._axis['user_face'][1] = self._axis['user_name'][1]
                        self._axis['user_face'][2] = self._axis['user_face'][0]+(self._axis['user_number'][2]-self._axis['user_number'][0])*0.58
                        self._axis['user_face'][3] = self._axis['user_number'][1]-(self._axis['user_number'][3]-self._axis['user_number'][1])
                        self._axis['user_card'][0] = self._axis['user_address'][0]-(self._axis['user_address'][2]-self._axis['user_address'][0])*0.45
                        self._axis['user_card'][1] = self._axis['user_name'][1]-(self._axis['user_name'][3]-self._axis['user_name'][1])*2.25
                        self._axis['user_card'][2] = self._axis['user_face'][0]+(self._axis['user_number'][2]-self._axis['user_number'][0])*0.68
                        self._axis['user_card'][3] = self._axis['user_number'][1]+(self._axis['user_number'][3]-self._axis['user_number'][1])*3
                except:
                    pass
        else:
            for i in self._result_back[0]:
                if '图片模糊' in self._info.get('user_type', ''):
                    if '居住证' in i[1][0] or '身份证' in i[1][0]:
                        self._info['user_type'] = i[1][0]
                        self._axis['user_type'] = i[0][0]+i[0][2]
                        continue
                if '图片模糊' in self._info.get('user_organization', ''):
                    temp = i[1][0].strip()
                    for char in ['公委局', '公农局']:
                        temp = temp.replace(char, '公安局')
                    if '公安局' in i[1][0] or '分局' in i[1][0]:
                        for char in self._char_organization:
                            if char in temp:
                                if temp[temp.find(char)+len(char):]!='':
                                    temp = temp[temp.find(char)+len(char):]
                                    break
                        self._info['user_organization'] = temp
                        self._axis['user_organization'] = i[0][0]+i[0][2]
                        continue
                if '图片模糊' in self._info.get('user_validity_period', ''):
                    if sum([1 for char in ['长期', '.', '-', '一'] if char in i[1][0]])>0:
                        if sum([1 for char in self._char_number if char in i[1][0]])>1:
                            if sum([1 for j in i[1][0][-21:] if j in '0123456789.-一'])==21:
                                temp = i[1][0][-21:]
                                self._info['user_validity_period'] = f'{temp[:4]}.{temp[5:7]}.{temp[8:10]}-{temp[11:15]}.{temp[16:18]}.{temp[19:21]}'
                            elif i[1][0].endswith('长期'):
                                if sum([1 for j in i[1][0][-13:] if j in '0123456789.-长期'])==13:
                                    self._info['user_validity_period'] = i[1][0][-13:]
                                else:
                                    temp = i[1][0]
                                    for j in ['.', '一', ':', '-', ',']:
                                        temp = temp.replace(j, '')
                                    while temp[0] not in self._char_number:
                                        temp = temp[1:]
                                    if len(temp)==10:
                                        self._info['user_validity_period'] = f'{temp[:4]}.{temp[4:6]}.{temp[6:8]}-长期'
                            else:
                                temp = i[1][0]
                                for j in ['.', '一', ':', '-', ',']:
                                    temp = temp.replace(j, '')
                                while temp[0] not in self._char_number:
                                    temp = temp[1:]
                                if len(temp)==16:
                                    self._info['user_validity_period'] = f'{temp[:4]}.{temp[4:6]}.{temp[6:8]}-{temp[8:12]}.{temp[12:14]}.{temp[14:16]}'
                                elif len(temp) in [14, 15]:
                                    self._info['user_validity_period'] = f'{temp[:4]}.{temp[4:6]}.{temp[6:8]}-{temp[8:12]}.{temp[12:14]}.{temp[6:8]}'
                                elif len(temp) in [12, 13]:
                                    self._info['user_validity_period'] = f'{temp[:4]}.{temp[4:6]}.{temp[6:8]}-{temp[8:12]}.{temp[4:6]}.{temp[6:8]}'
                                elif len(temp) in [8, 9, 10, 11]:
                                    try:
                                        self._info['user_validity_period'] = f'{temp[:4]}.{temp[4:6]}.{temp[6:8]}-{int(temp[:4])+20}.{temp[4:6]}.{temp[6:8]}'
                                    except:
                                        pass
                            if '图片模糊' not in self._info['user_validity_period']:
                                self._axis['user_validity_period'] = i[0][0]+i[0][2]
                                if '长期' not in self._info['user_validity_period']:
                                    if sum([1 for j in self._info['user_validity_period'] if j in '0123456789'])>=16:
                                        if abs(int(self._info['user_validity_period'][:4])-int(self._info['user_validity_period'][11:15]))>30:
                                            self._info['user_validity_period'] = '图片模糊:未识别出有效期限'
                                            self._axis.pop('user_validity_period')
                                            continue
            if '图片模糊' in self._info.get('user_type', ''):
                self._info['user_type'] = '居民身份证'
        for i in self._axis:
            self._axis[i] = [int(max(0, j)) for j in self._axis[i]]

    def draw_mask(self):
        if 'front' in self._mode:
            image = self._image_front.copy()
        elif 'back' in self._mode:
            image = self._image_back.copy()
        else:
             image = self._image.copy()
        try:
            t = [la.image.box_convert(self._axis[i], 'xyxy', 'axis') for i in self._axis if i in self._keys]
            if len(t)>0:
                image = la.image.draw_box(image, t, width=2)
        except:
            pass
        return image
    
    def env_check(self):
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
            name_list = ['user_name', 'user_sex', 'user_nation', 'user_born', 'user_address', 'user_number',
                         'user_type', 'user_organization', 'user_validity_period']

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

