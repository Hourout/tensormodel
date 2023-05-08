from collections import defaultdict

import paddleocr
import linora as la

__all__ = ['OCRIDCard']


class OCRIDCard():
    def __init__(self):
        self.ocr = paddleocr.PaddleOCR(show_log=False)
        self._error = 'ok'
        self._char_name = [i+j for i in ['姓', '娃', '妇', '性'] for j in ['名', '容', '吉']]
        self._char_sex = ['性别']
        self._char_nation = ['民族', '民旅', '民康', '民旗', '民路', '昆旗']
        self._char_address = ['住址', '佳址', '主址', '住 址', '往址', '生址', '佳道']
        self._char_organization = ['签发机关', '鑫发机关', '金设机关', '签发物关']
        self._char_number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        self._keys = ['user_name', 'user_sex', 'user_nation', 'user_born', 'user_address', 
                      'user_number', 'user_face', 'user_card']
        
    def predict(self, image, back=True):
        self._axis = None
        if isinstance(image, str):
            image = la.image.read_image(image)
            self._image = la.image.color_convert(image)
        self._direction_transform(self._image, back)
        if isinstance(self._info, str):
            self._direction_transform(la.image.enhance_brightness(self._image, 0.8), back)
        if isinstance(self._info, str):
            return self._info
        self._axis_transform_up()
        self._axis_transform_down()
        for i in self._info:
            if '图片模糊' in self._info[i]:
                self._temp = self._info.copy()
                self._direction_transform(la.image.enhance_brightness(self._image, 0.8), back)
                self._axis_transform_up()
                self._axis_transform_down()
                if isinstance(self._info, str):
                    self._info = self._temp.copy()
                else:
                    for j in self._temp:
                        if '图片模糊' not in self._temp[j]:
                            self._info[j] = self._temp[j]
                break

        if self._angle_up!=-1:
            angle = self._angle_up
            if '图片模糊' in self._info['user_number']:
                self._error = '图片模糊:未识别出身份证号码'
            if self._info['user_address'].strip()=='':
                self._error = '图片模糊:未识别出住址'
            if '图片模糊' in self._info['user_name']:
                self._error = '图片模糊:未识别出姓名'
        elif self._angle_down!=-1:
            angle = self._angle_down
        else:
            angle = 0
        return {'data':self._info, 'axis':self._axis, 'angle':angle, 'error':self._error}
        
    def _direction_transform(self, image, back):
        state_up = False
        state_down = False
        self._result_up = []
        self._result_down = []
        self._angle_up = -1
        self._angle_down = -1
        for angle in [0, 90, 180, 270]:
            if angle>0:
                image1 = la.image.rotate(image, angle, expand=True)
                image1 = la.image.image_to_array(image1)
            else:
                image1 = la.image.image_to_array(image)
            result = self.ocr.ocr(image1, cls=False)
            
            if not state_up:
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
                    state_up = True
                    self._result_up = result.copy()
                    self._angle_up = angle

            if back:
                if not state_down:
                    rank = [0,0]
                    for r, i in enumerate(result[0], start=1):
                        if '中华人民共和国' in i[1][0] or '居民身份证' in i[1][0]:
                            rank[0] = r
                        elif '机关' in i[1][0] or '有效期限' in i[1][0]:
                            rank[1] = r
                    if rank[1]>rank[0]:
                        state_down = True
                        self._result_down = result.copy()
                        self._angle_down = angle
                        
                if state_down and state_up:
                    break
            else:
                if state_up:
                    break
        
        self._info = {}
        if state_up:
            self._info['user_name'] = '图片模糊:未识别出姓名'
            self._info['user_sex'] = '图片模糊:未识别出性别'
            self._info['user_nation'] = '图片模糊:未识别出民族'
            self._info['user_born'] = '图片模糊:未识别出出生日期'
            self._info['user_address'] = '图片模糊:未识别出地址'
            self._info['user_number'] = '图片模糊:未识别出身份证号码'
        if back:
            if state_down:
                self._info['user_type'] = '居民身份证'
                self._info['user_organization'] = '图片模糊:未识别出签发机关'
                self._info['user_validity_period'] = '图片模糊:未识别出有效期限'
            if state_up == state_down == False:
                self._info = '图片模糊:未识别出有效信息'
                self._error = '图片模糊:未识别出有效信息'
        else:
            if not state_up:
                self._info = '图片模糊:未识别出有效信息'
                self._error = '图片模糊:未识别出有效信息'
    
    def _axis_transform_up(self):
        if len(self._result_up)==0:
            return 0
        fix_x = []
        axis_true = defaultdict(list)
        axis_dict = defaultdict(list)
        
#         height = [(i[0][3][1]+i[0][2][1]-i[0][1][1]-i[0][0][1])/2 for i in self._result_up[0]]
#         height = sum(height)/len(height)
        
        for i in self._result_up[0]:
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
            if len(i[1][0]) in [9, 10, 11] and i[1][0].find('年')==4 and '月' in i[1][0] and i[1][0].endswith('日'):
                fix_x.append(i[0][0][0])
                x = i[0][0][0]
                y = i[0][0][1]
                axis_true['user_born'] = [min(i[0][0][0], i[0][3][0]), min(i[0][0][1], i[0][1][1]), 
                                          max(i[0][1][0], i[0][2][0]), max(i[0][2][1], i[0][3][1])]
                axis_dict['user_name'].append(([x, y-h*5.5, x+w*0.5, y-h*3.5], 0.6))
                axis_dict['user_sex'].append(([x, y-h*2.25, x+w*0.2, y-h*0.8], 0.8))
                axis_dict['user_nation'].append(([x+w*0.65, y-h*2.25, x+w*0.85, y-h*0.8], 0.8))
                axis_dict['user_address'].append(([x, y+h*2, x+w*1.3, y+h*6.5], 0.8))
            if 'user_born' not in axis_true:
                if '出生'==i[1][0]:
                    w = w/(len(i[1][0])+1)
                    x = i[0][0][0]
                    y = i[0][0][1]
                    axis_true['user_born'] = [x+w*3.5, y-h*0.5, x+w*18, y+h*0.5]
                    axis_dict['user_name'].append(([x+w*3.5, y-h*5.5, x+w*10, y-h*3], 0.6))
                    axis_dict['user_sex'].append(([x+w*3.5, y-h*3, x+w*6, y-h], 0.8))
                    axis_dict['user_nation'].append(([x+w*11.5, y-h*3, x+w*15, y-h], 0.8))
                    axis_dict['user_address'].append(([x+w*3.5, y+h*1.5, x+w*21.5, y+h*5.5], 0.8))
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
                        axis_true['user_address'] = [x+w*3.2, y-h*0.5, x+w*21.5, y+h*4.5]
                        axis_dict['user_name'].append(([x+w*3.5, y-h*8, x+w*10, y-h*5.5], 0.4))
                        axis_dict['user_sex'].append(([x+w*3.5, y-h*5.25, x+w*6, y-h*3.5], 0.6))
                        axis_dict['user_nation'].append(([x+w*11.5, y-h*5.25, x+w*15, y-h*3.5], 0.4))
                        axis_dict['user_born'].append(([x+w*3.5, y-h*2.75, x+w*18, y-h], 0.8))
                        break
            if 'user_number' not in axis_true:
                if sum([1 for j in i[1][0][-18:] if j in '0123456789xX'])==18:
                    axis_true['user_number'] = [min(i[0][0][0], i[0][3][0]), min(i[0][0][1], i[0][1][1]), 
                                                max(i[0][1][0], i[0][2][0]), max(i[0][2][1], i[0][3][1])]
                    if len(i[1][0][:-18]):
                        axis_true['user_number'][0] = axis_true['user_number'][0]+w*len(i[1][0][:-18])/(len(i[1][0])-18+13+1.5)
                    axis_dict['user_address'].append(([axis_true['user_number'][0]-w*4.5/13, axis_true['user_number'][1]-h*6, 
                                                       axis_true['user_number'][0]+w*6.5/13, axis_true['user_number'][1]-h*1.5], 0.8))
        
        for i in ['user_name', 'user_sex', 'user_nation', 'user_born', 'user_address', 'user_number']:
            if i not in axis_true:
                if i in axis_dict:
                    weight = sum([j[1] for j in axis_dict[i]])
                    axis_true[i] = [sum([j[0][0]*j[1] for j in axis_dict[i]])/weight,
                                    sum([j[0][1]*j[1] for j in axis_dict[i]])/weight,
                                    sum([j[0][2]*j[1] for j in axis_dict[i]])/weight,
                                    sum([j[0][3]*j[1] for j in axis_dict[i]])/weight]
                else:
                    self._error = '图片模糊:未识别出有效信息'
                    return 0
        if self._axis is None:
            self._axis = axis_true.copy()
        for i in axis_true:
            axis_true[i] = tuple(axis_true[i])
        
        address = ''
        for i in self._result_up[0]:
            h = (i[0][3][1]+i[0][2][1]-i[0][1][1]-i[0][0][1])/2
            w = (i[0][1][0]+i[0][2][0]-i[0][0][0]-i[0][3][0])/2
            h1 = min(max(i[0][3][1], i[0][2][1]), axis_true['user_address'][3])-max(min(i[0][0][1], i[0][1][1]), axis_true['user_address'][1])
            w1 = min(max(i[0][1][0], i[0][2][0]), axis_true['user_address'][2])-max(min(i[0][0][0], i[0][3][0]), axis_true['user_address'][0])
#             print(i,h1,w1,h1/h, w1/w, address,axis_true['user_address'])
            if h1/h>0.6 and w1/w>0.6:
#                 print(i[1][0],'\n')
                if len(address)==0:
                    for char in self._char_address:
                        if i[1][0].startswith(char):
                            address += (i[1][0][i[1][0].find(char)+len(char):]).strip()
                            self._axis['user_address'][1] = max(i[0][0][1], i[0][1][1])
                            self._axis['user_address'][2] = max(i[0][1][0], i[0][2][0])
                            break
                    if len(address)==0:
                        address += i[1][0]
                        self._axis['user_address'][1] = max(i[0][0][1], i[0][1][1])
                        self._axis['user_address'][2] = max(i[0][1][0], i[0][2][0])
                        fix_x.append(i[0][0][0])
                else:
                    address += i[1][0]
                    self._axis['user_address'][3] = max(i[0][2][1],i[0][3][1])
                    fix_x.append(i[0][0][0])
            if '图片模糊' in self._info['user_number']:
                if sum([1 for j in i[1][0][-18:] if j in '0123456789xX'])==18:
                    self._info['user_number'] = i[1][0][-18:]
                    self._info['user_sex'] =  '男' if int(self._info['user_number'][16])%2 else '女'
                    self._info['user_born'] = f"{self._info['user_number'][6:10]}年{int(self._info['user_number'][10:12])}月{int(self._info['user_number'][12:14])}日"
            if '图片模糊' in self._info['user_nation']:
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
            if '图片模糊' in self._info['user_name']:
                h1 = min(max(i[0][3][1], i[0][2][1]), axis_true['user_name'][3])-max(min(i[0][0][1], i[0][1][1]), axis_true['user_name'][1])
                w1 = min(max(i[0][1][0], i[0][2][0]), axis_true['user_name'][2])-max(min(i[0][0][0], i[0][3][0]), axis_true['user_name'][0])            
                if h1/h>0.6 and w1/w>0.6:
                    if len(i[1][0])>3 and i[1][0].find('名')==1:
                        self._info['user_name'] = i[1][0][2:]
                        self._axis['user_name'] = [self._axis['user_name'][0], i[0][0][1]]+i[0][2]
                    elif i[1][0].startswith('名') and len(i[1][0])>2:
                        self._info['user_name'] = i[1][0][1:]
                        self._axis['user_name'] = [self._axis['user_name'][0], i[0][0][1]]+i[0][2]
                    elif len(i[1][0])>1:
                        self._info['user_name'] = i[1][0]
                        self._axis['user_name'] = [self._axis['user_name'][0], i[0][0][1]]+i[0][2]
                        fix_x.append(i[0][0][0])
        if '图片模糊' in self._info['user_address'] or self._info['user_address']=='':
            self._info['user_address'] = address
        if '图片模糊' in self._info['user_nation']:
            self._info['user_nation'] = '汉'
        
        
        if len(fix_x)>0:
            fix_x = sum(fix_x)/len(fix_x)
            self._axis['user_name'][0] = fix_x
            self._axis['user_sex'][0] = fix_x
            self._axis['user_born'][0] = fix_x
            self._axis['user_address'][0] = fix_x
        
        fix_y = []
        for i in self._result_up[0]:
            if '性别' in i[1][0]:
                fix_y.append(min(i[0][0][1], i[0][1][1]))
            elif sum([1 for char in self._char_nation if char in i[1][0]])==1:
                fix_y.append(min(i[0][0][1], i[0][1][1]))
        if len(fix_y)>0:
            fix_y = sum(fix_y)/len(fix_y)
            self._axis['user_sex'][1] = fix_y
            self._axis['user_nation'][1] = fix_y
        
        self._axis['user_sex'][2] = self._axis['user_address'][0]+(self._axis['user_address'][2]-self._axis['user_address'][0])/11*1.5
        self._axis['user_born'][2] = min(self._axis['user_born'][2], self._axis['user_address'][2])
        
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
        for i in self._axis:
            self._axis[i] = [int(max(0, j)) for j in self._axis[i]]
    
    def _axis_transform_down(self):
        if len(self._result_down)==0:
            return 0
        for i in self._result_down[0]:
            if '居住证' in i[1][0]:
                self._info['user_type'] = i[1][0]
                break
        
        for i in self._result_down[0]:
            if '公安局' in i[1][0] or '分局' in i[1][0]:
                self._info['user_organization'] = i[1][0]
                for char in self._char_organization:
                    if char in i[1][0]:
                        if (i[1][0][i[1][0].find(char)+len(char):]).strip()!='':
                            self._info['user_organization'] = (i[1][0][i[1][0].find(char)+len(char):]).strip()
                            break
                break
        for i in ['公委局', '公农局']:
            self._info['user_organization'] = self._info['user_organization'].replace(i, '公安局')
        
        for i in self._result_down[0]:
            if sum([1 for char in ['长期', '.', '-', '一'] if char in i[1][0]])>1:
                if sum([1 for char in self._char_number if char in i[1][0]])>1:
                    if sum([1 for j in i[1][0][-21:] if j in '0123456789.-一'])==21:
                        self._info['user_validity_period'] = i[1][0][-21:].replace('一', '-')
                        break
                    elif i[1][0].endswith('长期'):
                        if sum([1 for j in i[1][0][-13:] if j in '0123456789.-长期'])==13:
                            self._info['user_validity_period'] = i[1][0][-13:]
                            break
                        else:
                            temp = i[1][0]
                            for j in ['.一:-']:
                                temp = temp.replace(j, '')
                            for j in temp:
                                if j in self._char_number:
                                    break
                                else:
                                    temp = temp[temp.find(j)+1:]
                            if len(temp)==10:
                                self._info['user_validity_period'] = f'{temp[:4]}.{temp[4:6]}.{temp[6:8]}-长期'
                    else:
                        temp = i[1][0]
                        for j in '.一:-':
                            temp = temp.replace(j, '')
                        for j in temp:
                            if j in self._char_number:
                                break
                            else:
                                temp = temp[temp.find(j)+1:]
                        if len(temp)==16:
                            self._info['user_validity_period'] = f'{temp[:4]}.{temp[4:6]}.{temp[6:8]}-{temp[8:12]}.{temp[12:14]}.{temp[14:16]}'
    
    def draw_mask(self, image=None, axis=None, box_axis='all', mask_axis=None):
        if image is None:
            image = self._image.copy()
        angle = self._angle_up if axis is None else axis['angle']
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



