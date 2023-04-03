from collections import defaultdict

import paddleocr
import linora as la


__all__ = ['OCRIDCard']


class OCRIDCard():
    def __init__(self):
        self.ocr = paddleocr.PaddleOCR(show_log=False)
        self._error = 'ok'
        self._char_name = [i+j for i in ['姓', '娃', '妇', '性'] for j in ['名', '容', '吉']]
        self._char_nation = ['民族', '民旅', '民康', '民旗', '民路', '昆旗']
        self._char_address = ['住址', '佳址', '主址', '住 址', '往址', '生址', '佳道']
        self._char_organization = ['签发机关', '鑫发机关', '金设机关', '签发物关']
        self._char_number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        
    def predict(self, image, axis=False):
        if isinstance(image, str):
            image = la.image.read_image(image)
            self._image = la.image.color_convert(image)
        self._direction_transform(self._image)
        if isinstance(self._info, str):
            self._direction_transform(la.image.enhance_brightness(self._image, 0.8))
        if isinstance(self._info, str):
            return self._info
        self._axis_transform_up()
        self._axis_transform_down()
        for i in self._info:
            if '图片模糊' in self._info[i]:
                self._temp = self._info.copy()
                self._direction_transform(la.image.enhance_brightness(self._image, 0.8))
                self._axis_transform_up()
                self._axis_transform_down()
                if isinstance(self._info, str):
                    self._info = self._temp.copy()
                else:
                    for j in self._temp:
                        if '图片模糊' not in self._temp[j]:
                            self._info[j] = self._temp[j]
                break

        if axis:
            if len(self._result_up)==0:
                return {'info':self._info}
            else:
                return {'info':self._info, 'axis':self._axis, 'angle_up':self._angle_up}
        return self._info
        
    def _direction_transform(self, image):
        state_up = False
        state_down = False
        self._result_up = []
        self._result_down = []
        for angle in [0, 90, 180, 270]:
            if angle>0:
                image1 = la.image.rotate(image, angle, expand=True)
                image1 = la.image.image_to_array(image1)
            else:
                image1 = la.image.image_to_array(image)
            result = ocr.ocr(image1, cls=False)
            
            if not state_up:
                rank = [0,0,0,0,0]
                for r, i in enumerate(result[0], start=1):
                    if sum([1 for char in self._char_name if char in i[1][0]]):
                        rank[0] = r
                    elif sum([1 for char in self._char_nation if char in i[1][0]]) or '性别' in i[1][0]:
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
                    self._angle_up = angle
            
            if state_down and state_up:
                break
        
        self._info = {}
        if state_up:
            self._info['user_name'] = '图片模糊:未识别出姓名'
            self._info['user_sex'] = '图片模糊:未识别出性别'
            self._info['user_nation'] = '图片模糊:未识别出民族'
            self._info['user_born'] = '图片模糊:未识别出出生日期'
            self._info['user_address'] = '图片模糊:未识别出地址'
            self._info['user_number'] = '图片模糊:未识别出身份证号码'
        if state_down:
            self._info['user_type'] = '图片模糊:未识别出身份证类型'
            self._info['user_organization'] = '图片模糊:未识别出签发机关'
            self._info['user_validity_period'] = '图片模糊:未识别出有效期限'
        if state_up == state_down == False:
            self._info = '图片模糊:未识别出有效信息'
            self._error = '图片模糊:未识别出有效信息'
    
    def _axis_transform_up(self):
        if len(self._result_up)==0:
            return 0
        axis_true = {}
        axis_dict = defaultdict(list)
        
#         height = [(i[0][3][1]+i[0][2][1]-i[0][1][1]-i[0][0][1])/2 for i in self._result_up[0]]
#         height = sum(height)/len(height)
        
        for i in self._result_up[0]:
            height = (i[0][3][1]+i[0][2][1]-i[0][1][1]-i[0][0][1])/2
            width = (i[0][1][0]+i[0][2][0]-i[0][0][0]-i[0][3][0])/2
            if sum([1 for char in self._char_name if i[1][0].startswith(char)]):
                w = width/(len(i[1][0])+1) if len(i[1][0])==2 else width/(len(i[1][0])+2.5)
                h = height if len(i[1][0])==2 else height*0.7
                x = i[0][0][0]
                y = i[0][0][1] if len(i[1][0])==2 else i[0][0][1]+height*0.3
                axis_true['user_name'] = [x+w*3.5, y-h, x+w*10, y+h*1.5]
                axis_dict['user_sex'].append(([x+w*3.5, y+h*2, x+w*6, y+h*3.5], 0.8))
                axis_dict['user_nation'].append(([x+w*11.5, y+h*2, x+w*15, y+h*3.5], 0.8))
                axis_dict['user_born'].append(([x+w*3.5, y+h*4.5, x+w*18, y+h*6.5], 0.6))
                axis_dict['user_address'].append(([x+w*3.5, y+h*6, x+w*21.5, y+h*10], 0.4))
                axis_dict['user_face'].append(([x+w*21.5, y-h*0.5, x+w*31.5, y+h*14], 0.8))
                axis_dict['user_card'].append(([x-w*5, y-h*4, x+w*34.5, y+h*20], 0.8))
            elif i[1][0].startswith('性别') and len(i[1][0])<4:
                w = width/(len(i[1][0])+1) if len(i[1][0])==2 else width/(len(i[1][0])+2.5)
                h = height if len(i[1][0])==2 else height*0.75
                x = i[0][0][0]
                y = i[0][0][1]
                axis_dict['user_name'].append(([x+w*3.5, y-h*3.5, x+w*10, y-h], 0.8))
                axis_true['user_sex'] = [x+w*3.5, y, x+w*6, y+h*1.5]
                axis_dict['user_nation'].append(([x+w*11.5, y, x+w*15, y+h*1.5], 0.8))
                axis_dict['user_born'].append(([x+w*3.5, y+h*2, x+w*18, y+h*4], 0.8))
                axis_dict['user_address'].append(([x+w*3.5, y+h*4, x+w*21.5, y+h*8], 0.6))
                axis_dict['user_face'].append(([x+w*21.5, y-h*3, x+w*31.5, y+h*11.5], 0.8))
                axis_dict['user_card'].append(([x-w*5, y-h*5.5, x+w*34.5, y+h*17.5], 0.8))
            elif sum([1 for char in self._char_nation if i[1][0].startswith(char)])==1:
                w = width/(len(i[1][0])+1) if len(i[1][0])==2 else width/(len(i[1][0])+2)
                h = height if len(i[1][0])==2 else height*0.75
                x = i[0][0][0]
                y = i[0][0][1]
                axis_dict['user_name'].append(([x-w*5.5, y-h*3.5, x+w*4, y-h], 0.8))
                axis_dict['user_sex'].append(([x-w*5.5, y, x-w*2, y+h*1.5], 0.8))
                axis_true['user_nation'] = [x+w*3.5, y, x+w*7, y+h*1.5]
                axis_dict['user_born'].append(([x-w*5.5, y+h*1.5, x+w*9, y+h*4], 0.8))
                axis_dict['user_address'].append(([x-w*5.5, y+h*4, x+w*12, y+h*8], 0.6))
                axis_dict['user_face'].append(([x+w*12, y-h*3, x+w*22, y+h*11.5], 0.8))
                axis_dict['user_card'].append(([x-w*14, y-h*5.5, x+w*25, y+h*17.5], 0.8))
            elif '出生'==i[1][0]:
                w = width/(len(i[1][0])+1)
                h = height
                x = i[0][0][0]
                y = i[0][0][1]
                axis_dict['user_name'].append(([x+w*3.5, y-h*5.5, x+w*10, y-h*3], 0.6))
                axis_dict['user_sex'].append(([x+w*3.5, y-h*3, x+w*6, y-h], 0.8))
                axis_dict['user_nation'].append(([x+w*11.5, y-h*3, x+w*15, y-h], 0.8))
                axis_true['user_born'] = [x+w*3.5, y-h*0.5, x+w*18, y+h*0.5]
                axis_dict['user_address'].append(([x+w*3.5, y+h*1.5, x+w*21.5, y+h*5.5], 0.8))
                axis_dict['user_face'].append(([x+w*21.5, y-h*5, x+w*31.5, y+h*9], 0.8))
                axis_dict['user_card'].append(([x-w*5, y-h*8, x+w*34.5, y+h*15], 0.8))
            elif sum([1 for char in self._char_address if i[1][0].startswith(char)])==1:
                w = width/(len(i[1][0])+1) if len(i[1][0])==2 else width/(len(i[1][0])+2.5)
                h = height if len(i[1][0])==2 else height*0.75
                x = i[0][0][0]
                y = i[0][0][1]
                axis_dict['user_name'].append(([x+w*3.5, y-h*8, x+w*10, y-h*5.5], 0.4))
                axis_dict['user_sex'].append(([x+w*3.5, y-h*5.5, x+w*6, y-h*3.5], 0.6))
                axis_dict['user_nation'].append(([x+w*11.5, y-h*5.5, x+w*15, y-h*3.5], 0.6))
                axis_dict['user_born'].append(([x+w*3.5, y-h*3.5, x+w*18, y-h], 0.8))
                axis_true['user_address'] = [x+w*3.2, y-h*0.5, x+w*21.5, y+h*4.5]
                axis_dict['user_face'].append(([x+w*21.5, y-h*8, x+w*31.5, y+h*5.5], 0.8))
                axis_dict['user_card'].append(([x-w*5, y-h*13, x+w*34.5, y+h*12], 0.8))
            elif len(i[1][0])==18 or '号码' in i[1][0] or '公民' in i[1][0]:
                if sum([1 for j in i[1][0] if j in '0123456789xX'])==18:
                    axis_true['user_number'] = [i[0][0][0], i[0][0][1], i[0][2][0], i[0][2][1]]
                elif i[1][0].startswith('公民') and len(i[1][0])>20:
                    axis_true['user_number'] = [i[0][0][0]+(i[0][2][0]-i[0][0][0])*0.35, 
                                                i[0][0][1], i[0][2][0], i[0][2][1]]
                elif i[1][0].startswith('号码') and len(i[1][0])>20:
                    axis_true['user_number'] = [i[0][0][0]+(i[0][2][0]-i[0][0][0])*0.18, 
                                                i[0][0][1], i[0][2][0], i[0][2][1]]

        for i in self._result_up[0]:
            if len(i[1][0]) in [9, 10, 11] and i[1][0].find('年')==4 and '月' in i[1][0] and i[1][0].endswith('日'):
                w = i[0][2][0]-i[0][0][0]
                h = i[0][2][1]-i[0][0][1]
                x = i[0][0][0]
                y = i[0][0][1]
                axis_true['user_born'] = [i[0][0][0], i[0][0][1], i[0][2][0], i[0][2][1]]
                axis_dict['user_address'].append(([x-w*0.1, y+h*2, x+w*1.3, y+h*6.5], 0.8))
                axis_dict['user_sex'].append(([x-w*0.1, y-h*2.5, x+w*0.2, y-h*0.8], 0.8))
                axis_dict['user_nation'].append(([x+w*0.65, y-h*2.5, x+w*0.85, y-h*0.8], 0.8))
                axis_dict['user_name'].append(([x-w*0.1, y-h*5.5, x+w*0.5, y-h*3.5], 0.8))
                axis_dict['user_face'].append(([x+w*1.25, y-h*4, x+w*2.2, y+h*6], 0.8))
                axis_dict['user_card'].append(([x-w*0.5, y-h*6.5, x+w*2.5, y+h*10.5], 0.8))
                break
                
        for i in ['user_number', 'user_name', 'user_sex', 'user_nation', 'user_born', 'user_address', 'user_face', 'user_card']:
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
        self._axis = axis_true
        
        for i in self._result_up[0]:
            height = (i[0][3][1]+i[0][2][1]-i[0][1][1]-i[0][0][1])/2
            width = (i[0][1][0]+i[0][2][0]-i[0][0][0]-i[0][3][0])/2
            if len(i[1][0])==18 or '号码' in i[1][0] or '公民' in i[1][0]:
                if len(i[1][0])>=18 and i[1][1]<0.65:
                    self._error = '图片模糊:身份证号码识别概率较低'
                    return 0
                if sum([1 for j in i[1][0][-18:] if j in '0123456789xX'])==18:
                    self._info['user_number'] = i[1][0][-18:]
                    self._info['user_sex'] =  '男' if int(self._info['user_number'][16])%2 else '女'
                    self._info['user_born'] = f"{self._info['user_number'][6:10]}年{int(self._info['user_number'][10:12])}月{int(self._info['user_number'][12:14])}日"
            else:
                for char in self._char_nation:
                    if char in i[1][0]:
                        if '汉' in i[1][0]:
                            self._info['user_nation'] = '汉'
                            break
                        elif (i[1][0][i[1][0].find(char)+2:]).strip()!='':
                            self._info['user_nation'] = (i[1][0][i[1][0].find(char)+2:]).strip()
                            break
        if '图片模糊' in self._info['user_nation']:
            self._info['user_nation'] = '汉'
        
        address = ''
        for i in self._result_up[0]:
            h = (i[0][3][1]+i[0][2][1]-i[0][1][1]-i[0][0][1])/2
            w = (i[0][1][0]+i[0][2][0]-i[0][0][0]-i[0][3][0])/2
            h1 = min(max(i[0][3][1], i[0][2][1]), axis_true['user_address'][3])-max(min(i[0][0][1], i[0][1][1]), axis_true['user_address'][1])
            w1 = min(max(i[0][1][0], i[0][2][0]), axis_true['user_address'][2])-max(min(i[0][0][0], i[0][3][0]), axis_true['user_address'][0])
#             print(i,h1/h, w1/w, address)
            if h1/h>0.6 and w1/w>0.6:
#                 print(i,'\n')
                if len(address)==0:
                    for char in self._char_address:
                        if i[1][0].startswith(char):
                            address += (i[1][0][i[1][0].find(char)+len(char):]).strip()
                            break
                    if len(address)==0:
                        address += i[1][0]
                else:
                    address += i[1][0]
        if address.strip()=='':
            self._error = '图片模糊:未识别出住址'
            return 0
        self._info['user_address'] = address

        for i in self._result_up[0]:
            if '图片模糊' not in self._info['user_name']:
                break
            for char in self._char_name:
                if i[1][0].startswith(char) and len(i[1][0])>len(char)+1:
                    self._info['user_name'] = (i[1][0][i[1][0].find(char)+len(char):]).strip()
                    break
            h = (i[0][3][1]+i[0][2][1]-i[0][1][1]-i[0][0][1])/2
            w = (i[0][1][0]+i[0][2][0]-i[0][0][0]-i[0][3][0])/2
            h1 = min(max(i[0][3][1], i[0][2][1]), axis_true['user_name'][3])-max(min(i[0][0][1], i[0][1][1]), axis_true['user_name'][1])
            w1 = min(max(i[0][1][0], i[0][2][0]), axis_true['user_name'][2])-max(min(i[0][0][0], i[0][3][0]), axis_true['user_name'][0])            
            if h1/h>0.6 and w1/w>0.6:
                if len(i[1][0])>3 and i[1][0].find('名')==1:
                    self._info['user_name'] = i[1][0][2:]
                    break
                elif i[1][0].startswith('名') and len(i[1][0])>2:
                    self._info['user_name'] = i[1][0][1:]
                    break
                elif len(i[1][0])>1:
                    self._info['user_name'] = i[1][0]
                    break
        if '图片模糊' in self._info['user_name']:
            self._error = '图片模糊:未识别出姓名'
            return 0
    
    def _axis_transform_down(self):
        if len(self._result_down)==0:
            return 0
        for i in self._result_down[0]:
            if '居住证' in i[1][0]:
                self._info['user_type'] = i[1][0]
            else:
                self._info['user_type'] = '居民身份证'
        
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
                        



