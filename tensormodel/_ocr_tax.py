from collections import defaultdict

import cv2
import paddleocr
import linora as la

__all__ = ['OCRTaxCertificate']



class OCRTaxCertificate():
    def __init__(self, ocr=None, remark_function=None):
        self.ocr = paddleocr.PaddleOCR(show_log=False) if ocr is None else ocr
        self.remark_function = remark_function
        self._keys = []
        self._char_tax_id = ['No']
        self._char_tax_date = ['填发日期']
        self._char_tax_organ = ['税务机关']
        self._char_tax_user_id = ['纳税人识别号']
        self._char_tax_user_name = ['纳税人名称']
        self._char_tax_amount = ['金额合计']
        self._char_tax_ticket_filler = ['填票人']
        self._char_number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        
    def predict(self, image, axis=False, ocr_result=None):
        self._axis = None
        self._show_axis = axis
        self._error = 'ok'
        self._angle = -1
        self._mode = ''
        if ocr_result is not None:
            self._result = ocr_result
            self._direction_transform(image, use_ocr_result=True)
            self._axis_transform()
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
            self._axis_transform()
            for i in self._info:
                if '图片模糊' in self._info[i]:
                    self._temp_info = self._info.copy()
                    self._temp_axis = self._axis.copy()
                    self._direction_transform(la.image.enhance_brightness(self._image, 0.6))
                    self._axis_transform()
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
                
                rank = [0,0,0,0,0,0]
                for r, i in enumerate(result[0], start=1):
                    if '填发日期' in i[1][0] or '税务机关' in i[1][0]:
                        rank[0] = r
                    elif '纳税人识别号' in i[1][0] or '纳税人名称' in i[1][0]:
                        rank[1] = r
                    elif '原凭证号' in i[1][0] or '品目名称' in i[1][0] or '税款所属时期' in i[1][0]:
                        rank[2] = r
                    elif '实缴(退)金额' in i[1][0] or '入(退)库日期' in i[1][0]:
                        rank[2] = r
                    elif '金额合计' in i[1][0]:
                        rank[3] = r
                    elif '填票人' in i[1][0]:
                        rank[4] = r
                    elif '妥善保管' in i[1][0]:
                        rank[5] = r
                rank = [i for i in rank if i>0]
                if rank==sorted(rank) and len(rank)>1:
                    self._result = result.copy()
                    self._angle = angle
                    self._keys = ['tax_id', 'tax_date', 'tax_organ', 'tax_user_id', 'tax_user_name', 
                                  'tax_class', 'tax_amount', 'tax_ticket_filler']
                    if self.remark_function is not None:
                        self._keys.append('tax_remark')
                    break
                    
        self._info = {}
        if self._angle!=-1:
            self._info['tax_id'] = '图片模糊:未识别出税票编号'
            self._info['tax_date'] = '图片模糊:未识别出填发日期'
            self._info['tax_organ'] = '图片模糊:未识别出税务机关'
            self._info['tax_user_id'] = '图片模糊:未识别出纳税人识别号'
            self._info['tax_user_name'] = '图片模糊:未识别出纳税人名称'
            self._info['tax_class'] = '图片模糊:未识别出税种'
            self._info['tax_amount'] = '图片模糊:未识别出金额合计'
            self._info['tax_ticket_filler'] = '图片模糊:未识别出填票人'
            if self.remark_function is not None:
                self._info['tax_remark'] = '图片模糊:未识别出备注内容'
        else:
            self._info = '图片模糊:未识别出有效信息'
            self._error = '图片模糊:未识别出有效信息'
    
    def _axis_transform(self):
        if len(self._result)==0:
            return 0
#         fix_x = []
        axis_true = defaultdict(list)
        axis_dict = defaultdict(list)

        for i in self._result[0]:
            h = (i[0][3][1]+i[0][2][1]-i[0][1][1]-i[0][0][1])/2
            w = (i[0][1][0]+i[0][2][0]-i[0][0][0]-i[0][3][0])/2
            x = min(i[0][0][0], i[0][3][0])
            y = min(i[0][0][1], i[0][1][1])
            if 'tax_id' not in axis_true:
                for char in self._char_tax_id:
                    if char in i[1][0]:
                        axis_true['tax_id'] = i[0][0]+i[0][2]
                        axis_dict['tax_organ'].append(([x+w*0.3, y+h, x+w*1.8, y+h*2.8], 0.8))
                        axis_dict['tax_date'].append(([x-w*1.8, y+h*2, x-w*0.5, y+h*3], 0.6))
                        axis_dict['tax_user_name'].append(([x-w*0.4, y+h*3, x+w*1.4, y+h*5.5], 0.6))
                        break
                if 'tax_id' in axis_true:
                    continue
            if 'tax_date' not in axis_true:
                for char in self._char_tax_date:
                    if char in i[1][0]:
                        axis_true['tax_date'] = [x+w*1.2, y, x+w*4, y+h]
                        axis_dict['tax_organ'].append(([x+w*6, y-h, x+w*9.5, y+h], 0.6))
                        axis_dict['tax_user_name'].append(([x+w*4, y+h, x+w*7, y+h*3.5], 0.8))
                        axis_dict['tax_user_id'].append(([x-w*2, y+h, x+w*2.4, y+h*3.5], 0.8))
                        break
                if 'tax_date' in axis_true:
                    continue
            if 'tax_organ' not in axis_true:
                for char in self._char_tax_organ:
                    if char in i[1][0]:
                        axis_true['tax_organ'] = [x+w, y-h, x+w*5, y+h]
                        axis_dict['tax_id'].append(([x+w*0.3, y-h*2, x+w*3, y-h], 0.8))
                        axis_dict['tax_date'].append(([x-w*3.5, y, x-w, y+h], 0.8))
                        axis_dict['tax_user_name'].append(([x-w, y+h, x+w*2, y+h*3.5], 0.8))
                        axis_dict['tax_user_id'].append(([x-w*7, y+h, x-w*2.5, y+h*3.5], 0.6))
                        break
                if 'tax_organ' in axis_true:
                    continue
            if 'tax_user_id' not in axis_true:
                for char in self._char_tax_user_id:
                    if char in i[1][0]:
                        axis_true['tax_user_id'] = [x+w*1.2, y, x+w*4.5, y+h]
                        axis_dict['tax_date'].append(([x+w*3.5, y-h*2, x+w*6, y-h], 0.8))
                        axis_dict['tax_organ'].append(([x+w*7.2, y-h*3, x+w*10.2, y-h], 0.6))
                        axis_dict['tax_user_name'].append(([x+w*6, y, x+w*8, y+h], 0.6))
                        axis_dict['tax_class'].append(([x+w*1.5, y+h*4, x+w*3, y+h*13], 0.6))
                        break
                if 'tax_user_id' in axis_true:
                    continue
            if 'tax_user_name' not in axis_true:
                for char in self._char_tax_user_name:
                    if char in i[1][0]:
                        axis_true['tax_user_name'] = [x+w*1.2, y, x+w*4, y+h]
                        axis_dict['tax_date'].append(([x-w*1.5, y-h*2, x+w, y-h], 0.8))
                        axis_dict['tax_organ'].append(([x+w*2.5, y-h*2.5, x+w*6, y-h], 0.8))
                        axis_dict['tax_user_id'].append(([x-w*4, y, x-w*0.5, y+h], 0.8))
                        axis_dict['tax_class'].append(([x-w*3.8, y+h*4, x-w*2, y+h*13], 0.6))
                        break
                if 'tax_user_name' in axis_true:
                    continue
            if '原凭证号'==i[1][0]:
                axis_dict['tax_class'].append(([x+w*1.3, y+h*2, x+w*3, y+h*11], 0.8))
                continue
            if '品目名称'==i[1][0]:
                axis_dict['tax_class'].append(([x-w*2, y+h*2, x-w*0.5, y+h*11], 0.8))
                continue
            if 'tax_amount' not in axis_true:
                for char in self._char_tax_amount:
                    if char in i[1][0]:
                        axis_true['tax_amount'] = [x+w*12.5, y, x+w*14, y+h]
                        axis_dict['tax_ticket_filler'].append(([x+w*4.5, y+h*4, x+w*6.5, y+h*5], 0.6))
                        if self.remark_function is not None:
                            axis_dict['tax_remark'].append(([x+w*7, y+h*2, x+w*14, y+h*9], 0.6))
                        break
                if 'tax_amount' in axis_true:
                    continue
            if 'tax_ticket_filler' not in axis_true:
                for char in self._char_tax_ticket_filler:
                    if char in i[1][0]:
                        axis_true['tax_ticket_filler'] = [x-w*0.5, y+h, x+w*1.5, y+h*3]
                        axis_dict['tax_amount'].append(([x+w*5.5, y-h*3.5, x+w*7.5, y-h*2], 0.6))
                        if self.remark_function is not None:
                            axis_dict['tax_remark'].append(([x+w*1.5, y-h*2, x+w*7.5, y+h*5], 0.6))
                        break
                if 'tax_ticket_filler' in axis_true:
                    continue
                    
#         print(axis_true, axis_dict)
        for i in self._keys:
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
        
        tax_date = ''
        tax_organ = ''
        tax_class = []
        if self.remark_function is not None:
            tax_remark = ''
        for i in self._result[0]:
            h = (i[0][3][1]+i[0][2][1]-i[0][1][1]-i[0][0][1])/2
            w = (i[0][1][0]+i[0][2][0]-i[0][0][0]-i[0][3][0])/2
            if h==0:
                h = 1
            if w==0:
                w = 1
            x = min(i[0][0][0], i[0][3][0])
            y = min(i[0][0][1], i[0][1][1])
            if '图片模糊' in self._info['tax_id'] and 'tax_id' in axis_true:
                h1 = min(max(i[0][3][1], i[0][2][1]), axis_true['tax_id'][3])-max(min(i[0][0][1], i[0][1][1]), axis_true['tax_id'][1])
                w1 = min(max(i[0][1][0], i[0][2][0]), axis_true['tax_id'][2])-max(min(i[0][0][0], i[0][3][0]), axis_true['tax_id'][0])            
                if h1/h>0.6 and w1/w>0.6:
                    self._info['tax_id'] = i[1][0]
                    self._axis['tax_id'] = [x, y]+i[0][2]
                if '图片模糊' not in self._info['tax_id']:
                    continue
            if 'tax_organ' in axis_true:
                h1 = min(max(i[0][3][1], i[0][2][1]), axis_true['tax_organ'][3])-max(min(i[0][0][1], i[0][1][1]), axis_true['tax_organ'][1])
                w1 = min(max(i[0][1][0], i[0][2][0]), axis_true['tax_organ'][2])-max(min(i[0][0][0], i[0][3][0]), axis_true['tax_organ'][0])            
                if h1/h>0.6 and w1/w>0.6:
                    if '税务机关' not in i[1][0]:
                        tax_organ += i[1][0]
                        self._axis['tax_organ'] = [x, y]+i[0][2]
                    else:
                        tax_organ += i[1][0].replace(' ', '').replace('：', ':').split(':')[-1]
                        self._axis['tax_organ'][3] = i[0][2][1]
                    continue
                if '税务机关' in i[1][0]:
                    tax_organ += i[1][0].replace(' ', '').replace('：', ':').split(':')[-1]
                    self._axis['tax_organ'][3] = i[0][2][1]
                    continue
            if 'tax_date' in axis_true:
                h1 = min(max(i[0][3][1], i[0][2][1]), axis_true['tax_date'][3])-max(min(i[0][0][1], i[0][1][1]), axis_true['tax_date'][1])
                w1 = min(max(i[0][1][0], i[0][2][0]), axis_true['tax_date'][2])-max(min(i[0][0][0], i[0][3][0]), axis_true['tax_date'][0])            
                if h1/h>0.6 and w1/w>0.6:
                    tax_date += i[1][0]
#                     self._axis['tax_date'] = [x, y]+i[0][2]
                    continue
            if '图片模糊' in self._info['tax_user_id'] and 'tax_user_id' in axis_true:
                h1 = min(max(i[0][3][1], i[0][2][1]), axis_true['tax_user_id'][3])-max(min(i[0][0][1], i[0][1][1]), axis_true['tax_user_id'][1])
                w1 = min(max(i[0][1][0], i[0][2][0]), axis_true['tax_user_id'][2])-max(min(i[0][0][0], i[0][3][0]), axis_true['tax_user_id'][0])            
                if h1/h>0.6 and w1/w>0.6:
                    self._info['tax_user_id'] = i[1][0]
                    self._axis['tax_user_id'] = [x, y]+i[0][2]
                if '图片模糊' not in self._info['tax_user_id']:
                    continue
            if '图片模糊' in self._info['tax_user_name'] and 'tax_user_name' in axis_true:
                h1 = min(max(i[0][3][1], i[0][2][1]), axis_true['tax_user_name'][3])-max(min(i[0][0][1], i[0][1][1]), axis_true['tax_user_name'][1])
                w1 = min(max(i[0][1][0], i[0][2][0]), axis_true['tax_user_name'][2])-max(min(i[0][0][0], i[0][3][0]), axis_true['tax_user_name'][0])            
                if h1/h>0.6 and w1/w>0.6:
                    self._info['tax_user_name'] = i[1][0]
                    self._axis['tax_user_name'] = [x, y]+i[0][2]
                if '图片模糊' not in self._info['tax_user_name']:
                    continue
            if '图片模糊' in self._info['tax_class'] and 'tax_class' in axis_true:
                h1 = min(max(i[0][3][1], i[0][2][1]), axis_true['tax_class'][3])-max(min(i[0][0][1], i[0][1][1]), axis_true['tax_class'][1])
                w1 = min(max(i[0][1][0], i[0][2][0]), axis_true['tax_class'][2])-max(min(i[0][0][0], i[0][3][0]), axis_true['tax_class'][0])            
                if h1/h>0.6 and w1/w>0.1 and '税' in i[1][0]:
                    tax_class.append(''.join([j for j in i[1][0] if j not in self._char_number]))
#                     self._axis['tax_class'] = [x, y]+i[0][2]
                if '图片模糊' not in self._info['tax_class']:
                    continue
            if '图片模糊' in self._info['tax_amount'] and 'tax_amount' in axis_true:
                h1 = min(max(i[0][3][1], i[0][2][1]), axis_true['tax_amount'][3])-max(min(i[0][0][1], i[0][1][1]), axis_true['tax_amount'][1])
                w1 = min(max(i[0][1][0], i[0][2][0]), axis_true['tax_amount'][2])-max(min(i[0][0][0], i[0][3][0]), axis_true['tax_amount'][0])            
                if h1/h>0.6 and w1/w>0.6:
                    self._info['tax_amount'] = i[1][0]
                    self._axis['tax_amount'] = [x, y]+i[0][2]
                if '图片模糊' not in self._info['tax_amount']:
                    continue
            if '图片模糊' in self._info['tax_ticket_filler'] and 'tax_ticket_filler' in axis_true:
                h1 = min(max(i[0][3][1], i[0][2][1]), axis_true['tax_ticket_filler'][3])-max(min(i[0][0][1], i[0][1][1]), axis_true['tax_ticket_filler'][1])
                w1 = min(max(i[0][1][0], i[0][2][0]), axis_true['tax_ticket_filler'][2])-max(min(i[0][0][0], i[0][3][0]), axis_true['tax_ticket_filler'][0])            
                if h1/h>0.6 and w1/w>0.6:
                    self._info['tax_ticket_filler'] = i[1][0]
                    self._axis['tax_ticket_filler'] = [x, y]+i[0][2]
                if '图片模糊' not in self._info['tax_ticket_filler']:
                    continue
            if 'tax_remark' in axis_true:
                h1 = min(max(i[0][3][1], i[0][2][1]), axis_true['tax_remark'][3])-max(min(i[0][0][1], i[0][1][1]), axis_true['tax_remark'][1])
                w1 = min(max(i[0][1][0], i[0][2][0]), axis_true['tax_remark'][2])-max(min(i[0][0][0], i[0][3][0]), axis_true['tax_remark'][0])            
                if h1/h>0.6 and w1/w>0.6:
                    tax_remark += i[1][0]
#                     self._axis['tax_date'] = [x, y]+i[0][2]
                    continue

        if '图片模糊' in self._info['tax_date'] and tax_date!='':
            self._info['tax_date'] = tax_date
        if '图片模糊' in self._info['tax_organ'] and tax_organ!='':
            self._info['tax_organ'] = tax_organ
        tax_class = '|'.join(tax_class)
        if '图片模糊' in self._info['tax_class'] and tax_class!='':
            self._info['tax_class'] = tax_class
        if 'tax_remark' in axis_true and tax_remark!='':
            self._info['tax_remark'] = self.remark_function(tax_remark)

#         try:
#             if len(fix_x)>0:
#                 fix_x = sum(fix_x)/len(fix_x)
#                 self._axis['household_type'][0] = fix_x
#                 self._axis['household_id'][0] = fix_x
#         except:
#             pass
    
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

def remark(remark):
    s = remark.replace('：',':').replace('，',',')
    try:
        address = s[s.find('地址')+2:s.find('权属')]
        if address.startswith(':'):
            address = address[1:]
    except:
        address = ''
    try:
        amount = s[s.find('计税金额'):]
        for i in ['共有人', '房源编号', '房屋产权证书号']:
            if i in amount:
                amount = amount[:amount.find(i)]
    except:
        amount = ''
    return {'tax_remark_address':address, 'tax_remark_amount':amount}