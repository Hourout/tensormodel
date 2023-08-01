from collections import defaultdict

import cv2
import paddleocr
import linora as la

__all__ = ['OCRWanShuiPiao']


class OCRWanShuiPiao():
    def __init__(self, ocr=None, remark_function=None):
        self.ocr = paddleocr.PaddleOCR(show_log=False) if ocr is None else ocr
        self._remark_function = remark_function
        self._keys = []
        self._char_tax_date = ['填发日期']
        self._char_tax_organ = ['税务机关']
        self._char_tax_user_id = ['纳税人识别号']
        self._char_tax_user_name = ['纳税人名称']
        self._char_tax_amount = [f'{i}{j}合计' for i in ['金','全'] for j in ['额','题','融']]
        self._char_tax_ticket_filler = ['填票人', '慎票人']
        self._char_number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        self._char_class = ['翼', '奥', '奖', '樊', '靓', '超', '美', '楚']
        self._char_name = '阎段冼凌辜石顾齐祝乐虞伏夏穆练赖宇姓苗甄于盖许彭晏敖谌汤梅娄潘李华花陈阮苟池陆韦鲍卜宗管沙覃糜傅周龙耿徐项蓝孟游廉楚扬樊奚竺龚毕章阳党全郁萧张成连漕舒柏卞时巫庄叶文冯霍臧喻揭应蔺唐芦谷席郎桂常单薛丛吴黎佟黄保施闵官盛商侯季宋瞿胡范牛畅蒋秦宫纪苏颜伍巩滕万江杨詹严程童燕百韩任辛关仇计司卓谈谢聂岳桑迟封吕梁袁祁卢冷匡毛温左罗白柳马孙薄家习安涂古邬康仲隋殷沿僧庞尤木牟排贾郭边金孔谭朱葛姜郑位费宁高赵贺陶崔向熊翁栗明银褚柯卫姚强莫戚汪鞠来乔魏符史丁钱廖曲稽吉栾冉井洪都蔡和兰翟欧何刘甘姬易屠胥靳屈嵺查鲁车蒙田戴窦冀景包焦衣沈申曾邱尹邢名裴董丘王邸岑郝解刁饶米柴路台艾原邓俞邹苑余倪方晋郜鄢植荆林杜钟邵武曹房简'
        
    def predict(self, image, axis=False, ocr_result=None):
        self._axis = None
        self._show_axis = axis
        self._error = 'ok'
        self._angle = -1
        self._mode = ''
        self._image_str = None
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
                self._fit_direction(la.image.enhance_brightness(self._image, 0.6))
            if isinstance(self._info, str):
                if self._show_axis:
                    return {'data':self._info, 'axis':[], 'angle':0, 'error':self._error}
                else:
                    return {'data':self._info, 'angle':0, 'error':self._error}
            ax = self._fit_axis()
            self._fit_characters(ax)
            for i in self._info:
                if '图片模糊' in self._info[i]:
                    self._temp_info = self._info.copy()
                    if self._show_axis:
                        self._temp_axis = self._axis.copy()
                    if self._image_str is not None and self._angle==0:
                        self._fit_direction(self._image_str)
                    else:
                        self._fit_direction(la.image.enhance_brightness(self._image, 0.75))
                    ax = self._fit_axis()
                    self._fit_characters(ax)
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
        for i in self._info:
            if '图片模糊' in self._info[i]:
                self._info[i] = ''
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
                
                rank = [0,0,0,0,0,0]
                for r, i in enumerate(result[0], start=1):
                    if '填发日期' in i[1][0] or '税务机关：' in i[1][0]:
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
                    self._keys = ['tax_date', 'tax_organ', 'tax_user_id', 'tax_user_name', 
                                  'tax_class', 'tax_amount', 'tax_ticket_filler']
                    if self._remark_function is not None:
                        self._keys.append('tax_remark')
                    break
                    
        self._info = {}
        if self._angle!=-1:
            self._info['tax_date'] = '图片模糊:未识别出填发日期'
            self._info['tax_organ'] = '图片模糊:未识别出税务机关'
            self._info['tax_user_id'] = '图片模糊:未识别出纳税人识别号'
            self._info['tax_user_name'] = '图片模糊:未识别出纳税人名称'
            self._info['tax_class'] = '图片模糊:未识别出税种'
            self._info['tax_amount'] = '图片模糊:未识别出金额合计'
            self._info['tax_ticket_filler'] = '图片模糊:未识别出填票人'
        else:
            self._info = '图片模糊:未识别出有效信息'
            self._error = '图片模糊:未识别出有效信息'
    
    def _fit_axis(self):
        if len(self._result)==0:
            return 0

        axis_true = defaultdict(list)
        axis_dict = defaultdict(list)
        tax_organ_logic = True
        for i in self._result[0]:
            h = (i[0][3][1]+i[0][2][1]-i[0][1][1]-i[0][0][1])/2
            w = (i[0][1][0]+i[0][2][0]-i[0][0][0]-i[0][3][0])/2
            x = min(i[0][0][0], i[0][3][0])
            y = min(i[0][0][1], i[0][1][1])
            if 'tax_date' not in axis_true:
                for char in self._char_tax_date:
                    if char in i[1][0]:
                        if len(i[1][0])>5:
                            w = w*(len(char)+1)/len(i[1][0])
                        axis_true['tax_date'] = [x+w*1.2, y-h*0.25, x+w*4, y+h*1.25]
                        axis_dict['tax_organ'].append(([x+w*6, y-h, x+w*9.5, y+h], 0.8))
                        axis_dict['tax_user_name'].append(([x+w*3.5, y+h, x+w*7, y+h*3.5], 0.8))
                        axis_dict['tax_user_id'].append(([x-w*2, y+h, x+w*2.4, y+h*3.5], 0.8))
                        break
                if 'tax_date' in axis_true:
                    continue
            if 'tax_organ' not in axis_true and tax_organ_logic:
                for char in self._char_tax_organ:
                    if char in i[1][0]:
                        if len(i[1][0])>4:
                            w = w*(len(char)+0.5)/len(i[1][0])
                        axis_true['tax_organ'] = [x+w, y-h, x+w*5, y+h]
                        axis_dict['tax_id'].append(([x+w*0.3, y-h*2, x+w*3, y-h], 0.8))
                        axis_dict['tax_date'].append(([x-w*3.5, y, x-w, y+h], 0.8))
                        axis_dict['tax_user_name'].append(([x-w, y+h, x+w*2, y+h*3.5], 0.8))
                        axis_dict['tax_user_id'].append(([x-w*7, y+h, x-w*2.5, y+h*3.5], 0.6))
                        break
                if 'tax_organ' in axis_true:
                    tax_organ_logic = False
                    continue
            if 'tax_user_id' not in axis_true:
                for char in self._char_tax_user_id:
                    if char in i[1][0]:
                        if len(i[1][0])>6:
                            w = w*(len(char)+0.5)/len(i[1][0])
                        axis_true['tax_user_id'] = [x+w, y-h*0.75, x+w*4.2, y+h*1.75]
                        axis_dict['tax_date'].append(([x+w*3.5, y-h*2, x+w*6, y-h], 0.8))
                        axis_dict['tax_organ'].append(([x+w*7.2, y-h*1.8, x+w*10.2, y-h*0.5], 0.6))
                        axis_dict['tax_user_name'].append(([x+w*5.5, y-h*0.75, x+w*8, y+h*1.75], 0.8))
                        axis_dict['tax_class'].append(([x+w*1.5, y+h*3.5, x+w*3, y+h*14], 0.6))
                        break
                if 'tax_user_id' in axis_true:
                    continue
            if 'tax_user_name' not in axis_true:
                for char in self._char_tax_user_name:
                    if char in i[1][0]:
                        if len(i[1][0])>5:
                            w = w*(len(char)+0.5)/len(i[1][0])
                        axis_true['tax_user_name'] = [x+w, y-h*0.75, x+w*4, y+h*1.75]
                        axis_dict['tax_date'].append(([x-w*1.5, y-h*2, x+w, y-h], 0.8))
                        axis_dict['tax_organ'].append(([x+w*2.5, y-h*1.8, x+w*6, y-h*0.5], 0.8))
                        axis_dict['tax_user_id'].append(([x-w*4, y, x-w*0.75, y+h*1.75], 0.8))
                        axis_dict['tax_class'].append(([x-w*3.8, y+h*3.5, x-w*2, y+h*14], 0.6))
                        break
                if 'tax_user_name' in axis_true:
                    continue
            if '原凭证号'==i[1][0]:
                axis_dict['tax_date'].append(([x+w*3.5, y-h*4, x+w*6, y-h*2.75], 0.6))
                axis_dict['tax_organ'].append(([x+w*7, y-h*4, x+w*10, y-h*2.75], 0.4))
                axis_dict['tax_user_id'].append(([x+w, y-h*3, x+w*4, y-h*0.5], 0.8))
                axis_dict['tax_user_name'].append(([x+w*5.5, y-h*3, x+w*8.5, y-h*0.5], 0.6))
                axis_dict['tax_class'].append(([x+w*1.25, y+h*1.5, x+w*3, y+h*12.5], 0.8))
                tax_organ_logic = False
                continue
            if '品目名称'==i[1][0]:
                axis_dict['tax_date'].append(([x, y-h*4, x+w*2.5, y-h*2.75], 0.6))
                axis_dict['tax_organ'].append(([x+w*4, y-h*4, x+w*7, y-h*2.75], 0.4))
                axis_dict['tax_user_id'].append(([x-w*2.3, y-h*3, x+w, y-h*0.5], 0.8))
                axis_dict['tax_user_name'].append(([x+w*2.3, y-h*3, x+w*5.5, y-h*0.5], 0.8))
                axis_dict['tax_class'].append(([x-w*2.25, y+h*1.5, x-w*0.5, y+h*12.5], 0.8))
                tax_organ_logic = False
                continue
            if '税款所属时期'==i[1][0]:
                axis_dict['tax_date'].append(([x-w*1.8, y-h*4, x+w*0.5, y-h*2.75], 0.6))
                axis_dict['tax_organ'].append(([x+w*2, y-h*4, x+w*4.5, y-h*2.75], 0.6))
                axis_dict['tax_user_id'].append(([x-w*4, y-h*3, x-w, y-h*0.5], 0.8))
                axis_dict['tax_user_name'].append(([x+w*0.5, y-h*3, x+w*3.5, y-h*0.5], 0.8))
                axis_dict['tax_class'].append(([x-w*3.75, y+h*1.5, x-w*2, y+h*12.5], 0.8))
                tax_organ_logic = False
                continue
            if '入（退）库日期'==i[1][0]:
                axis_dict['tax_date'].append(([x-w*3.5, y-h*4, x-w, y-h*2.75], 0.6))
                axis_dict['tax_organ'].append(([x+w*0.5, y-h*4, x+w*3, y-h*2.75], 0.6))
                axis_dict['tax_user_id'].append(([x-w*6, y-h*3, x-w*3.5, y-h*0.5], 0.8))
                axis_dict['tax_user_name'].append(([x-w, y-h*3, x+w*2, y-h*0.5], 0.8))
                axis_dict['tax_class'].append(([x-w*5.75, y+h*1.5, x-w*4, y+h*12.5], 0.8))
                tax_organ_logic = False
                continue
            if '实缴（退）金额'==i[1][0]:
                axis_dict['tax_date'].append(([x-w*2.5, y-h*4, x-w*4.5, y-h*2.75], 0.6))
                axis_dict['tax_organ'].append(([x-w*2, y-h*4, x+w*2, y-h*2.75], 0.6))
                axis_dict['tax_user_id'].append(([x-w*7, y-h*3, x-w*4, y-h*0.5], 0.8))
                axis_dict['tax_user_name'].append(([x-w*2.5, y-h*3, x+w, y-h*0.5], 0.8))
                axis_dict['tax_class'].append(([x-w*6.75, y+h*1.5, x-w*5, y+h*12.5], 0.8))
                tax_organ_logic = False
                continue
            if 'tax_amount' not in axis_true:
                for char in self._char_tax_amount:
                    if char in i[1][0]:
                        if len(i[1][0])>4:
                            w = w*(len(char)+0.5)/len(i[1][0])
                        axis_true['tax_amount'] = [x+w*11.5, y-h*0.75, x+w*13.5, y+h*1.75]
                        axis_dict['tax_ticket_filler'].append(([x+w*4.5, y+h*4, x+w*6.5, y+h*5], 0.6))
                        if self._remark_function is not None:
                            axis_dict['tax_remark'].append(([x+w*6.5, y+h*1.5, x+w*13, y+h*9], 0.6))
                        break
                if 'tax_amount' in axis_true:
                    tax_organ_logic = False
                    continue
            if 'tax_ticket_filler' not in axis_true:
                for char in self._char_tax_ticket_filler:
                    if char in i[1][0]:
                        axis_true['tax_ticket_filler'] = [x-w*0.5, y+h, x+w*1.5, y+h*3]
                        axis_dict['tax_amount'].append(([x+w*5.5, y-h*3.5, x+w*7.5, y-h*2], 0.4))
                        if self._remark_function is not None:
                            axis_dict['tax_remark'].append(([x+w*1.5, y-h*2, x+w*7.5, y+h*5], 0.8))
                        break
                if 'tax_ticket_filler' in axis_true:
                    continue

        for i in self._keys:
            if i not in axis_true:
                if i in axis_dict:
                    weight = sum([j[1] for j in axis_dict[i]])
                    axis_true[i] = [sum([j[0][0]*j[1] for j in axis_dict[i]])/weight,
                                    sum([j[0][1]*j[1] for j in axis_dict[i]])/weight,
                                    sum([j[0][2]*j[1] for j in axis_dict[i]])/weight,
                                    sum([j[0][3]*j[1] for j in axis_dict[i]])/weight]
        return axis_true
        
    def _fit_characters(self, axis):
        self._axis = axis.copy()
        axis_true = {i:tuple(axis[i]) for i in axis}
        
        tax_date = ''
        tax_organ = ''
        tax_class = []
        if self._remark_function is not None:
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
            if '图片模糊' in self._info['tax_organ'] and 'tax_organ' in axis_true:
                h1 = min(max(i[0][3][1], i[0][2][1]), axis_true['tax_organ'][3])-max(min(i[0][0][1], i[0][1][1]), axis_true['tax_organ'][1])
                w1 = min(max(i[0][1][0], i[0][2][0]), axis_true['tax_organ'][2])-max(min(i[0][0][0], i[0][3][0]), axis_true['tax_organ'][0])            

                if sum([1 for char in i[1][0] if char not in '0123456789'])>8:
                    if sum([1 for char in set(i[1][0]) if char in '国家税务局市区地方'])>3:
                        if sum([1 for char in i[1][0] if char not in '机关所办服厅：（）'])>8:
                            temp = i[1][0].replace(' ', '').replace('：', ':').split(':')[-1]
                            for char in self._char_tax_organ:
                                if char in temp:
                                    temp = temp[temp.find(char)+len(char):]
                                    continue
                            self._info['tax_organ'] = temp
                            self._axis['tax_organ'] = self._axis['tax_organ'][:2]+i[0][2]
                            continue
                
#                 if sum([1 for char in self._char_tax_organ if char in i[1][0]])>0:
#                     for char in self._char_tax_organ:
#                         if char in i[1][0]:
#                             temp = i[1][0][i[1][0].find(char)+len(char):]
#                             if len(temp.replace('：', '').replace('（', '').replace('）', ''))>8:
#                                 self._info['tax_organ'] = temp
#                                 self._axis['tax_organ'][3] = i[0][2][1]
#                                 continue
#                 if h1/h>0.45 and w1/w>0.6:
#                     if sum([1 for char in i[1][0] if char not in '0123456789'])>8:
#                         if sum([1 for char in i[1][0] if char in '国家税务局市区地方'])>3:
#                             if sum([1 for char in i[1][0] if char not in '机关所办服厅：（）'])>8:
#                                 self._info['tax_organ'] = i[1][0].replace(' ', '').replace('：', ':').split(':')[-1]
#                                 self._axis['tax_organ'][3] = i[0][2][1]
#                                 continue
            if 'tax_date' in axis_true:
                h1 = min(max(i[0][3][1], i[0][2][1]), axis_true['tax_date'][3])-max(min(i[0][0][1], i[0][1][1]), axis_true['tax_date'][1])
                w1 = min(max(i[0][1][0], i[0][2][0]), axis_true['tax_date'][2])-max(min(i[0][0][0], i[0][3][0]), axis_true['tax_date'][0])            
                if sum([1 for char in self._char_tax_date if char in i[1][0]])>0:
                    for char in self._char_tax_date:
                        if char in i[1][0]:
                            tax_date += ''.join([j for j in i[1][0][i[1][0].find(char)+len(char):] if j in '0123456789年月日'])
                            break
                    continue
                if h1/h>0.6 and w1/w>0.6:
                    temp = i[1][0].replace(' ', '').replace('：', '')
                    if len(temp)==sum([1 for char in temp if char in '0123456789年月日']):
                        if sum([1 for char in temp if char in '年月日']):
                            tax_date += temp
    #                       self._axis['tax_date'] = [x, y]+i[0][2]
                            continue
            if '图片模糊' in self._info['tax_user_id'] and 'tax_user_id' in axis_true:
                h1 = min(max(i[0][3][1], i[0][2][1]), axis_true['tax_user_id'][3])-max(min(i[0][0][1], i[0][1][1]), axis_true['tax_user_id'][1])
                w1 = min(max(i[0][1][0], i[0][2][0]), axis_true['tax_user_id'][2])-max(min(i[0][0][0], i[0][3][0]), axis_true['tax_user_id'][0])            
                if sum([1 for char in self._char_tax_user_id if char in i[1][0]])>0:
                    for char in self._char_tax_user_id:
                        if char in i[1][0] and len(i[1][0][i[1][0].find(char)+len(char):])>1:
                            self._info['tax_user_id'] = i[1][0][i[1][0].find(char)+len(char):].strip()
                            self._axis['tax_user_id'] = [self._axis['tax_user_id'][0], i[0][0][1]]+i[0][2]
                            break
                    continue
                if h1/h>0.6 and w1/w>0.6:
                    self._info['tax_user_id'] = i[1][0]
                    self._axis['tax_user_id'] = [x, y]+i[0][2]
                if '图片模糊' not in self._info['tax_user_id']:
                    continue
            if '图片模糊' in self._info['tax_user_name'] and 'tax_user_name' in axis_true:
                h1 = min(max(i[0][3][1], i[0][2][1]), axis_true['tax_user_name'][3])-max(min(i[0][0][1], i[0][1][1]), axis_true['tax_user_name'][1])
                w1 = min(max(i[0][1][0], i[0][2][0]), axis_true['tax_user_name'][2])-max(min(i[0][0][0], i[0][3][0]), axis_true['tax_user_name'][0])            
                if sum([1 for char in self._char_tax_user_name if char in i[1][0]])>0:
                    for char in self._char_tax_user_name:
                        if char in i[1][0] and len(i[1][0][i[1][0].find(char)+len(char):])>1:
                            temp = i[1][0][i[1][0].find(char)+len(char):].strip().replace(' ', '')
                            if len(temp)==4:
                                temp = temp[:2]+'|'+temp[2:]
                            elif len(temp)==5:
                                temp = temp[:3]+'|'+temp[3:] if temp[3] in self._char_name else temp[:2]+'|'+temp[2:]
                            elif len(temp)==6:
                                temp = temp[:3]+'|'+temp[3:]
                            self._info['tax_user_name'] = temp
                            self._axis['tax_user_name'] = [self._axis['tax_user_name'][0], i[0][0][1]]+i[0][2]
                            break
                    continue
                if h1/h>0.6 and w1/w>0.6 and len(i[1][0])>1 and sum([1 for char in i[1][0] if char in '税务机所：'])<1:
                    temp = i[1][0].strip().replace(' ', '')
                    if len(temp)==4:
                        temp = temp[:2]+'|'+temp[2:]
                    elif len(temp)==5:
                        temp = temp[:3]+'|'+temp[3:] if temp[3] in self._char_name else temp[:2]+'|'+temp[2:]
                    elif len(temp)==6:
                        temp = temp[:3]+'|'+temp[3:]
                    self._info['tax_user_name'] = temp
                    self._axis['tax_user_name'] = [x, y]+i[0][2]
                if '图片模糊' not in self._info['tax_user_name']:
                    continue
            if '图片模糊' in self._info['tax_class'] and 'tax_class' in axis_true:
                h1 = min(max(i[0][3][1], i[0][2][1]), axis_true['tax_class'][3])-max(min(i[0][0][1], i[0][1][1]), axis_true['tax_class'][1])
                w1 = min(max(i[0][1][0], i[0][2][0]), axis_true['tax_class'][2])-max(min(i[0][0][0], i[0][3][0]), axis_true['tax_class'][0])            
                temp = ''.join([j for j in i[1][0] if j not in '0123456789']).replace(' ', '').replace('臧', '城').replace('时', '附').replace('锐', '税')
                if h1/h>0.6 and w1/w>0.6 and temp!='税':
                    if '契税' in temp:
                        tax_class.append('契税')
                        continue
                    if temp.endswith('税') or temp.endswith('附加'):
                        if len(temp)==2 and temp[1]=='税':
                            temp = '契税'
                        tax_class.append(temp)
                        continue
                    if i[1][0].replace(' ', '')=='契':
                        tax_class.append('契税')
                        continue
                if i[0][0][1]>self._axis['tax_user_name'][1] and i[0][1][0]<self._axis['tax_class'][2] and i[0][2][1]<self._axis['tax_amount'][1]:
                    if temp.endswith('税') or temp.endswith('附加'):
                        if len(i[1][0])-len(temp)>12:
                            tax_class.append(temp)
                            continue
            if '图片模糊' in self._info['tax_amount'] and 'tax_amount' in axis_true:
                h1 = min(max(i[0][3][1], i[0][2][1]), axis_true['tax_amount'][3])-max(min(i[0][0][1], i[0][1][1]), axis_true['tax_amount'][1])
                w1 = min(max(i[0][1][0], i[0][2][0]), axis_true['tax_amount'][2])-max(min(i[0][0][0], i[0][3][0]), axis_true['tax_amount'][0])            
                if '￥' in i[1][0] or '¥' in i[1][0] or 'Y'==i[1][0].strip()[0] or 'X'==i[1][0].strip()[0]:
                    if i[1][0][1]!='0':
                        self._info['tax_amount'] = i[1][0].replace(' ', '')
                        self._axis['tax_amount'] = [x, y]+i[0][2]
                        continue
                if h1/h>0.6 and w1/w>0.6 and len(i[1][0])>2:
                    self._info['tax_amount'] = i[1][0].replace(' ', '')
                    self._axis['tax_amount'] = [x, y]+i[0][2]
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
                    continue
        
#         print(tax_date)
        tax_date = tax_date.replace(' ', '').replace('：', '')
        index = [r+1 for r,i in enumerate(tax_date) if i in '日年月']
        if len(index)==3:
            s = [tax_date[:index[0]], tax_date[index[0]:index[1]], tax_date[index[1]:]]
            s1 = ['','','']
            for i in s:
                if '年' in i:
                    s1[0] = i
                elif '月' in i:
                    s1[1] = i
                elif '日' in i:
                    s1[2] = i
            tax_date = ''.join(s1)
        if f"{len(tax_date)}{tax_date.find('年')}{tax_date.find('月')}" in ['946', '1046', '1047', '1147'] and tax_date[-1]=='日':
            if tax_date[5]=='0':
                tax_date = tax_date[:5]+tax_date[6:]
            if tax_date[-3]=='0':
                tax_date = tax_date[:-3]+tax_date[-2:]
            if tax_date.startswith('20') and tax_date[-3] in '123月':
                self._info['tax_date'] = tax_date
        if '图片模糊' in self._info['tax_date']:
            for i in self._result[0]:
                if i[0][0][1]<self._axis['tax_user_name'][1] or i[0][0][0]<self._axis['tax_date'][0]:
                    continue
                temp = i[1][0].replace(' ', '')
                if len(temp)==8 and sum([1 for j in temp if j in '0123456789'])==8:
                    self._info['tax_date'] = f"{temp[:4]}年{int(temp[4:6])}月{int(temp[6:8])}日"
                    break
                if len(temp)==10 and sum([1 for j in temp if j in '0123456789'])==8 and temp[4]=='-' and temp[7]=='-':
                    self._info['tax_date'] = f"{temp[:4]}年{int(temp[5:7])}月{int(temp[8:10])}日"
                    break
                if len(temp)>19 and sum([1 for j in temp[-8:] if j in '0123456789'])==8:
                    self._info['tax_date'] = f"{temp[-8:][:4]}年{int(temp[-8:][4:6])}月{int(temp[-8:][6:8])}日"
                    break
                if len(temp)>24 and sum([1 for j in temp[-10:] if j in '0123456789'])==8 and temp[-10:][4]=='-' and temp[-10:][7]=='-':
                    self._info['tax_date'] = f"{temp[-10:][:4]}年{int(temp[-10:][5:7])}月{int(temp[-10:][8:10])}日"
                    break
#             print(1, self._info['tax_date'])
            if '图片模糊' not in self._info['tax_date']:
                month = tax_date.split('月')[0].split('年')[-1]
                day = tax_date.split('日')[0].split('月')[-1]
                if len(month)>0 and len(month)==sum([1 for i in month if i in '0123456789']):
                    if len(day)>0 and len(day)==sum([1 for i in day if i in '0123456789']):
                        if 0<int(month)<13:
                            self._info['tax_date'] = (self._info['tax_date'][:self._info['tax_date'].find('年')+1]+str(int(month))
                                                      +self._info['tax_date'][self._info['tax_date'].find('月'):])
                        if 0<int(day)<32:
                            self._info['tax_date'] = self._info['tax_date'][:self._info['tax_date'].find('月')+1]+str(int(day))+'日'
        
        tax_class = ['印花税' if i.endswith('花税') else i for i in tax_class]
        tax_class = '|'.join([i for i in tax_class if i not in ['已完税']])
        for i in self._char_class:
            tax_class = tax_class.replace(i, '契')
        if '图片模糊' in self._info['tax_class'] and len(tax_class)>1:
            self._info['tax_class'] = tax_class
        else:
            self._info['tax_class'] = '契税'
        if 'tax_remark' in axis_true:
            temp = self._remark_function(tax_remark)
            for i in temp:
                self._info[i if 'tax_remark_' in i else 'tax_remark_'+i] = temp[i]

        self._info['tax_amount'] = self._analysis_tax_amount(self._info['tax_amount'])
        self._info['tax_organ'] = self._analysis_tax_organ(self._info['tax_organ'])

#         print(self._info)
        for i in self._info:
            if self._info[i]=='':
                self._info[i] = '图片模糊'
    
        for i in self._axis:
            self._axis[i] = [int(max(0, j)) for j in self._axis[i]]
    
    def _analysis_tax_amount(self, data):
        if '图片模糊' in data:
            amount = ''
        else:
            amount = data.replace('￥', '¥').replace(',', '').replace('，', '').replace('-', '')
            amount = amount[:-3].replace('.', '')+amount[-3:]
            if amount.count('.')==0:
                amount = amount[:-2]+'.'+amount[-2:]
            if amount[0] not in '¥0123456789':
                amount = amount[1:]
            amount = ('¥' + amount).replace('¥¥', '¥')
            if amount[-2]=='.':
                amount = amount+'0'
            if amount[-1]=='.':
                amount = amount+'00'
        return amount
    
    def _analysis_tax_organ(self, data):
        organ = data
        for i,j in [('厨','局'), ('积','税'), ('同', '局'), ('届', '局')]:
            if i in organ:
                organ = organ.replace(i, j)

        if '务局' in organ:
            organ = organ[:organ.find('务局')+2]

            temp = organ[::-1]
            for i in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
                if i in temp:
                    temp = temp[:temp.find(i)]
            organ = temp[::-1]

            for i in ['税务机关', '所']:
                if organ.startswith(i):
                    organ = organ[organ.find(i)+len(i):]

            if '国家' in organ:
                organ = f"国家税务总局{organ[organ.find('局')+1:][:-3]}税务局"
            else:
                organ = f"{organ[:-5]}地方税务局"
                
            organ = organ.replace('北景','北京')
        else:
            organ = ''
        return organ
    
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
            name_list = ['tax_date', 'tax_organ', 'tax_user_name', 'tax_class', 'tax_amount',
                         'tax_remark_address', 'tax_remark_amount']

        score_a = {i:0 for i in name_list}
        score_b = {i:0 for i in name_list}
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

def remark(remark):
    s = remark.replace('：',':').replace('，','').replace(' ','').replace(',','')
    try:
        amount = s
        for i,j in [('机','税'), ('积','税'), ('全','金'), ('企','金'), ('题','额'), ('频','额'), ('卒','率'), 
                    ('车','率'), ('事','率'), ('单','率'), ('Q','0'), ('o','0')]:
            amount = amount.replace(i, j)
        amount = amount[amount.find('计税金额'):]
        for i in ['共有人', '房源', '房屋产权证书号']:
            if i in amount:
                amount = amount[:amount.find(i)]
        
        if amount.endswith('税'):
            amount = amount+'率:'
        elif amount.endswith('率'):
            amount = amount+':'
        elif amount.count('计税金额')==amount.count('税率')+1:
            amount = amount+'税率:'
        
        temp = amount[::-1]
        index = -1
        for r,i in enumerate(temp[:temp.find('率税')]):
            if i not in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', '.', '%']:
                index = r
        amount = temp[index+1:][::-1]
        
        if amount.endswith('率'):
            amount = amount+':'
        if amount.endswith(':'):
            amount = amount+('5.0' if amount.split('计税金额:')[-1].split('税率')[0]=='0'else '0.03')
            
        for i in amount:
            if i not in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '计', '税', '金', '额', ':', '.', '率', '%']:
                amount = amount.replace(i, '')
        for i,j in [('.计','计'), (':', ''), ('率00','率0.0'), ('额','额:'), ('率', '率:')]:
            amount = amount.replace(i, j)
        
        temp =[r+3 for r,i in enumerate(amount) if i=='率'and amount[r+3]!='.']
        while temp:
            amount = amount[:temp[0]]+'.'+amount[temp[0]:]
            temp =[r+3 for r,i in enumerate(amount) if i=='率'and amount[r+3]!='.']
        
        if len(amount)<10:
            amount = ''
    except:
        amount = ''
        
    try:
        address = s
        address_info = ['街','区','院','室','楼','层','门','号','幢','栋','单元']
        address_info_filter = '0123456789'+''.join(address_info)
        for i in ['地址', '位置', '址:', '置:']:
            if i in address:
                address = address[address.find(i)+len(i):]
        
        for i in ['权属', '权届转移', '房屋面积', '交易面积', '房屋产权', '建筑面积', '房座面积']:
            if i in address:
                address = address[:address.find(i)]
        
        if ':' in address:
            temp = address.split(':')
            index = [sum([1 for j in address_info if j in i]) for i in temp]
            address = temp[index.index(max(index))]
            
        temp = address[::-1]
        index = -1
        for k in ['室', '楼', '层', '门', '号']:
            if k in temp:
                for r,i in enumerate(temp[:temp.find(k)]):
                    if i in [',', '(', '（']:
                        index = r
                address = temp[index+1:][::-1]
                break

        for i,j in [('号娄','号楼')]:
            address = address.replace(i, j)
        
        if sum([1 for i in address_info+['路'] if i in address])<3:
            address = ''
        if sum([1 for i in address if i not in address_info_filter])<3:
            address = ''
    except:
        address = ''
    return {'tax_remark_amount':amount, 'tax_remark_address':address}

