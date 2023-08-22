import time

import paddleocr
import linora as la

__all__ = ['OCRWanShuiPiao']


class OCRWanShuiPiao():
    def __init__(self, model=True, name_list=None, remark_function=None):
        if model==True:
            self._model = paddleocr.PaddleOCR(show_log=False)
        elif model:
            self._model = model
        else:
            self._model = None
        self._remark_function = remark_function
        self._keys = ['tax_date', 'tax_organ', 'tax_user_id', 'tax_user_name', 
                      'tax_class', 'tax_amount', 'tax_ticket_filler']
        if name_list is None:
            name_list = self._keys.copy()
        else:
            for i in name_list:
                if i not in self._keys and not i.startswith('tax_remark_'):
                    raise ValueError(f'Variable name `{i}`  does not conform to the specification.')
        self._name_list = name_list
        self._char_tax_date = ['填发日期']
        self._char_tax_organ = ['税务机关']
        self._char_tax_user_id = ['纳税人识别号']
        self._char_tax_user_name = ['纳税人名称']
        self._char_tax_amount = [f'{i}{j}合计' for i in ['金','全'] for j in ['数','额','题','融']]
        self._char_tax_ticket_filler = ['填票人', '慎票人']
        self._char_class = ['翼', '奥', '奖', '樊', '靓', '超', '美', '楚', '烫']
        self._char_name = '阎段冼凌辜石顾齐祝乐虞伏夏穆练赖宇姓苗甄于盖许彭晏敖谌汤梅娄潘李华花陈阮苟池陆韦鲍卜宗管沙覃糜傅周龙耿徐项蓝孟游廉楚扬樊奚竺龚毕章阳党全郁萧张成连漕舒柏卞时巫庄叶文冯霍臧喻揭应蔺唐芦谷席郎桂常单薛丛吴黎佟黄保施闵官盛商侯季宋瞿胡范牛畅蒋秦宫纪苏颜伍巩滕万江杨詹严程童燕百韩任辛关仇计司卓谈谢聂岳桑迟封吕梁袁祁卢冷匡毛温左罗白柳马孙薄家习安涂古邬康仲隋殷沿僧庞尤木牟排贾郭边金孔谭朱葛姜郑位费宁高赵贺陶崔向熊翁栗明银褚柯卫姚强莫戚汪鞠来乔魏符史丁钱廖曲稽吉栾冉井洪都蔡和兰翟欧何刘甘姬易屠胥靳屈嵺查鲁车蒙田戴窦冀景包焦衣沈申曾邱尹邢名裴董丘王邸岑郝解刁饶米柴路台艾原邓俞邹苑余倪方晋郜鄢植荆林杜钟邵武曹房简'
        tax_class_list = ['个人所得税', '印花税', '营业税', '土地增值税', '增值税', '城市维护建设税', 
                          '教育费附加', '地方教育附加', 
                                '契税']
        self._char_tax_class = []
        for i in tax_class_list:
            if len(i)==2:
                self._char_tax_class.append((i, [1,2]))
            elif len(i) in [3,4,5,6]:
                self._char_tax_class.append((i, list(range(1, len(i)+1))[-2:]))
            else:
                self._char_tax_class.append((i, list(range(1, len(i)+1))[max(len(i),4)-3:]))
        
    def predict(self, image, axis=False, model=None):
        self._show_axis = axis
        self._info = '图片模糊或非税票图片'
        self._error = '图片模糊或非税票图片'
        
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
        
        error_list = [i for i in self._info if '图片模糊' in self._info[i]]
        if error_list:
            self._result_crop = []
            tax_remark_logic = True
            for i in error_list:
                if i.startswith('tax_remark') and tax_remark_logic and 'tax_remark' in self._axis:
                    image = la.image.crop(self._image, self._axis['tax_remark'])
                    tax_remark_logic = False
                elif i in self._axis:
                    image = la.image.crop(self._image, self._axis[i])
                else:
                    continue
                t = (self._model if model is None else model).ocr(la.image.image_to_array(image), cls=False)
                if t[0]:
                    for j in t[0]:
                        if i.startswith('tax_remark'):
                            i = 'tax_remark'
                        self._result_crop.append([[self._axis[i][:2], [self._axis[i][2], self._axis[i][1]], 
                                                   self._axis[i][2:], [self._axis[i][0], self._axis[i][3]]], j[1]])
            self._fit_characters(self._axis, [self._result_crop])
        
        self._info['tax_amount_large'] = amount_transform(self._info['tax_amount'])
        if '图片模糊' in self._info.get('tax_class', ''):
            self._info['tax_class'] = '契税'

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
            index1 = ['品目名称', '税款所属时期', '实缴(退)金额', '实缴（退）金额', '入(退)库日期', '入（退）库日期']
            rank = [0,0,0,0,0,0,0]
            for r, i in enumerate(self._result[0], start=1):
                if '税收完税证明' in i[1][0]:
                    if rank[0]==0:
                        rank[0] = r
                elif '填发日期' in i[1][0] or '税务机关：' in i[1][0]:
                    if rank[1]==0:
                        rank[1] = r
                elif '纳税人识别号' in i[1][0] or '纳税人名称' in i[1][0]:
                    if rank[2]==0 or '纳税人名称' in i[1][0]:
                        rank[2] = r
                elif sum([1 for char in index1 if char in i[1][0]]):
                    if rank[3]==0:
                        rank[3] = r
                elif sum([1 for char in self._char_tax_amount if char in i[1][0]]):
                    if rank[4]==0:
                        rank[4] = r
                elif '填票人' in i[1][0]:
                    if rank[5]==0:
                        rank[5] = r
                elif '妥善保管' in i[1][0]:
                    if rank[6]==0:
                        rank[6] = r
            rank = [i for i in rank if i>0]
            if rank==sorted(rank) and len(rank)>1:
                self._image = image
                self._angle = angle
                self._info = {i:'图片模糊' for i in self._name_list}
                if self._remark_function is not None:
                    self._keys.append('tax_remark')
                break
    
    def _fit_axis(self):
        if len(self._result)==0:
            return 0

        axis_true = dict()
        axis_dict = {i:[] for i in self._keys}
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
                            w = w*len(char)/(len(i[1][0])+1)
                        axis_true['tax_date'] = [x+w*1.2, y-h*0.25, x+w*4, y+h*1.25]
                        axis_dict['tax_organ'].append(([x+w*6, y-h*2, x+w*9.5, y+h], 0.8))
                        axis_dict['tax_user_name'].append(([x+w*3.5, y+h, x+w*7, y+h*3.5], 0.8))
                        axis_dict['tax_user_id'].append(([x-w*2, y+h, x+w*2.4, y+h*3.5], 0.8))
                        break
                if 'tax_date' in axis_true:
                    continue
            if 'tax_organ' not in axis_true and tax_organ_logic:
                for char in self._char_tax_organ:
                    if char in i[1][0]:
                        if len(i[1][0])>4:
                            w = w*len(char)/(len(i[1][0])+0.5)
                        axis_true['tax_organ'] = [x+w, y-h*2, x+w*5, y+h]
                        axis_dict['tax_date'].append(([x-w*3.5, y, x-w, y+h], 0.8))
                        axis_dict['tax_user_name'].append(([x-w, None, x+w*2, y+h*3.5], 0.8))
                        axis_dict['tax_user_name'].append(([None, y+h*1.1, None, None], 100))
                        axis_dict['tax_user_id'].append(([x-w*7, y+h, x-w*2.5, y+h*3.5], 0.6))
                        break
                if 'tax_organ' in axis_true:
                    tax_organ_logic = False
                    continue
            if 'tax_user_id' not in axis_true:
                for char in self._char_tax_user_id:
                    if char in i[1][0]:
                        if len(i[1][0])>6:
                            w = w*len(char)/(len(i[1][0])+0.5)
                        axis_true['tax_user_id'] = [x+w, y-h*0.75, x+w*4.2, y+h*1.75]
                        axis_dict['tax_date'].append(([x+w*3.5, y-h*2, x+w*6, y-h], 0.8))
                        axis_dict['tax_organ'].append(([x+w*7.2, y-h*1.8, x+w*10.2, y-h*0.5], 0.6))
                        axis_dict['tax_user_name'].append(([x+w*5.5, y-h*0.75, x+w*8, y+h*1.75], 0.8))
                        axis_dict['tax_class'].append(([x+w*1.5, y+h*3.5, x+w*3, y+h*14], 0.6, 3))
                        tax_organ_logic = False
                        break
                if 'tax_user_id' in axis_true:
                    continue
            if 'tax_user_name' not in axis_true:
                for char in self._char_tax_user_name:
                    if char in i[1][0]:
                        if len(i[1][0])>5:
                            w = w*len(char)/(len(i[1][0])+0.5)
                        axis_true['tax_user_name'] = [x+w, y-h*0.75, x+w*4, y+h*1.75]
                        axis_dict['tax_date'].append(([x-w*1.5, y-h*2, x+w, y-h], 0.8))
                        axis_dict['tax_organ'].append(([x+w*2.5, y-h*1.8, x+w*6, y-h*0.5], 0.8))
                        axis_dict['tax_user_id'].append(([x-w*4, y, x-w*0.75, y+h*1.75], 0.8))
                        axis_dict['tax_class'].append(([x-w*3.8, y+h*3.5, x-w*2, y+h*14], 0.6, 2))
                        if self._remark_function is not None:
                            axis_dict['tax_remark'].append(([x-w*0.25, None, None, None], 100))
                        tax_organ_logic = False
                        break
                if 'tax_user_name' in axis_true:
                    continue
            if '原凭证号'==i[1][0]:
                axis_dict['tax_date'].append(([x+w*3.5, y-h*4, x+w*6, y-h*2.75], 0.6))
                axis_dict['tax_organ'].append(([x+w*7, y-h*4, x+w*10, y-h*2.75], 0.4))
                axis_dict['tax_user_id'].append(([x+w, y-h*3, x+w*4, y-h*0.5], 0.8))
                axis_dict['tax_user_name'].append(([x+w*5.5, y-h*3, x+w*8.5, y-h*0.5], 0.6))
                axis_dict['tax_class'].append(([None, None, x+w*3, y+h*12.5], 0.8))
                axis_dict['tax_class'].append(([x+w*1.3, y+h, None, None], 100))
                tax_organ_logic = False
                continue
            if '品目名称'==i[1][0]:
                axis_dict['tax_date'].append(([x, y-h*4, x+w*2.5, y-h*2.75], 0.6))
                axis_dict['tax_organ'].append(([x+w*4, y-h*4, x+w*7, y-h*2.75], 0.4))
                axis_dict['tax_user_id'].append(([x-w*2.3, y-h*3, x+w, y-h*0.5], 0.8))
                axis_dict['tax_user_name'].append(([x+w*2.3, y-h*3, x+w*5.5, y-h*0.5], 0.8))
                axis_dict['tax_class'].append(([x-w*2.25, None, None, y+h*12.5], 0.8))
                axis_dict['tax_class'].append(([None, y+h, x-w*0.5, None], 100))
                if self._remark_function is not None:
                    axis_dict['tax_remark'].append(([x+w*1.25, None, None, None], 100))
                tax_organ_logic = False
                continue
            if '税款所属时期'==i[1][0]:
                axis_dict['tax_date'].append(([x-w*1.8, y-h*4, x+w*0.5, y-h*2.75], 0.6))
                axis_dict['tax_organ'].append(([x+w*2, y-h*4, x+w*4.5, y-h*2.75], 0.6))
                axis_dict['tax_user_id'].append(([x-w*4, y-h*3, x-w, y-h*0.5], 0.8))
                axis_dict['tax_user_name'].append(([x+w*0.5, y-h*3, x+w*3.5, None], 0.8))
                axis_dict['tax_user_name'].append(([None, None, None, y-h*0.25], 100))
                axis_dict['tax_class'].append(([x-w*3.75, None, x-w*2, y+h*12.5], 0.6))
                axis_dict['tax_class'].append(([None, y+h, None, None], 100))
                axis_dict['tax_amount'].append(([x+w*2.5, None, None, None], 100))
                if self._remark_function is not None:
                    axis_dict['tax_remark'].append(([x-w*.5, None, None, None], 100))
                tax_organ_logic = False
                continue
            if i[1][0] in ['入（退）库日期', '入(退)库日期']:
                axis_dict['tax_date'].append(([x-w*3.5, y-h*4, x-w, y-h*2.75], 0.6))
                axis_dict['tax_organ'].append(([None, y-h*4, None, y-h*2.75], 0.6))
                axis_dict['tax_organ'].append(([x+w*0.5, None, x+w*3, None], 100))
                axis_dict['tax_user_id'].append(([x-w*6, y-h*3, x-w*3.5, y-h*0.5], 0.8))
                axis_dict['tax_user_name'].append(([x-w, y-h*3, x+w*2, None], 0.8))
                axis_dict['tax_user_name'].append(([None, None, None, y-h*0.25], 100))
                axis_dict['tax_class'].append(([x-w*5.75, None, x-w*4, y+h*12.5], 0.4))
                axis_dict['tax_class'].append(([None, y+h, None, None], 100))
                axis_dict['tax_amount'].append(([x+w*1.1, None, None, None], 100))
                tax_organ_logic = False
                continue
            if i[1][0] in ['实缴（退）金额', '实缴(退)金额']:
                axis_dict['tax_date'].append(([x-w*2.5, y-h*4, x-w*4.5, y-h*2.75], 0.6))
                axis_dict['tax_organ'].append(([None, y-h*4, None, y-h*2.75], 0.6))
                axis_dict['tax_organ'].append(([x-w, None, x+w*1.5, None], 100))
                axis_dict['tax_user_id'].append(([x-w*7, y-h*3, x-w*4, y-h*0.5], 0.8))
                axis_dict['tax_user_name'].append(([x-w*2.5, y-h*3, x+w, None], 0.8))
                axis_dict['tax_user_name'].append(([None, None, None, y-h*0.25], 100))
                axis_dict['tax_class'].append(([x-w*6.75, None, x-w*5, y+h*12.5], 0.4))
                axis_dict['tax_class'].append(([None, y+h, None, None], 100))
                axis_dict['tax_amount'].append(([x-w*0.2, None, x+w*1.3, None], 100))
                if self._remark_function is not None:
                    axis_dict['tax_remark'].append(([None, None, x+w*1.5, None], 100))
                tax_organ_logic = False
                continue
            if '交纳税人作完税证明'==i[1][0]:
                axis_dict['tax_amount'].append(([None, y+h*0.9, None, None], 100))
                tax_organ_logic = False
                continue
            if 'tax_amount' not in axis_true:
                for char in self._char_tax_amount:
                    if char in i[1][0]:
                        if len(i[1][0])>4:
                            w = w*len(char)/(len(i[1][0])+1)
                        axis_dict['tax_class'].append(([None, None, None, y-h*0.5], 100))
                        axis_dict['tax_amount'].append(([None, y-h, None, y+h*1.75], 100))
                        axis_dict['tax_amount'].append(([x+w*11.5, None, x+w*13.5, None], 0.8))
                        axis_dict['tax_ticket_filler'].append(([x+w*4.5, y+h*4, x+w*6.5, y+h*5], 0.6))
                        if self._remark_function is not None:
                            axis_dict['tax_remark'].append(([x+w*6.5, y+h*1.5, x+w*13, y+h*9], 0.6))
                        tax_organ_logic = False
                        break
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
        
        tax_date = ''
        tax_class = []
        if self._remark_function is not None:
            tax_remark = ''
        for i in result[0]:
            h = (i[0][3][1]+i[0][2][1]-i[0][1][1]-i[0][0][1])/2
            w = (i[0][1][0]+i[0][2][0]-i[0][0][0]-i[0][3][0])/2
            if h==0:
                h = 1
            if w==0:
                w = 1
            x = min(i[0][0][0], i[0][3][0])
            y = min(i[0][0][1], i[0][1][1])
            if '图片模糊' in self._info.get('tax_organ', ''):
                temp = self._analysis_tax_organ(i[1][0])
                if '图片模糊' not in temp:
                    self._info['tax_organ'] = temp
                    self._axis['tax_organ'] = self._axis['tax_organ'][:2]+i[0][2]
                    continue
            if '图片模糊' in self._info.get('tax_date', ''):
                if sum([1 for char in self._char_tax_date if char in i[1][0]])>0:
                    for char in self._char_tax_date:
                        if char in i[1][0]:
                            tax_date += ''.join([j for j in i[1][0][i[1][0].find(char)+len(char):] if j in '0123456789年月日'])
                            break
                    continue
                if 'tax_date' in axis_true:
                    h1 = min(max(i[0][3][1], i[0][2][1]), axis_true['tax_date'][3])-max(min(i[0][0][1], i[0][1][1]), axis_true['tax_date'][1])
                    w1 = min(max(i[0][1][0], i[0][2][0]), axis_true['tax_date'][2])-max(min(i[0][0][0], i[0][3][0]), axis_true['tax_date'][0])            
                    if h1/h>0.6 and w1/w>0.6:
                        temp = i[1][0].replace(' ', '').replace('：', '').replace(':', '')
                        if len(temp)==sum([1 for char in temp if char in '0123456789年月日']):
                            if sum([1 for char in temp if char in '年月日']):
                                tax_date += temp
        #                       self._axis['tax_date'] = [x, y]+i[0][2]
                                continue
            if '图片模糊' in self._info.get('tax_user_id', ''):
                if sum([1 for char in self._char_tax_user_id if char in i[1][0]])>0:
                    for char in self._char_tax_user_id:
                        if char in i[1][0] and len(i[1][0][i[1][0].find(char)+len(char):])>1:
                            self._info['tax_user_id'] = i[1][0][i[1][0].find(char)+len(char):].strip()
                            self._axis['tax_user_id'] = [self._axis['tax_user_id'][0], i[0][0][1]]+i[0][2]
                            break
                    continue
                if 'tax_user_id' in axis_true:
                    h1 = min(max(i[0][3][1], i[0][2][1]), axis_true['tax_user_id'][3])-max(min(i[0][0][1], i[0][1][1]), axis_true['tax_user_id'][1])
                    w1 = min(max(i[0][1][0], i[0][2][0]), axis_true['tax_user_id'][2])-max(min(i[0][0][0], i[0][3][0]), axis_true['tax_user_id'][0])            
                    if h1/h>0.6 and w1/w>0.6:
                        self._info['tax_user_id'] = i[1][0]
                        self._axis['tax_user_id'] = [x, y]+i[0][2]
                        continue
            if '图片模糊' in self._info.get('tax_user_name', ''):
                if sum([1 for char in self._char_tax_user_name if char in i[1][0]])>0:
                    for char in self._char_tax_user_name:
                        temp = la.text.sequence_preprocess(i[1][0][i[1][0].find(char)+len(char):])
                        if char in i[1][0] and len(temp)>1:
                            if len(temp)==4:
                                temp = temp[:2]+'|'+temp[2:]
                            elif len(temp)==5:
                                temp = temp[:2]+'|'+temp[2:] if temp[2] in self._char_name else temp[:3]+'|'+temp[3:]
                            elif len(temp)==6:
                                temp = temp[:3]+'|'+temp[3:]
                            self._info['tax_user_name'] = temp
                            self._axis['tax_user_name'] = [self._axis['tax_user_name'][0], i[0][0][1]]+i[0][2]
                            break
                if '图片模糊' not in self._info['tax_user_name']:
                    continue
                if 'tax_user_name' in axis_true:
                    h1 = min(max(i[0][3][1], i[0][2][1]), axis_true['tax_user_name'][3])-max(min(i[0][0][1], i[0][1][1]), axis_true['tax_user_name'][1])
                    w1 = min(max(i[0][1][0], i[0][2][0]), axis_true['tax_user_name'][2])-max(min(i[0][0][0], i[0][3][0]), axis_true['tax_user_name'][0])            
                    if h1/h>0.6 and w1/w>0.6 and len(i[1][0])>1 and sum([1 for char in i[1][0] if char in '税务机所：:市区地（）()'])<1:
                        temp = la.text.sequence_preprocess(i[1][0])
                        if len(temp)>1:
                            if len(temp)==4:
                                temp = temp[:2]+'|'+temp[2:]
                            elif len(temp)==5:
                                temp = temp[:2]+'|'+temp[2:] if temp[2] in self._char_name else temp[:3]+'|'+temp[3:]
                            elif len(temp)==6:
                                temp = temp[:3]+'|'+temp[3:]
                            self._info['tax_user_name'] = temp
                            self._axis['tax_user_name'] = [x, y]+i[0][2]
                            continue
            if '图片模糊' in self._info.get('tax_class', '') and 'tax_class' in axis_true:
                h1 = min(max(i[0][3][1], i[0][2][1]), axis_true['tax_class'][3])-max(min(i[0][0][1], i[0][1][1]), axis_true['tax_class'][1])
                w1 = min(max(i[0][1][0], i[0][2][0]), axis_true['tax_class'][2])-max(min(i[0][0][0], i[0][3][0]), axis_true['tax_class'][0])            
                temp = ''.join([j for j in i[1][0] if j not in '0123456789|']).replace(' ', '').replace('臧', '城').replace('时', '附').replace('锐', '税')
                temp = la.text.sequence_preprocess(temp)
                for char in self._char_class:
                    temp = temp.replace(char, '契')
                if self._axis['tax_class'][1]<i[0][2][1]<self._axis['tax_class'][3] and self._axis['tax_class'][0]<i[0][2][0]<self._axis['tax_class'][2]:
                    if len(i[1][0])-len(temp)>12 and temp=='税':
                        temp = '契税'
                        continue
                    if (len(i[1][0])-len(temp)>12 or (h1/h>0.6 and w1/w>0.6)) and temp not in ['税', '税种', '已完税'] and len(temp)>0:
                        if temp.endswith('附加'):
                            tax_class.append(temp)
                            continue
                        break_logic = False
                        for tax_i, tax_j in self._char_tax_class:
                            if sum([1 for char in temp if char in tax_i]) in tax_j and len(temp) in tax_j and len(temp)<=len(tax_i):
                                tax_class.append(tax_i)
                                break_logic = True
                                break
                        if break_logic:
                            continue
            if '图片模糊' in self._info.get('tax_amount', ''):
                if '￥' in i[1][0] or '¥' in i[1][0] or 'Y'==i[1][0].strip()[0] or 'X'==i[1][0].strip()[0]:
                    temp = self._analysis_tax_amount(i[1][0])
                    if '图片模糊' not in temp:
                        self._info['tax_amount'] = temp
                        self._axis['tax_amount'] = [x, y]+i[0][2]
                        if 'tax_remark' in self._axis:
                            self._axis['tax_remark'][1] = i[0][2][1]+h*0.25
                        continue
                if 'tax_amount' in axis_true:
                    h1 = min(max(i[0][3][1], i[0][2][1]), axis_true['tax_amount'][3])-max(min(i[0][0][1], i[0][1][1]), axis_true['tax_amount'][1])
                    w1 = min(max(i[0][1][0], i[0][2][0]), axis_true['tax_amount'][2])-max(min(i[0][0][0], i[0][3][0]), axis_true['tax_amount'][0])            
                    if h1/h>0.6 and w1/w>0.6 and len(i[1][0])>2:
                        temp = self._analysis_tax_amount(i[1][0])
                        if '图片模糊' not in temp:
                            self._info['tax_amount'] = temp
                            self._axis['tax_amount'] = [x, y]+i[0][2]
                            if 'tax_remark' in self._axis:
                                self._axis['tax_remark'][1] = i[0][2][1]+h*0.25
                            continue
            if '图片模糊' in self._info.get('tax_ticket_filler', '') and 'tax_ticket_filler' in axis_true:
                h1 = min(max(i[0][3][1], i[0][2][1]), axis_true['tax_ticket_filler'][3])-max(min(i[0][0][1], i[0][1][1]), axis_true['tax_ticket_filler'][1])
                w1 = min(max(i[0][1][0], i[0][2][0]), axis_true['tax_ticket_filler'][2])-max(min(i[0][0][0], i[0][3][0]), axis_true['tax_ticket_filler'][0])            
                if h1/h>0.6 and w1/w>0.6:
                    self._info['tax_ticket_filler'] = i[1][0]
                    self._axis['tax_ticket_filler'] = [x, y]+i[0][2]
                    continue
            if 'tax_remark' in axis_true:
                h1 = min(max(i[0][3][1], i[0][2][1]), axis_true['tax_remark'][3])-max(min(i[0][0][1], i[0][1][1]), axis_true['tax_remark'][1])
                w1 = min(max(i[0][1][0], i[0][2][0]), axis_true['tax_remark'][2])-max(min(i[0][0][0], i[0][3][0]), axis_true['tax_remark'][0])            
                if h1/h>0.6 and w1/w>0.6:
                    tax_remark += i[1][0]
                    continue
    
        if '图片模糊' in self._info.get('tax_date', ''):
            self._info['tax_date'] = self._analysis_tax_date(tax_date)
        if '图片模糊' in self._info.get('tax_class', ''):
            self._info['tax_class'] = self._analysis_tax_class(tax_class)
        if 'tax_remark' in axis_true:
            self._analysis_tax_remark(tax_remark)
    
        for i in self._axis:
            self._axis[i] = [int(max(0, j)) for j in self._axis[i]]

    def _analysis_tax_remark(self, data):
        temp = self._remark_function(data)
        for i in temp:
            if '图片模糊' in self._info.get(i, ''):
                self._info[i] = '图片模糊' if temp[i]=='' else temp[i]
            
    def _analysis_tax_class(self, data):
        tax_class = '|'.join(data)
        if len(tax_class)<2:
            tax_class = '图片模糊'
        return tax_class
    
    def _analysis_tax_date(self, data):
        date_info = '图片模糊:未识别出填发日期'
        date = data.replace(' ', '').replace('：', '')
        index = [r+1 for r,i in enumerate(date) if i in '年月日']
        if len(index)==3:
            s = [date[:index[0]], date[index[0]:index[1]], date[index[1]:]]
            s1 = ['','','']
            for i in s:
                if '年' in i:
                    s1[0] = i
                elif '月' in i:
                    s1[1] = i
                elif '日' in i:
                    s1[2] = i
            date = ''.join(s1)
        if f"{len(date)}{date.find('年')}{date.find('月')}" in ['946', '1046', '1047', '1147'] and date[-1]=='日':
            if date[5]=='0':
                date = date[:5]+date[6:]
            if date[-3]=='0':
                date = date[:-3]+date[-2:]
            month = date.split('月')[0].split('年')[-1]
            day = date.split('日')[0].split('月')[-1]
            if date.startswith('20') and 0<int(month)<13 and 0<int(day)<32:
                date_info = date
        if '图片模糊' in date_info:
            for i in self._result[0]:
                if 'tax_date' in self._axis:
                    if i[0][0][1]<self._axis['tax_date'][1] or i[0][0][0]<self._axis['tax_date'][0]:
                        continue
                if 'tax_amount' in self._axis:
                    if i[0][0][1]>self._axis['tax_amount'][1]:
                        continue
                if 'tax_remark' in self._axis:
                    if i[0][0][1]>self._axis['tax_remark'][1]:
                        continue
                temp = i[1][0].replace(' ', '')
                if len(temp)==8 and sum([1 for j in temp if j in '0123456789'])==8:
                    date_info = f"{temp[:4]}年{int(temp[4:6])}月{int(temp[6:8])}日"
                elif len(temp)==10 and sum([1 for j in temp if j in '0123456789'])==8 and temp[4]=='-' and temp[7]=='-':
                    date_info = f"{temp[:4]}年{int(temp[5:7])}月{int(temp[8:10])}日"
                elif len(temp)>19 and sum([1 for j in temp[-8:] if j in '0123456789'])==8:
                    date_info = f"{temp[-8:][:4]}年{int(temp[-8:][4:6])}月{int(temp[-8:][6:8])}日"
                elif len(temp)>24 and sum([1 for j in temp[-10:] if j in '0123456789'])==8 and temp[-10:][4]=='-' and temp[-10:][7]=='-':
                    date_info = f"{temp[-10:][:4]}年{int(temp[-10:][5:7])}月{int(temp[-10:][8:10])}日"
                if '图片模糊' not in date_info:
                    month = date_info.split('月')[0].split('年')[-1]
                    day = date_info.split('日')[0].split('月')[-1]
                    if not date_info.startswith('20') or int(day)>31 or int(month)>12:
                        date_info = '图片模糊:未识别出填发日期'
                    else:
                        break
            if '图片模糊' not in date_info:
                month = date.split('月')[0].split('年')[-1]
                day = date.split('日')[0].split('月')[-1]
                if len(month)>0 and len(month)==sum([1 for i in month if i in '0123456789']):
                    if 0<int(month)<13:
                        date_info = date_info[:date_info.find('年')+1]+str(int(month))+date_info[date_info.find('月'):]
                if len(day)>0 and len(day)==sum([1 for i in day if i in '0123456789']):
                    if 0<int(day)<32:
                        date_info = date_info[:date_info.find('月')+1]+str(int(day))+'日'
        return date_info
    
    def _analysis_tax_amount(self, data):
        amount = data.strip()
        for i, j in [(':', ''), ('/', ''), ('d', '0'), ('b', '0'), ('o', '0'), ('B', '8'), ('￥', '¥'), (',', ''),
                     ('，', ''), ('-', ''), (';', ''), (' ', '')]:
            amount = amount.replace(i, j)
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
        
        if len(amount)>4:
            if amount[0]=='¥' and amount[-3]=='.':
                if not [1 for i in amount[1:-3]+amount[-2:] if i not in '0123456789']:
                    return amount
        return '图片模糊'
    
    def _analysis_tax_organ(self, data):
        if sum([1 for char in data if char not in '0123456789'])<9:
            return '图片模糊:未识别出税务机关'
        if sum([1 for char in set(data) if char in '国家税务局市区地方'])<4:
            return '图片模糊:未识别出税务机关'
        if sum([1 for char in data if char not in '机关所办服厅：（）'])<9:
            return '图片模糊:未识别出税务机关'
        organ = data.replace(' ', '').replace('：', ':').split(':')[-1]
        for char in self._char_tax_organ:
            if char in organ:
                organ = organ[organ.find(char)+len(char):]
                continue
        for i,j in [('厨','局'), ('积','税'), ('同', '局'), ('届', '局')]:
            if i in organ:
                organ = organ.replace(i, j)
        if organ.count('税务')>1:
            if '国家' in organ:
                temp = organ[organ.find('税务')+2:]
                organ = organ[:organ.find('税务')+2]+temp[:temp.find('税务')+2].replace('税务', '税务局')
            else:
                organ = organ[:organ.find('税务')+2]+'局'
        
        if organ.endswith('区'):
            organ = organ+('税务局' if '国家' in organ else '地方税务局')
        for i in '地方税务':
            if organ.endswith(i):
                organ = organ+'地方税务局'['地方税务'.find(i)+1:]
        
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
            
            if len(organ)<10:
                organ = '图片模糊:未识别出税务机关'
        else:
            organ = '图片模糊:未识别出税务机关'
        return organ
    
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
            name_list = ['tax_date', 'tax_organ', 'tax_user_name', 'tax_class', 'tax_amount']

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
        elif len(la.text.sequence_preprocess(amount.replace('计税金额:', '').replace('税率:', ''))):
            amount = ''
        elif [1 for i in amount.replace('计税金额:', '|').replace('税率:', '|').split('|') if len(i)>12]:
            amount = ''
        elif [1 for i in amount.replace('计税金额:', '|').replace('税率:', '|').split('|') if i.count('.')>1]:
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
        
        for i in ['权属', '权届', '权居', '权屋', '权转', '叔属', '权合', 
                  '房屋面积', '交易面积', '房屋产权', '建筑面积', '房座面积', '房星面积']:
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

        for i,j in [('号娄','号楼'), ('号接', '号楼'), ('单完', '单元'), ('层单', '层1单'), ('号号', '号'), ('（法院拍卖）', '')]:
            address = address.replace(i, j)
        if la.text.sequence_preprocess(address[0])=='':
            address = address[1:]
        if address[-1] in '.、#-':
            address = address[:-1]
        
        if sum([1 for i in address_info+['路'] if i in address])<3:
            address = ''
        elif sum([1 for i in address if i not in address_info_filter])<3:
            address = ''
        elif [1 for i in ['房源', '房屋', '税票'] if i in address]:
            address = ''
    except:
        address = ''
    return {'tax_remark_amount':amount, 'tax_remark_address':address}

def amount_transform(data):
    temp = data.replace(',', '')
    if '.' not in temp:
        temp += '.00'
    if temp[0] in '0123456789':
        temp = '¥'+temp
    trans_number = {'0':'', '1': '壹', '2': '贰', '3': '叁', '4': '肆', '5': '伍', '6': '陆', '7': '柒', '8': '捌', '9': '玖'}
    trans_str = {'¥':'(人民币)', '$':'(美元)'}
    amount = []
    if sum([1 for i in temp[1:-3]+temp[-2:] if i in '0123456789'])==len(temp)-2 and temp[1]!='0':
        cla = trans_str[temp[0]]
        temp = temp[1:]
        for i,j,k in [(-11, -15, '亿'), (-7, -11, '万'), (-3, -7, '元')]:
            t = (4-len(temp[j:i]))*'0'+temp[j:i]
            if temp[j:i]!='':
                if t[0]!='0':
                    amount.append(trans_number[t[0]]+'仟')
                else:
                    amount.append('零')
                if t[1]!='0':
                    amount.append(trans_number[t[1]]+'佰')
                elif t[0]!='0':
                    amount.append('零')
                if t[2]!='0':
                    amount.append(trans_number[t[2]]+'拾')
                elif t[1]!='0':
                    amount.append('零')
                amount.append(trans_number[t[3]]+k)
        
        if temp[-2:]=='00':
            amount.append('整')
        else:
            if temp[-2]!='0':
                amount.append(f"{trans_number[temp[-2]]}角")
            if temp[-1]!='0':
                amount.append(f"{trans_number[temp[-1]]}分")
        if amount[0]=='零':
            amount = amount[1:]
        amount = cla+''.join(amount)
        for i,j in [('零万', '万'), ('零元', '元'), ('零亿', '亿'), ('拾亿', '拾亿零'), ('拾万', '拾万零')]:
            amount = amount.replace(i,j)
    else:
        amount = ''
    return amount
