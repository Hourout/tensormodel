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
        self._keys = ['marriage_name', 'marriage_date', 'marriage_id', 
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
        self._char_user_name = ['姓名', '姓多']
        self._char_user_country = ['国籍', '国箱', '国馨', '国精']
        self._char_user_sex = ['性别']
        self._char_user_born = ['出生日期']
        self._char_user_number = ['身份证件号', '身份证件导']
        self._char_name = '阎段冼凌辜石顾齐祝乐虞伏夏穆练赖宇姓苗甄于盖许彭晏敖谌汤梅娄潘李华花陈阮苟池陆韦鲍卜宗管沙覃糜傅周龙耿徐项蓝孟游廉楚扬樊奚竺龚毕章阳党全郁萧张成连漕舒柏卞时巫庄叶文冯霍臧喻揭应蔺唐芦谷席郎桂常单薛丛吴黎佟黄保施闵官盛商侯季宋瞿胡范牛畅蒋秦宫纪苏颜伍巩滕万江杨詹严程童燕百韩任辛关仇计司卓谈谢聂岳桑迟封吕梁袁祁卢冷匡毛温左罗白柳马孙薄家习安涂古邬康仲隋殷沿僧庞尤木牟排贾郭边金孔谭朱葛姜郑位费宁高赵贺陶崔向熊翁栗明银褚柯卫姚强莫戚汪鞠来乔魏符史丁钱廖曲稽吉栾冉井洪都蔡和兰翟欧何刘甘姬易屠胥靳屈嵺查鲁车蒙田戴窦冀景包焦衣沈申曾邱尹邢名裴董丘王邸岑郝解刁饶米柴路台艾原邓俞邹苑余倪方晋郜鄢植荆林杜钟邵武曹房简'
        
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
        
        error_list = [i for i in self._info if '图片模糊' in self._info[i]]
        if error_list:
            self._result_crop = []
            for i in error_list:
                if i not in self._axis:
                    continue
                image = la.image.crop(self._image, self._axis[i])
                t = (self._model if model is None else model).ocr(la.image.image_to_array(image), cls=False)
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
            rank = [0,0,0,0,0]
            for r, i in enumerate(self._result[0], start=1):
                if '持证人' in i[1][0]:
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
                elif '身份证件号' in i[1][0]:
                    if rank[4]==0:
                        rank[4] = r
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
            h = (i[0][3][1]+i[0][2][1]-i[0][1][1]-i[0][0][1])/2
            w = (i[0][1][0]+i[0][2][0]-i[0][0][0]-i[0][3][0])/2
            x = min(i[0][0][0], i[0][3][0])
            y = min(i[0][0][1], i[0][1][1])
            if 'marriage_name' not in axis_true:
                temp = [char for char in self._char_marriage_name if char in i[1][0]]
                if temp:
                    if len(i[1][0][i[1][0].find(temp[0])+len(temp[0]):])>1:
                        w = w*3/len(i[1][0])
                    axis_true['marriage_name'] = [x, y-h*2, x+w*3, y+h*5.5]
                    axis_dict['marriage_date'].append(([x, y+h*3, x+w*3, y+h*6.5], 0.8))
                    axis_dict['marriage_id'].append(([x, y+h*6, x+w*4, y+h*9.5], 0.6))
                    continue
            if 'marriage_date' not in axis_true:
                temp = [char for char in self._char_marriage_date if char in i[1][0]]
                if temp:
                    if len(i[1][0][i[1][0].find(temp[0])+len(temp[0]):])>1:
                        w = w*0.3
                    axis_true['marriage_date'] = [x, y-h*2, x+w*3, y+h*5.5]
                    axis_dict['marriage_name'].append(([x, y-h*3.5, x+w*3, y-h*0.5], 0.8))
                    axis_dict['marriage_id'].append(([x, y+h*3, x+w*4, y+h*6.5], 0.8))
                    continue
            if 'marriage_id' not in axis_true:
                temp = [char for char in self._char_marriage_id if char in i[1][0]]
                if temp:
                    if len(i[1][0][i[1][0].find(temp[0])+len(temp[0]):])>1:
                        w = w*0.3
                    axis_true['marriage_id'] = [x, y-h*2, x+w*4, y+h*6]
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
                    axis_dict['user_number_up'].append(([None, y+h*3, None, y+h*7], 0.6))
                    axis_dict['user_name_down'].append(([None, y+h*6, None, y+h*10], 0.4))
                    axis_dict['user_country_up'].append(([x+w, None, x+w*4.5, None], 100))
                    axis_dict['user_number_up'].append(([x+w*2, None, x+w*9, None], 100))
                    axis_dict['user_name_down'].append(([x+w, None, x+w*4.5, None], 100))
                    axis_dict['user_country_down'].append(([x+w, None, x+w*4.5, None], 100))
                    axis_dict['user_number_down'].append(([x+w*2, None, x+w*9, None], 100))
                    continue
            if 'user_country_up' not in axis_true and i[0][0][1]<self._axis_up_down:
                temp = [char for char in self._char_user_country if char in i[1][0]]
                if temp:
                    if len(i[1][0][i[1][0].find(temp[0])+len(temp[0]):])>1:
                        w = w*2/(len(i[1][0])+1)
                    axis_true['user_country_up'] = [x+w, y-h, x+w*4.5, y+h*2]
                    axis_dict['user_name_up'].append(([None, y-h*4, None, y], 0.8))
                    axis_dict['user_number_up'].append(([None, y+h, None, y+h*5], 0.8))
                    axis_dict['user_name_down'].append(([None, y+h*4, None, y+h*8], 0.4))
                    axis_dict['user_name_up'].append(([x+w, None, x+w*4.5, None], 100))
                    axis_dict['user_number_up'].append(([x+w*2, None, x+w*9, None], 100))
                    axis_dict['user_name_down'].append(([x+w, None, x+w*4.5, None], 100))
                    axis_dict['user_country_down'].append(([x+w, None, x+w*4.5, None], 100))
                    axis_dict['user_number_down'].append(([x+w*2, None, x+w*9, None], 100))
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
                    axis_true['user_number_up'] = [x+w, y-h, x+w*4, y+h*2]
                    axis_dict['user_name_up'].append(([None, y-h*5, None, y-h*2], 0.6))
                    axis_dict['user_country_up'].append(([None, y-h*3, None, y], 0.8))
                    axis_dict['user_name_down'].append(([None, y+h*2, None, y+h*5], 0.4))
                    axis_dict['user_name_up'].append(([x+w*0.4, None, x+w*3, None], 100))
                    axis_dict['user_country_up'].append(([x+w*0.4, None, x+w*3, None], 100))
                    axis_dict['user_name_down'].append(([x+w*0.4, None, x+w*3, None], 100))
                    axis_dict['user_country_down'].append(([x+w*0.4, None, x+w*3, None], 100))
                    axis_dict['user_number_down'].append(([x+w, None, x+w*4, None], 100))
                    continue
            if 'user_name_down' not in axis_true and i[0][0][1]>self._axis_up_down:
                temp = [char for char in self._char_user_name if char in i[1][0]]
                if temp:
                    if len(i[1][0])>2:
                        w = w*2/(len(i[1][0])+1)
                    axis_true['user_name_down'] = [x+w, y-h*1.5, x+w*4.5, y+h*2]
                    axis_dict['user_number_up'].append(([None, y-h*5, None, y-h], 0.8))
                    axis_dict['user_country_down'].append(([None, y+h, None, y+h*4], 0.8))
                    axis_dict['user_number_down'].append(([None, y+h*3, None, y+h*7], 0.4))
                    axis_dict['user_name_up'].append(([x+w, None, x+w*4.5, None], 100))
                    axis_dict['user_country_up'].append(([x+w, None, x+w*4.5, None], 100))
                    axis_dict['user_number_up'].append(([x+w*2, None, x+w*9, None], 100))
                    axis_dict['user_country_down'].append(([x+w, None, x+w*4.5, None], 100))
                    axis_dict['user_number_down'].append(([x+w*2, None, x+w*9, None], 100))
                    continue
            if 'user_country_down' not in axis_true and i[0][0][1]>self._axis_up_down:
                temp = [char for char in self._char_user_country if char in i[1][0]]
                if temp:
                    if len(i[1][0][i[1][0].find(temp[0])+len(temp[0]):])>1:
                        w = w*2/(len(i[1][0])+1)
                    axis_true['user_country_down'] = [x+w, y-h, x+w*4.5, y+h*2]
                    axis_dict['user_number_up'].append(([None, y-h*7, None, y-h*3], 0.4))
                    axis_dict['user_name_down'].append(([None, y-h*3, None, y], 0.8))
                    axis_dict['user_number_down'].append(([None, y+h, None, y+h*5], 0.8))
                    axis_dict['user_name_up'].append(([x+w, None, x+w*4.5, None], 100))
                    axis_dict['user_country_up'].append(([x+w, None, x+w*4.5, None], 100))
                    axis_dict['user_number_up'].append(([x+w*2, None, x+w*9, None], 100))
                    axis_dict['user_name_down'].append(([x+w, None, x+w*4.5, None], 100))
                    axis_dict['user_number_down'].append(([x+w*2, None, x+w*9, None], 100))
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
                    axis_true['user_number_down'] = [x+w, y-h, x+w*4, y+h*2]
                    axis_dict['user_number_up'].append(([None, y-h*8, None, y-h*5], 0.4))
                    axis_dict['user_name_down'].append(([None, y-h*5, None, y-h*2], 0.6))
                    axis_dict['user_country_down'].append(([None, y-h*3, None, y], 0.8))
                    axis_dict['user_name_up'].append(([x+w*0.4, None, x+w*3, None], 100))
                    axis_dict['user_country_up'].append(([x+w*0.4, None, x+w*3, None], 100))
                    axis_dict['user_number_up'].append(([x+w, None, x+w*4, None], 100))
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
#         step = True
#         step_name = True
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
            if '图片模糊' in self._info.get('marriage_date', '') and 'marriage_date' in axis_true:
                h1 = min(max(i[0][3][1], i[0][2][1]), axis_true['marriage_date'][3])-max(min(i[0][0][1], i[0][1][1]), axis_true['marriage_date'][1])
                w1 = min(max(i[0][1][0], i[0][2][0]), axis_true['marriage_date'][2])-max(min(i[0][0][0], i[0][3][0]), axis_true['marriage_date'][0])
                temp = self._analysis_date(i[1][0])
                if h1/h>0.6 and w1/w>0.6 and len(temp)==11 and temp.find('年')==4:
                    self._info['marriage_date'] = temp
                    self._axis['marriage_date'] = [x, y]+i[0][2]
                    continue
            if '图片模糊' in self._info.get('marriage_id', '') and 'marriage_id' in axis_true:
                h1 = min(max(i[0][3][1], i[0][2][1]), axis_true['marriage_id'][3])-max(min(i[0][0][1], i[0][1][1]), axis_true['marriage_id'][1])
                w1 = min(max(i[0][1][0], i[0][2][0]), axis_true['marriage_id'][2])-max(min(i[0][0][0], i[0][3][0]), axis_true['marriage_id'][0])            
                temp = i[1][0].replace('T', 'J').replace('--', '-').replace(' ', '').replace('结宇', '结字').replace('.', '')
                for char in ['号', '婚证字', '备', '注']:
                    if char in temp:
                        temp = temp[temp.find(char)+len(char):]
                if h1/h>0.6 and w1/w>0.6 and sum([1 for char in temp if char in '年月日其他，补发此证'])<2:
                    if len(temp)>9:
                        if temp[0]=='B' and temp[1] in '0123456789':
                            temp = 'BJ'+temp[1:]
                        if len(temp.split('-')[0])==7 and temp[0] not in 'JL':
                            temp = 'J'+temp[1:]
                        elif len(temp.split('-')[0])==6:
                            temp = 'J'+temp

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
                    for char in ['姓', '名']:
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
#                         step_name = False
                        continue
            if '图片模糊' in self._info.get('user_country_up', '') and 'user_country_up' in axis_true and i[0][0][1]<self._axis_up_down:
                temp = la.text.sequence_preprocess(i[1][0]).replace('中华人民共和国', '中国').replace('国国', '中国')
                if sum([1 for char in temp if char in '姓名性别男女出生期身份证件号'])==0:
                    h1 = min(max(i[0][3][1], i[0][2][1]), axis_true['user_country_up'][3])-max(min(i[0][0][1], i[0][1][1]), axis_true['user_country_up'][1])
                    w1 = min(max(i[0][1][0], i[0][2][0]), axis_true['user_country_up'][2])-max(min(i[0][0][0], i[0][3][0]), axis_true['user_country_up'][0]) 
                    for char in '籍箱馨精奢':
                        if char in temp:
                            temp = temp[temp.find(char)+len(char):]
                    if (h1/h>0.6 and w1/w>0.4 and 10>len(temp)>1) or '中国' in temp:
                        if (not (len(temp) in [2,3] and temp[0] in self._char_name)) or '中国' in temp:
                            self._info['user_country_up'] = '中国' if '中国' in temp else temp
                            self._axis['user_country_up'] = [self._axis['user_country_up'][0], y]+i[0][2]
                            continue
            if '图片模糊' in self._info.get('user_number_up', '') and i[0][0][1]<self._axis_up_down+h*2:
                if sum([1 for j in i[1][0][-18:] if j in '0123456789xX'])==18:
                    self._info['user_number_up'] = i[1][0][-18:]
                    self._info['user_sex_up'] =  '男' if int(i[1][0][-18:][16])%2 else '女'
                    self._info['user_born_up'] = f"{i[1][0][-18:][6:10]}年{i[1][0][-18:][10:12]}月{i[1][0][-18:][12:14]}日"
                elif sum([1 for j in i[1][0][:18] if j in '0123456789xX'])==18:
                    self._info['user_number_up'] = i[1][0][:18]
                    self._info['user_sex_up'] =  '男' if int(i[1][0][:18][16])%2 else '女'
                    self._info['user_born_up'] = f"{i[1][0][:18][6:10]}年{i[1][0][:18][10:12]}月{i[1][0][:18][12:14]}日"
                elif sum([1 for j in i[1][0] if j in '0123456789xX'])==17:
                    temp = '1'+''.join([j for j in i[1][0] if j in '0123456789xX'])
                    if temp[6:8] in ['19', '20']:
                        self._info['user_number_up'] = temp
                        self._info['user_sex_up'] =  '男' if int(temp[16])%2 else '女'
                        self._info['user_born_up'] = f"{temp[6:10]}年{temp[10:12]}月{temp[12:14]}日"
                if '图片模糊' not in self._info['user_number_up']:
                    continue
            if '图片模糊' in self._info.get('user_name_down', '') and 'user_name_down' in axis_true:
                temp = la.text.sequence_preprocess(i[1][0])
                if sum([1 for char in temp if char in '国籍中性别男女出生期身份证件号'])<2:
                    h1 = min(max(i[0][3][1], i[0][2][1]), axis_true['user_name_down'][3])-max(min(i[0][0][1], i[0][1][1]), axis_true['user_name_down'][1])
                    w1 = min(max(i[0][1][0], i[0][2][0]), axis_true['user_name_down'][2])-max(min(i[0][0][0], i[0][3][0]), axis_true['user_name_down'][0])            
                    for char in ['姓', '名']:
                        if temp.startswith(char):
                            temp = temp[temp.find(char)+len(char):]
                    if h1/h>0.6 and w1/w>0.4 and len(temp)>1:
                        self._info['user_name_down'] = temp
                        self._axis['user_name_down'] = [self._axis['user_name_down'][0], y]+i[0][2]
#                         step = False
                        continue
            if '图片模糊' in self._info.get('user_country_down', '') and 'user_country_down' in axis_true and i[0][0][1]>self._axis_up_down:
                temp = la.text.sequence_preprocess(i[1][0]).replace('中华人民共和国', '中国').replace('国国', '中国')
                if sum([1 for char in temp if char in '姓名性别男女出生期身份证件号'])==0:
                    h1 = min(max(i[0][3][1], i[0][2][1]), axis_true['user_country_down'][3])-max(min(i[0][0][1], i[0][1][1]), axis_true['user_country_down'][1])
                    w1 = min(max(i[0][1][0], i[0][2][0]), axis_true['user_country_down'][2])-max(min(i[0][0][0], i[0][3][0]), axis_true['user_country_down'][0]) 
                    for char in '籍箱馨精':
                        if char in temp:
                            temp = temp[temp.find(char)+len(char):]
                    if (h1/h>0.6 and w1/w>0.4 and 10>len(temp)>1) or '中国' in temp:
                        if (not (len(temp) in [2,3] and temp[0] in self._char_name)) or '中国' in temp:
                            self._info['user_country_down'] = '中国' if '中国' in temp else temp
                            self._axis['user_country_down'] = [self._axis['user_country_down'][0], y]+i[0][2]
#                             step = False
                            continue
            if '图片模糊' in self._info.get('user_number_down', '') and i[0][0][1]>self._axis_up_down:
                if sum([1 for j in i[1][0][-18:] if j in '0123456789xX'])==18:
                    self._info['user_number_down'] = i[1][0][-18:]
                    self._info['user_sex_down'] =  '男' if int(i[1][0][-18:][16])%2 else '女'
                    self._info['user_born_down'] = f"{i[1][0][-18:][6:10]}年{i[1][0][-18:][10:12]}月{i[1][0][-18:][12:14]}日"
                elif sum([1 for j in i[1][0][:18] if j in '0123456789xX'])==18:
                    self._info['user_number_down'] = i[1][0][:18]
                    self._info['user_sex_down'] =  '男' if int(i[1][0][:18][16])%2 else '女'
                    self._info['user_born_down'] = f"{i[1][0][:18][6:10]}年{i[1][0][:18][10:12]}月{i[1][0][:18][12:14]}日"
                elif sum([1 for j in i[1][0] if j in '0123456789xX'])==17:
                    temp = '1'+''.join([j for j in i[1][0] if j in '0123456789xX'])
                    if temp[6:8] in ['19', '20']:
                        self._info['user_number_down'] = temp
                        self._info['user_sex_down'] =  '男' if int(temp[16])%2 else '女'
                        self._info['user_born_down'] = f"{temp[6:10]}年{temp[10:12]}月{temp[12:14]}日"

        if '图片模糊' in self._info.get('marriage_id', '') and marriage_id:
            self._info['marriage_id'] = marriage_id
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
                temp = self._analysis_date(i[1][0])
                if temp.find('年')==4 and '月' in temp:
                    date.append(temp)
            date = [i for i in date if i!=self._info.get('marriage_date', '')]
                
#             date = [i[1][0][-11:] for i in self._result[0] if i[1][0][-11:].find('年')==4 and '月' in i[1][0] and i[1][0].endswith('日')][-2:]
#             if date:
#                 if date[0]== self._info.get('marriage_date', ''):
#                     date = date[1:]
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
        
        if self._show_axis:
            try:
                if len(fix_x)>0:
                    fix_x = sum(fix_x)/len(fix_x)
                    self._axis['marriage_name'][0] = fix_x
                    self._axis['marriage_date'][0] = fix_x
                    self._axis['marriage_id'][0] = fix_x

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
    
    def _analysis_date(self, data):
        temp = data.replace('日月', '11月').replace('.', '-')
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
            name_list = ['marriage_name', 'marriage_date', 'marriage_id', 
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

