import time

import paddleocr
import linora as la
from fuzzywuzzy import fuzz

__all__ = ['OCRYinHangKa']


class OCRYinHangKa():
    def __init__(self, model=True, name_list=None):
        if model==True:
            self._model = paddleocr.PaddleOCR(show_log=False)
        elif model:
            self._model = model
        else:
            self._model = None
        self._keys = ['bank_name', 'bank_number', 'bank_type']
        if name_list is None:
            name_list = self._keys.copy()
        else:
            for i in name_list:
                if i not in self._keys:
                    raise ValueError(f'Variable name `{i}`  does not conform to the specification.')
        self._name_list = name_list
        self._char_direction = ['中国', '银行', '银联', '借记', '记卡', '信用', '章程', '热线', '签名', '持卡人',
                                '客户', '服务', '使用', '客服', '一卡通', '办理', '电话', '网上', '出租',
                                '转借', '转让', '结算', 'ATM', 'Bank', 'China', 'Union', 
                                '交通', '工商', '招商', '建设', '农业', '农商', '光大', '邮政', '储蓄', '浦发', '平安',
                               '广发', '中信', '622']
        self._char_bank_name = [
            '中国银行', '中国工商银行', '中国农业银行', '中国建设银行', '交通银行', '中国邮政储蓄银行', 
            
            '中信银行', '中国光大银行', '招商银行', '浦发银行', '民生银行', '华夏银行', '平安银行', 
            '兴业银行', '广发银行', '渤海银行', '浙商银行', '恒丰银行', 
            
            '北京银行', '天津银行', '河北银行', '张家口银行', '承德银行', '秦皇岛银行', '唐山银行', '廊坊银行', '保定银行', 
            '沧州银行', '衡水银行', '邢台银行', '邯郸银行', '晋商银行', '大同银行', '晋城银行', '晋中银行', '阳泉商业银行', 
            '长治银行', '包商银行', '内蒙古银行', '鄂尔多斯银行', '乌海银行', '盛京银行', '鞍山银行', '抚顺银行', '本溪商业银行', 
            '丹东银行', '锦州银行', '营口银行', '营口沿海银行', '阜新银行', '辽阳银行', '铁岭银行', '朝阳银行', '盘锦银行', 
            '葫芦岛银行', '大连银行', '吉林银行', '龙江银行', '哈尔滨银行', '上海银行', '江苏银行', '南京银行', '长江商业银行', 
            '苏州银行', '杭州银行', '温州银行', '嘉兴银行', '湖州银行', '绍兴银行', '金华银行', '浙江稠州商业银行', '台州银行', 
            '浙江泰隆商业银行', '浙江民泰商业银行', '宁波银行', '宁波东海银行', '宁波通商银行', '徽商银行', '福建海峡银行', 
            '泉州银行', '厦门银行', '厦门国际银行', '江西银行', '九江银行', '赣州银行', '上饶银行', '齐鲁银行', '齐商银行', 
            '烟台银行', '潍坊银行', '临商银行', '威海商业银行', '日照银行', '德州银行', '莱商银行', '东营银行', '济宁银行', 
            '泰安银行', '枣庄银行', '青岛银行', '中原银行', '郑州银行', '洛阳银行', '平顶山银行', '焦作中旅银行', '湖北银行', 
            '汉口银行', '长沙银行', '华融湘江银行', '广州银行', '珠海华润银行', '东莞银行', '广东华兴银行', '广东南粤银行', 
            '广西北部湾银行', '柳州银行', '桂林银行', '海南银行', '重庆银行', '重庆三峡银行', '成都银行', '泸州商业银行', 
            '攀枝花商业银行', '宜宾商业银行', '乐山商业银行', '四川天府银行', '自贡银行', '遂宁银行', '绵阳商业银行', 
            '凉山商业银行', '雅安商业银行', '达州银行', '长城华西银行', '贵州银行', '贵阳银行', '富滇银行', '曲靖商业银行', 
            '云南红塔银行', '西藏银行', '长安银行', '西安银行', '兰州银行', '甘肃银行', '青海银行', '宁夏银行', '石嘴山银行', 
            '新疆银行', '乌鲁木齐银行', '昆仑银行', '新疆汇和银行', '库尔勒银行', '哈密商业银行',
            
            '浙江网商银行', '温州民商银行', '武汉众邦银行', '福建华通银行', '威海蓝海银行', '天津金城银行', '上海华瑞银行', 
            '深圳前海微众银行', '湖南三湘银行', '重庆富民银行', '四川新网银行', '北京中关村银行', '吉林亿联银行', '江苏苏宁银行', 
            '梅州客商银行', '安徽新安银行', '辽宁振兴银行', '江西裕民银行',
            
            '石家庄汇融农村合作银行', '广西百色右江农村合作银行', '锡林浩特农村合作银行', '广西平果农村合作银行', '二连浩特农村合作银行', 
            '广西象州农村合作银行', '通辽奈曼农村合作银行', '广西来宾桂中农村合作银行', '广西柳江农村合作银行', '广西贺州桂东农村合作银行', 
            '广西柳城农村合作银行', '昆明官渡农村合作银行', '广西兴安农村合作银行', '陕西富县农村合作银行', '广西阳朔农村合作银行', 
            '陕西吴起农村合作银行', '广西平乐农村合作银行', '甘肃榆中农村合作银行', '广西荔浦农村合作银行', '天水秦州农村合作银行', 
            '广西桂林漓江农村合作银行', '天水麦积农村合作银行', '广西永福农村合作银行', '甘肃宁县农村合作银行', '广西全州农村合作银行', 
            '陇南武都农村合作银行', '广西宜州农村合作银行', '新疆石河子农村合作银行',
            
            '汇丰银行', '星展银行', '摩根士丹利国际银行', '东亚银行', '瑞穗银行', '德意志银行', '南洋商业银行',  '三井住友银行', 
            '瑞士银行', '恒生银行', '三菱日联银行', '花旗银行', '首都银行', '摩根大通银行', '大新银行', '新联商业银行', '华美银行', 
            '华侨永亨银行', '盘谷银行', '蒙特利尔银行', '华商银行', '正信银行', '富邦华一银行', '韩亚银行', '永丰银行', 
            '浦发硅谷银行', '新韩银行', '玉ft银行', '开泰银行', '友利银行', '澳大利亚和新西兰银行', '彰银商业银行', '国民银行', 
            '法国巴黎银行', '国泰世华银行', '企业银行', '法国兴业银行', '渣打银行', '大华银行', '东方汇理银行',
        ]
        nongshangyinhang = ['北京', '天津', '上海', '无锡']
        self._char_bank_name += [i+'农商银行' for i in nongshangyinhang]
        
    def predict(self, image, axis=False, model=None):
        self._show_axis = axis
        self._info = '图片模糊或非银行卡图片'
        self._error = '图片模糊或非银行卡图片'
        
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
        for angle, gray in zip([0, 0, 90, 90, 270, 270, 180, 180], [0, 1,0, 1,0, 1,0, 1]):
            image = la.image.rotate(self._image, angle, expand=True)
            if gray==1:
                image1 = la.image.image_to_array(la.image.color_convert(image, la.image.ColorMode.grayscale))[:,:,0]
            else:
                image1 = la.image.image_to_array(image)
            self._result = model.ocr(image1, cls=False)
#             print(angle, gray, self._result)
            vertical = [1 if (i[0][1][0]-i[0][0][0])>(i[0][3][1]-i[0][0][1]) else 0 for i in self._result[0] if len(la.text.sequence_preprocess(i[1][0]))>1]
            if sum(vertical)/max(len(vertical), 0.1)<0.7:
                continue
            
            text = ' '.join([i[1][0] for i in self._result[0]])
            if sum([1 for char in self._char_direction if char in text])>2:
                self._image = image
                self._angle = angle
                self._info = {i:'图片模糊' for i in self._keys if i in self._name_list}
                break
    
    def _fit_axis(self):
        if len(self._result)==0:
            return 0
        
        axis_true = dict()
        bank_number = []
        for i in self._result[0]:
            h = (i[0][3][1]+i[0][2][1]-i[0][1][1]-i[0][0][1])/2
            w = (i[0][1][0]+i[0][2][0]-i[0][0][0]-i[0][3][0])/2
            x = min(i[0][0][0], i[0][3][0])
            y = min(i[0][0][1], i[0][1][1])
            if 'bank_number' not in axis_true and len(i[1][0])>3:
                temp1 = i[1][0].replace('b', '6').replace(' ', '').replace('-', '')
                temp = ''.join([char for char in temp1 if char in '0123456789'])
                if len(temp)==len(temp1) and 20>len(temp):
                    bank_number.append(([0, y-h*0.2, self._image.width, y+h*1.2], len(temp)))
                    continue
        if bank_number:
            axis_true['bank_number'] = sorted(bank_number, key=lambda x:x[1])[-1][0]
            
        self._axis = axis_true
            
    def _fit_characters(self, axis, result):
        if len(result)==0:
            return 0

        axis_true = {i:tuple(axis[i]) for i in axis}
        
        text = ' '.join([la.text.sequence_preprocess(i[1][0]) for i in self._result[0]])
#         print(text)
        if '图片模糊' in self._info.get('bank_name', ''):
            if [i for i in ['银', '行'] if i in text]:
                score = [(char, fuzz.partial_ratio(char, text)) for char in self._char_bank_name]
    #             print(sorted(score, key=lambda x:x[1]))
                score = sorted(score, key=lambda x:x[1])[-1]
                if score[1]>=60:
                    self._info['bank_name'] = score[0]
        if '图片模糊' in self._info.get('bank_type', ''):
            if [i for i in ['借记', '记卡', '储蓄', '蓄卡'] if i in text]:
                self._info['bank_type'] = '储蓄卡'
            elif [i for i in ['信用', '用卡'] if i in text]:
                self._info['bank_type'] = '信用卡'
        
        bank_number = []
        for i in result[0]:
            h = max((i[0][3][1]+i[0][2][1]-i[0][1][1]-i[0][0][1])/2, 1)
            w = max((i[0][1][0]+i[0][2][0]-i[0][0][0]-i[0][3][0])/2, 1)
            x = min(i[0][0][0], i[0][3][0])
            y = min(i[0][0][1], i[0][1][1])
#             if '图片模糊' in self._info.get('bank_name', ''):
#                 if '银' in i[1][0] or '行' in i[1][0]:
#                     score = [(char, fuzz.partial_ratio(char, i[1][0])) for char in self._char_bank_name]
# #                     print(sorted(score, key=lambda x:x[1]))
#                     score = sorted(score, key=lambda x:x[1])[-1]
#                     if score[1]>75:
#                         self._info['bank_name'] = score[0]
#                         continue
#             if '图片模糊' in self._info.get('bank_type', ''):
#                 if '借记' in i[1][0]:
#                     self._info['bank_type'] = '储蓄卡'
#                     continue
#                 elif '信用' in i[1][0] or '用卡' in i[1][0]:
#                     self._info['bank_type'] = '信用卡'
#                     continue
            if '图片模糊' in self._info.get('bank_number', '') and 'bank_number' in self._axis:
                h1 = min(max(i[0][3][1], i[0][2][1]), axis_true['bank_number'][3])-max(min(i[0][0][1], i[0][1][1]), axis_true['bank_number'][1])
                w1 = min(max(i[0][1][0], i[0][2][0]), axis_true['bank_number'][2])-max(min(i[0][0][0], i[0][3][0]), axis_true['bank_number'][0])            
                if h1/h>0.6 and w1/w>0.6:
                    temp = i[1][0].replace('b', '6')
                    temp = ''.join([char for char in temp if char in '0123456789'])
                    if 20>len(temp)>14:
                        if len(temp) in [15, 18]:
                            temp = '6'+temp
                        elif len(temp)==19 and temp[0] not in '469':
                            temp = '6'+temp[1:]
                    bank_number.append((temp, [x,y]+i[0][2]))
            
        if bank_number:
            if len(bank_number)==1 and len(bank_number[-1][0]) in [16,17,19]:
                self._info['bank_number'] = bank_number[-1][0]
                self._axis['bank_number'] = bank_number[-1][1]
            elif [i for i in bank_number if len(i) in [16,17,19]]:
                self._info['bank_number'] = [i for i,j in bank_number if len(i) in [16,17,19]][-1]
                self._axis['bank_number'] = [j for i,j in bank_number if len(i) in [16,17,19]][-1]
            else:
                temp = sorted(bank_number, key=lambda x:x[1][0])
                if len(''.join([i[0] for i in temp])) in [16,17,19]:
                    self._info['bank_number'] = ''.join([i[0] for i in temp])
                    self._axis['bank_number'] = temp[0][1][:2]+temp[-1][1][-2:]
                
        if '图片模糊' in self._info.get('bank_type', ''):
            self._info['bank_type'] = '储蓄卡'
    
    def draw_mask(self):
        image = self._image.copy()
        try:
            t = [la.image.box_convert(self._axis[i], 'xyxy', 'axis') for i in self._axis if i in self._keys]
            if len(t)>0:
                image = la.image.draw_box(image, t, width=2)
        except:
            pass
        return image
    
