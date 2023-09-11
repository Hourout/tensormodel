import time

import paddleocr
import linora as la

__all__ = ['OCRPOSPiao']


class OCRPOSPiao():
    def __init__(self, model=True, name_list=None):
        if model==True:
            self._model = paddleocr.PaddleOCR(show_log=False)
        elif model:
            self._model = model
        else:
            self._model = None
        self._keys = ['merchant_name', 'merchant_id', 'terminal_id', 'issuance_bank', 'acquiring_bank',
                      'voucher_id', 'authorization_id', 'batch_id', 'reference_id', 'trace_id', 'invoice_id',
                      'trade_type', 'trade_date', 'trade_id', 'trade_amount']
        if name_list is None:
            name_list = self._keys.copy()
        else:
            for i in name_list:
                if i not in self._keys:
                    raise ValueError(f'Variable name `{i}`  does not conform to the specification.')
        self._name_list = name_list
        self._char_merchant_name = ['商户名', '商广名']
        self._char_merchant_id = ['商户编号', '商广编号']
        self._char_terminal_id = ['终端编号', '冬端编号']
        self._char_trade_date = ['交易日期', '日期时间', '日期/时间']
    
    def predict(self, image, axis=False, model=None):
        self._show_axis = axis
        self._info = '图片模糊或非POS小票图片'
        self._error = '图片模糊或非POS小票图片'
        
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
        
        
        for g in self._info:
            for aug in [0,1,2]:
                error_list = [i for i in self._info[g] if '图片模糊' in self._info[g][i]]
                if error_list:
                    self._result_crop = []
                    image = la.image.crop(self._image, self._axis[g])
                    if aug==0:
                        image = la.image.image_to_array(image)
                    elif aug==1:
                        image = la.image.image_to_array(la.image.color_convert(image, la.image.ColorMode.grayscale))[:,:,0]
                    elif aug==2:
                        image = la.image.image_to_array(la.image.enhance_brightness(image, 0.65))
                    t = (self._model if model is None else model).ocr(image, cls=False)
                    if t[0]:
                        for j in t[0]:
                            self._result_crop.append([[self._axis[g][:2], [self._axis[g][2], self._axis[g][1]], 
                                                       self._axis[g][2:], [self._axis[g][0], self._axis[g][3]]], j[1]])
                    self._fit_characters(self._axis, [self._result_crop])

        info = self._analysis_concat()
        self._error = '图片模糊' if [1 for g in self._info for i in self._info[g] if '图片模糊' in self._info[g][i]] else 'ok'
#         self._info = {i:('' if '图片模糊' in j else j) for i,j in self._info.items()}
        if self._show_axis:
            return {'data':info, 'axis':self._axis, 'angle':self._angle, 'error':self._error}
        else:
            return {'data':info, 'angle':self._angle, 'error':self._error}
    
    def _fit_direction(self, model):
        for angle in [0, 90, 270, 180]:
            image = la.image.rotate(self._image, angle, expand=True)
            self._result = model.ocr(la.image.image_to_array(image), cls=False)
#             print(angle, result, '\n')
#             t = [1 if len(i[1][0])>4 and (i[0][1][0]-i[0][0][0])<(i[0][3][1]-i[0][0][1]) else 0 for i in self._result[0]]
#             print(sum(t)/len(t))
#             if sum(t)/len(t)>0.04:
#                 continue
            
#             print(angle, result, '\n')
            rank = [0,0,0,0,0]
            for r, i in enumerate(self._result[0], start=1):
                if [char for char in self._char_merchant_name if char in i[1][0]] and sum(rank)==0:
                    rank[0] = r
                elif [char for char in self._char_merchant_id if char in i[1][0]] and sum(rank[1:])==0:
                    rank[1] = r
                elif [char for char in self._char_terminal_id if char in i[1][0]] and sum(rank[2:])==0:
                    rank[2] = r
                elif '交易类型' in i[1][0] and sum(rank[3:])==0:
                    rank[3] = r
                elif [1 for char in ['凭证号', '授权码', '批次号', '参考号', '交易日期'] if char in i[1][0]] and sum(rank[4:])==0:
                    rank[4] = r
#                 elif '持卡人签名' in i[1][0] and sum(rank[5:])==0:
#                     rank[5] = r
#             print(angle, rank)
            rank = [i for i in rank if i>0]
            if rank==sorted(rank) and len(rank)>1:
                self._image = image
                self._angle = angle
                self._info = {'group0':{i:'图片模糊' for i in self._keys if i in self._name_list}}
                break
    
    def _fit_axis(self):
        if len(self._result)==0:
            return 0

        axis_true = dict()
        group = 0
        for i in self._result[0]:
            h = (i[0][3][1]+i[0][2][1]-i[0][1][1]-i[0][0][1])/2
            w = (i[0][1][0]+i[0][2][0]-i[0][0][0]-i[0][3][0])/2
            x = min(i[0][0][0], i[0][3][0])
            y = min(i[0][0][1], i[0][1][1])
            
            logic = False
            for g in axis_true:
                h1 = min(max(i[0][3][1], i[0][2][1]), axis_true[g][3])-max(min(i[0][0][1], i[0][1][1]), axis_true[g][1])
                w1 = min(max(i[0][1][0], i[0][2][0]), axis_true[g][2])-max(min(i[0][0][0], i[0][3][0]), axis_true[g][0])            
                if h1/h>0.6 and w1/w>0.6:
                    logic = True
                    break
            if logic:
                continue
                
            if [char for char in self._char_merchant_name if char in i[1][0]]:
                if len(i[1][0])>5 and len(la.text.sequence_preprocess(i[1][0]))<5:
                    w = w*4/((len(i[1][0])-4)/2.5+4)
                elif len(i[1][0])>5 and len(la.text.sequence_preprocess(i[1][0]))>=5:
                    w = w*4/len(i[1][0])
                axis_true[f'group{group}'] = [x-w*0.25, y-h*0.5, x+w*4.5, y+h*24]
                group += 1
                continue
            if [char for char in self._char_merchant_id if char in i[1][0]]:
                if len(i[1][0])>5:
                    w = w*4/((len(i[1][0])-4)/2.5+4)
                axis_true[f'group{group}'] = [x-w*0.25, y-h*1.5, x+w*4.5, y+h*23]
                group += 1
                continue
            if [char for char in self._char_terminal_id if char in i[1][0]]:
                if len(i[1][0])>5:
                    w = w*4/((len(i[1][0])-4)/2.5+4)
                axis_true[f'group{group}'] = [x-w*0.25, y-h*2.5, x+w*4.5, y+h*22]
                group += 1
                continue
            if '交易类型' in i[1][0]:
                if len(i[1][0])>5:
                    w = w*4/(len(i[1][0])-0.5)
                axis_true[f'group{group}'] = [x-w*0.25, y-h*7, x+w*3.5, y+h*9]
                group += 1
                continue
        self._axis = axis_true
        if axis_true:
            self._info = {i:self._info['group0'].copy() for i in axis_true}
    
    def _fit_characters(self, axis, result):
        if len(result)==0:
            return 0
        axis_true = {i:tuple(axis[i]) for i in axis}
        
        for i in result[0]:
            h = max((i[0][3][1]+i[0][2][1]-i[0][1][1]-i[0][0][1])/2, 1)
            w = max((i[0][1][0]+i[0][2][0]-i[0][0][0]-i[0][3][0])/2, 1)
            x = min(i[0][0][0], i[0][3][0])
            y = min(i[0][0][1], i[0][1][1])
            
            for g in axis_true:
                h1 = min(max(i[0][3][1], i[0][2][1]), axis_true[g][3])-max(min(i[0][0][1], i[0][1][1]), axis_true[g][1])
                w1 = min(max(i[0][1][0], i[0][2][0]), axis_true[g][2])-max(min(i[0][0][0], i[0][3][0]), axis_true[g][0])            
                if h1/h>0.6 and w1/w>0.6:
                    temp = i[1][0].replace('：', ':').replace('）', ')').replace('（', '(')
                    if '图片模糊' in self._info[g].get('merchant_name', ''):
                        char = [char for char in self._char_merchant_name if char in temp]
                        if char:
                            char = temp.split(char[0])[-1].split(':')[-1]
                            if len(char)>1:
                                self._info[g]['merchant_name'] = char
                                break
                    if '图片模糊' in self._info[g].get('merchant_id', ''):
                        char = [char for char in self._char_merchant_id if char in temp]
                        if char:
                            char = temp.split(char[0])[-1].split(':')[-1]
                            if len(char)>3:
                                self._info[g]['merchant_id'] = char
                                break
                    if '图片模糊' in self._info[g].get('terminal_id', ''):
                        char = [char for char in self._char_terminal_id if char in temp]
                        if char:
                            char = temp.split(char[0])[-1].split(':')[-1]
                            if len(char)>3:
                                self._info[g]['terminal_id'] = char
                                break
                    if '图片模糊' in self._info[g].get('issuance_bank', ''):
                        if sum([1 for char in ['发','卡','行','ISSUER'] if char in temp])>1:
                            t = temp.lower().split('发卡行')[-1].split('issuer')[-1].replace('(', '').replace(')', '')
                            index = [char for r,char in enumerate(['收单', '单行', 'ACQ']) if char in t]
                            t = t[:t.find(index[0])] if index else t
                            t = t[t.find(':')+1:]
                            if len(t)>2:
                                self._info[g]['issuance_bank'] = t.replace(' ', '')
                    if '图片模糊' in self._info[g].get('acquiring_bank', ''):
                        if sum([1 for char in ['收','单','行','ACQ'] if char in temp])>1:
                            t = temp.lower().split('收单行')[-1].split('acq')[-1].replace('(', '').replace(')', '')
                            t = t[t.find(':')+1:]
                            if len(t)>2:
                                self._info[g]['acquiring_bank'] = t
                    if '图片模糊' in self._info[g].get('voucher_id', ''):
                        if '凭证号' in temp or 'voucher' in temp.lower():
                            t = temp.lower().split('凭证号')[-1].split('voucher')[-1].replace(' ', '')
                            t = t[t.find(':')+1:]
                            number = [(r,char) for r,char in enumerate(t) if char in '0123456789']
                            if len(number)>3:
                                t = t[number[0][0]:]
                                index = [r for r,char in enumerate(t) if char not in '0123456789']
                                self._info[g]['voucher_id'] = t[:min(index)] if index else t
                    if '图片模糊' in self._info[g].get('authorization_id', ''):
                        if '授权码' in temp or 'auth' in temp.lower():
                            t = temp.lower().split('授权码')[-1].split('auth')[-1].replace(' ', '')
                            t = t[t.find(':')+1:]
                            number = [(r,char) for r,char in enumerate(t) if char in '0123456789']
                            if len(number)>3:
                                t = t[number[0][0]:]
                                index = [r for r,char in enumerate(t) if char not in '0123456789']
                                self._info[g]['authorization_id'] = t[:min(index)] if index else t
                    if '图片模糊' in self._info[g].get('batch_id', ''):
                        if '批次号' in temp or 'batch' in temp.lower():
                            t = temp.lower().split('批次号')[-1].split('batch')[-1].replace(' ', '')
                            t = t[t.find(':')+1:]
                            number = [(r,char) for r,char in enumerate(t) if char in '0123456789']
                            if len(number)>3:
                                t = t[number[0][0]:]
                                index = [r for r,char in enumerate(t) if char not in '0123456789']
                                self._info[g]['batch_id'] = t[:min(index)] if index else t
                    if '图片模糊' in self._info[g].get('reference_id', ''):
                        if '参考号' in temp or 'ref' in temp.lower():
                            t = temp.lower().split('参考号')[-1].split('ref')[-1].replace(' ', '')
                            t = t[t.find(':')+1:]
                            number = [(r,char) for r,char in enumerate(t) if char in '0123456789']
                            if len(number)>3:
                                t = t[number[0][0]:]
                                index = [r for r,char in enumerate(t) if char not in '0123456789']
                                self._info[g]['reference_id'] = t[:min(index)] if index else t
                    if '图片模糊' in self._info[g].get('trace_id', ''):
                        if '流水号' in temp or 'trace' in temp.lower():
                            t = temp.lower().split('流水号')[-1].split('trace')[-1].replace(' ', '')
                            t = t[t.find(':')+1:]
                            number = [(r,char) for r,char in enumerate(t) if char in '0123456789']
                            if len(number)>3:
                                t = t[number[0][0]:]
                                index = [r for r,char in enumerate(t) if char not in '0123456789']
                                self._info[g]['trace_id'] = t[:min(index)] if index else t
                    if '图片模糊' in self._info[g].get('invoice_id', ''):
                        if '票据号' in temp or 'invoice' in temp.lower():
                            t = temp.lower().split('票据号')[-1].split('invoice')[-1].replace(' ', '')
                            t = t[t.find(':')+1:]
                            number = [(r,char) for r,char in enumerate(t) if char in '0123456789']
                            if len(number)>3:
                                t = t[number[0][0]:]
                                index = [r for r,char in enumerate(t) if char not in '0123456789']
                                self._info[g]['invoice_id'] = t[:min(index)] if index else t

                    if '图片模糊' in self._info[g].get('trade_type', ''):
                        t = [char for char in ['交易类型'] if char in temp]
                        if t:
                            t = temp.split(t[0])[-1]
                            t = t[t.find(':')+1:]
                            if len(t)>1:
                                self._info[g]['trade_type'] = t
                                break
                    if '图片模糊' in self._info[g].get('trade_date', ''):
                        t = [char for char in self._char_trade_date if char in temp]
                        if t:
                            t = temp.split(t[0])[-1]
                            t = t[t.find(':')+1:]
                            if len(t)>1:
                                t = t.replace('/', '-')
                                if t[10]!=' ':
                                    t = t[:10]+' '+t[10:]
                                self._info[g]['trade_date'] = t
                                break
                    if '图片模糊' in self._info[g].get('trade_id', ''):
                        if '订单号' in temp and len(temp)>5:
                            self._info[g]['trade_id'] = temp.split('订单号')[-1].split(':')[-1]
                            break
                    if '图片模糊' in self._info[g].get('trade_amount', ''):
                        if 'RMB' in temp.upper() and len(temp)>5:
                            self._info[g]['trade_amount'] = '¥'+temp.upper().replace(' ', '').split('RMB')[-1].split(':')[-1]
                            break
        
        for g in axis_true:
            if [1 for i in self._info[g] if '图片模糊' in self._info[g].get(i, '')]:
                for r,i in enumerate(result[0]):
                    h = max((i[0][3][1]+i[0][2][1]-i[0][1][1]-i[0][0][1])/2, 1)
                    w = max((i[0][1][0]+i[0][2][0]-i[0][0][0]-i[0][3][0])/2, 1)
                    h1 = min(max(i[0][3][1], i[0][2][1]), axis_true[g][3])-max(min(i[0][0][1], i[0][1][1]), axis_true[g][1])
                    w1 = min(max(i[0][1][0], i[0][2][0]), axis_true[g][2])-max(min(i[0][0][0], i[0][3][0]), axis_true[g][0])
                    if h1/h>0.6 and w1/w>0.6 and '图片模糊' in self._info[g].get('merchant_name', ''):
                        if [char for char in self._char_merchant_name if char in result[0][max(r-1,0)][1][0]]:
                            if sum([1 for char in self._char_merchant_id if char in i[1][0]])==0:
                                self._info[g]['merchant_name'] = i[1][0]
                                continue
                    if h1/h>0.6 and w1/w>0.6 and '图片模糊' in self._info[g].get('merchant_id', ''):
                        if [char for char in self._char_merchant_id if char in result[0][max(r-1,0)][1][0]]:
                            if sum([1 for char in self._char_terminal_id if char in i[1][0]])==0:
                                self._info[g]['merchant_id'] = i[1][0]
                                continue
                    if h1/h>0.6 and w1/w>0.6 and '图片模糊' in self._info[g].get('terminal_id', ''):
                        if [char for char in self._char_terminal_id if char in result[0][max(r-1,0)][1][0]]:
                            if sum([1 for char in i[1][0] if char in ':0123456789'])==len(i[1][0]):
                                self._info[g]['terminal_id'] = i[1][0].replace(':','')
                                continue
                    
                    if h1/h>0.6 and w1/w>0.6 and '图片模糊' in self._info[g].get('voucher_id', ''):
                        if [char for char in ['凭证号', 'VOUCHER'] if char in result[0][max(r-1,0)][1][0]]:
                            if sum([1 for char in i[1][0] if char in ':0123456789'])==len(i[1][0]):
                                self._info[g]['voucher_id'] = i[1][0].replace(':','')
                                continue
                    if h1/h>0.6 and w1/w>0.6 and '图片模糊' in self._info[g].get('authorization_id', ''):
                        if [char for char in ['授权码', 'AUTH'] if char in result[0][max(r-1,0)][1][0]]:
                            if sum([1 for char in i[1][0] if char in ':0123456789'])==len(i[1][0]):
                                self._info[g]['authorization_id'] = i[1][0].replace(':','')
                                continue
                    if h1/h>0.6 and w1/w>0.6 and '图片模糊' in self._info[g].get('batch_id', ''):
                        if [char for char in ['批次号', 'BATCH'] if char in result[0][max(r-1,0)][1][0]]:
                            if sum([1 for char in i[1][0] if char in ':0123456789'])==len(i[1][0]):
                                self._info[g]['batch_id'] = i[1][0].replace(':','')
                                continue
                    if h1/h>0.6 and w1/w>0.6 and '图片模糊' in self._info[g].get('trace_id', ''):
                        if [char for char in ['流水号', 'TRACE'] if char in result[0][max(r-1,0)][1][0]]:
                            if sum([1 for char in i[1][0] if char in ':0123456789'])==len(i[1][0]):
                                self._info[g]['trace_id'] = i[1][0].replace(':','')
                                continue
                    if h1/h>0.6 and w1/w>0.6 and '图片模糊' in self._info[g].get('invoice_id', ''):
                        if [char for char in ['票据号', 'INVOICE'] if char in result[0][max(r-1,0)][1][0]]:
                            if sum([1 for char in i[1][0] if char in ':0123456789'])==len(i[1][0]):
                                self._info[g]['invoice_id'] = i[1][0].replace(':','')
                                continue
                    if h1/h>0.6 and w1/w>0.6 and '图片模糊' in self._info[g].get('reference_id', ''):
                        if [char for char in ['参考号', 'REF'] if char in result[0][max(r-1,0)][1][0]]:
                            if sum([1 for char in i[1][0] if char in ':0123456789'])==len(i[1][0]):
                                self._info[g]['reference_id'] = i[1][0].replace(':','')
                                continue
                    
                    
                    if h1/h>0.6 and w1/w>0.6 and '图片模糊' in self._info[g].get('trade_type', ''):
                        if [char for char in ['交易类型'] if char in result[0][max(r-1,0)][1][0]]:
                            for char in ['消费', '转账']:
                                if char in i[1][0]:
                                    self._info[g]['trade_type'] = char
                                    break
                    if h1/h>0.6 and w1/w>0.6 and '图片模糊' in self._info[g].get('trade_date', ''):
                        if [char for char in self._char_trade_date if char in result[0][max(r-1,0)][1][0]]:
                            if sum([1 for char in '金额AMOUNT' if char in i[1][0]])==0:
                                char = i[1][0].replace('/', '-')
                                if char[10]!=' ':
                                    char = char[:10]+' '+char[10:]
                                self._info[g]['trade_date'] = char
                                continue
        
    def _analysis_concat(self):
        info = self._info.copy()
        if len(self._info)>1:
            for key in ['reference_id', 'trade_date', 'trade_amount', 'trade_id']:
                temp = set([info[g][key] for g in info if info[g].get(key, '') not in ['图片模糊', '']])
#                 print(temp)
                for i in temp:
#                     print(info,'\n')
                    t = [info[g] for g in info if info[g].get(key, '')==i]
                    t = {char:[j[char] for j in t if '图片模糊' not in j[char]] for char in self._name_list}
                    t = {char:t[char][0] if t[char] else '图片模糊' for char in t}
                    info = [info[g] for g in info if info[g].get(key, '') not in [i, '']]
                    info.append(t)
                    info = {f'group{r}':j for r,j in enumerate(info)}
#                     print(info,'\n')
        t  = ['trade_id', 'reference_id', 'voucher_id', 'trade_amount']
        g = [(g,sum([float(info[g][i][1:4]) if '图片模糊' not in info[g][i] else 0 for i in t])) for g in info]
        info = {i[0]:info[i[0]] for i in sorted(g, key=lambda x:x[1])}
        return info
                
    def draw_mask(self):
        image = self._image.copy()
        try:
            t = [la.image.box_convert(self._axis[i], 'xyxy', 'axis') for i in self._axis]
            if len(t)>0:
                image = la.image.draw_box(image, t, width=2)
        except:
            pass
        return image
    
    def metrics(self, data, image_root, name_list=None, debug=False, test_sample_nums=None):
        if la.gfile.isfile(data):
            with open(data) as f:
                data = f.read().strip().split('\n')
            data = [eval(i) for i in data]
        if name_list is None:
            name_list = ['merchant_name', 'merchant_id', 'terminal_id', 'issuance_bank', 'acquiring_bank',
                         'voucher_id', 'authorization_id', 'batch_id', 'reference_id', 'trace_id', 'invoice_id',
                         'trade_type', 'trade_date', 'trade_id', 'trade_amount']
        
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
                    if len(i)==len(t):
                        for g in i:
                            for j in name_list:
                                if j in i[g]:
                                    if j in t[g]:
                                        if t[g][j]==i[g][j]:
                                            score_a[j] +=1
                                        else:
                                            error[j] = {'pred':t[g][j], 'label':i[g][j], 'group':g}
                    else:
                        error['error'] = 'group count error'
                else:
                    error['error'] = t
            except:
                error['error'] = 'program error'
            for j in name_list:
                for g in i:
                    if j in i[g]:
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

