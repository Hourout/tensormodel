import time

import paddleocr
import linora as la


class OCRLvMaHeYan():
    def __init__(self, model=True, name_list=None):
        if model==True:
            self._model = paddleocr.PaddleOCR(show_log=False)
        elif model:
            self._model = model
        else:
            self._model = None
        self._keys = ['check_id', 'check_effective_time', 'check_purpose', 'check_status', 'check_serial_number']
        if name_list is None:
            name_list = self._keys.copy()
        else:
            for i in name_list:
                if i not in self._keys:
                    raise ValueError(f'Variable name `{i}`  does not conform to the specification.')
        self._name_list = name_list
        
    def predict(self, image, axis=False, model=None):
        self._show_axis = axis
        self._info = '图片模糊或非核验图片'
        self._error = '图片模糊或非核验图片'
        
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
                if i not in self._axis:
                    continue
                image = la.image.crop(self._image, self._axis[i])
                t = (self._model if model is None else model).ocr(la.image.image_to_array(image), cls=False)
                if t[0]:
                    for j in t[0]:
                        if i.startswith('tax_remark'):
                            i = 'tax_remark'
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
            rank = [0,0,0,0,0,0,0]
            for r, i in enumerate(self._result[0], start=1):
                if '核验编号' in i[1][0]:
                    if rank[0]==0:
                        rank[0] = r
                elif '核验用途' in i[1][0]:
                    if rank[1]==0:
                        rank[1] = r
                elif '申请时间' in i[1][0]:
                    if rank[2]==0:
                        rank[2] = r
                elif '有效时间' in i[1][0]:
                    if rank[3]==0:
                        rank[3] = r
                elif '家庭类型' in i[1][0]:
                    if rank[4]==0:
                        rank[4] = r
                elif '家庭描述' in i[1][0]:
                    if rank[5]==0:
                        rank[5] = r
                elif '拟购房区' in i[1][0]:
                    if rank[6]==0:
                        rank[6] = r
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
        for i in self._result[0]:
            h = (i[0][3][1]+i[0][2][1]-i[0][1][1]-i[0][0][1])/2
            w = (i[0][1][0]+i[0][2][0]-i[0][0][0]-i[0][3][0])/2
            x = min(i[0][0][0], i[0][3][0])
            y = min(i[0][0][1], i[0][1][1])
            if 'check_id' not in axis_true and i[1][0] in ['核验编号', '核验编号：']:
                axis_true['check_id'] = [x+w*1.2, y-h*0.25, x+w*5, y+h*1.25]
                continue
            if 'check_effective_time' not in axis_true and i[1][0] in ['有效时间', '有效时间：']:
                axis_true['check_effective_time'] = [x+w*1.2, y-h*0.25, x+w*5, y+h*1.25]
                continue
            if 'check_purpose' not in axis_true and i[1][0] in ['核验用途', '核验用途：']:
                axis_true['check_purpose'] = [x+w*1.2, y-h*0.25, x+w*5, y+h*1.25]
                continue
#             if 'check_status' not in axis_true:
#                 axis_true['check_status'] = [x+w*1.2, y-h*0.25, x+w*4, y+h*1.25]
#                 continue
#             if 'check_serial_number' not in axis_true:
#                 axis_true['check_serial_number'] = [x+w*1.2, y-h*0.25, x+w*4, y+h*1.25]
#                 continue

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
        for i in result[0]:
            h = (i[0][3][1]+i[0][2][1]-i[0][1][1]-i[0][0][1])/2
            w = (i[0][1][0]+i[0][2][0]-i[0][0][0]-i[0][3][0])/2
            if h==0:
                h = 1
            if w==0:
                w = 1
            x = min(i[0][0][0], i[0][3][0])
            y = min(i[0][0][1], i[0][1][1])
            if '图片模糊' in self._info.get('check_id', '') and 'check_id' in axis_true:
                h1 = min(max(i[0][3][1], i[0][2][1]), axis_true['check_id'][3])-max(min(i[0][0][1], i[0][1][1]), axis_true['check_id'][1])
                w1 = min(max(i[0][1][0], i[0][2][0]), axis_true['check_id'][2])-max(min(i[0][0][0], i[0][3][0]), axis_true['check_id'][0])            
                if h1/h>0.6 and w1/w>0.6:
                    temp = i[1][0].replace(' ', '')
                    if len(temp)==sum([1 for char in temp if char in '0123456789']):
                        self._info['check_id'] = temp
                        self._axis['check_id'] = [x, y]+i[0][2]
                        continue
            if '图片模糊' in self._info.get('check_effective_time', '') and 'check_effective_time' in axis_true:
                h1 = min(max(i[0][3][1], i[0][2][1]), axis_true['check_effective_time'][3])-max(min(i[0][0][1], i[0][1][1]), axis_true['check_effective_time'][1])
                w1 = min(max(i[0][1][0], i[0][2][0]), axis_true['check_effective_time'][2])-max(min(i[0][0][0], i[0][3][0]), axis_true['check_effective_time'][0])            
                if h1/h>0.6 and w1/w>0.6:
                    temp = i[1][0].replace(' ', '')
                    if len(temp)==sum([1 for char in temp if char in '0123456789至-——']):
                        self._info['check_effective_time'] = temp
                        self._axis['check_effective_time'] = [x, y]+i[0][2]
                        continue
            if '图片模糊' in self._info.get('check_purpose', '') and 'check_purpose' in axis_true:
                h1 = min(max(i[0][3][1], i[0][2][1]), axis_true['check_purpose'][3])-max(min(i[0][0][1], i[0][1][1]), axis_true['check_purpose'][1])
                w1 = min(max(i[0][1][0], i[0][2][0]), axis_true['check_purpose'][2])-max(min(i[0][0][0], i[0][3][0]), axis_true['check_purpose'][0])            
                if h1/h>0.6 and w1/w>0.6:
                    self._info['check_purpose'] = i[1][0]
                    self._axis['check_purpose'] = [x, y]+i[0][2]
                    continue
            if '图片模糊' in self._info.get('check_purpose', '') and '核验' in i[1][0]:
                if i[1][0] in ['核验通过', '初步核验通过']:
                    self._info['check_status'] = i[1][0]
                    self._axis['check_status'] = [x, y]+i[0][2]
                    continue
            
    
        for i in self._axis:
            self._axis[i] = [int(max(0, j)) for j in self._axis[i]]
    
    def draw_mask(self, image=None, axis=None, box_axis='all', mask_axis=None):
        angle = self._angle if axis is None else axis['angle']
        axis = self._axis if axis is None else axis['axis']
        if image is None:
            image = self._image.copy()
        else:
            image = la.image.rotate(image, angle, expand=True)
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
        
    def metrics(self, data, image_root, name_list=None, debug=False, test_sample_nums=None):
        if la.gfile.isfile(data):
            with open(data) as f:
                data = f.read().replace('\n', '').replace('}', '}\n').strip().split('\n')
            data = [eval(i) for i in data]
        if name_list is None:
            name_list = ['check_id', 'check_effective_time', 'check_purpose', 'check_status', 'check_serial_number']

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

        score = {f'{i}_acc':score_a[i]/max(score_b[i], 0.0000001) for i in score_a}
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

