# Ultralytics YOLO 🚀, AGPL-3.0 license

from pathlib import Path, PosixPath
import os

import numpy as np
import torch

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import LOGGER, ops
from ultralytics.utils.metrics import DetMetrics, box_iou, ReIDMetrics
from ultralytics.utils.torch_utils import smart_inference_mode
from ultralytics.utils.loss import StateMetrics #￥#添加人员状态预测评估指标


class JDEValidator(DetectionValidator):
    """
    A class extending the DetectionValidator class for validation based on a joint detection and embedding model.

    Example:
        ```python
        from ultralytics.models.yolo.jde import JDEValidator

        args = dict(model="yolov8n-jde.pt", data="coco8-seg.yaml")
        validator = JDEValidator(args=args)
        validator()
        ```
    """

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """Initialize SegmentationValidator and set task to 'segment', metrics to SegmentMetrics."""
        # ========== 在调用super()之前提取自定义参数 ==========
        # 从args中提取自定义参数（如果args是字典）
        if args is not None and isinstance(args, dict):
            # 提取自定义参数
            self._model_name = args.pop('model_name', None)
            self._save_excel = args.pop('save_excel', False)
            self._excel_save_dir = args.pop('excel_save_dir', None)
            self._excel_name = args.pop('excel_name', 'result_all.xlsx')
            self._save_tag_to_txt = args.pop('save_tag_to_txt', False)
        else:
            # 如果args不是字典或为None，设置默认值
            self._model_name = None
            self._save_excel = False
            self._excel_save_dir = None
            self._excel_name = 'result_all.xlsx'
            self._save_tag_to_txt = False
        
        # 现在调用super().__init__()，此时args中已经不包含自定义参数了
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.plot_masks = None
        self.process = None
        #￥#self.args.task = "jde"  # 确保task设置为jde，以支持6列标签格式
        self.metrics = DetMetrics(save_dir=self.save_dir, on_plot=self.on_plot)
        self.reid_metrics = ReIDMetrics()
        #model.person_states = data.get("person_states", {})
        
        # 添加状态预测指标 #￥#添加人员状态预测评估指标
        #self.state_metrics = StateMetrics(num_states=166) #&#初始化不能访问model #根据实际状态数量设置 #￥#添加人员状态预测评估指标
        self.state_metrics = None #&#初始化不能访问model替换
        self._num_states_hint = getattr(self.args, "state_classes", None)  # #&#初始化不能访问model替换 可选：从 args 提示
        # 存储状态预测数据用于批处理评估
        self.all_pred_states = [] #￥#添加人员状态预测评估指标
        self.all_target_states = [] #￥#添加人员状态预测评估指标
        self.state_iou = 0.5  # 默认0.5，0.75等
        
        # 添加状态检测指标（复用检测指标计算，使用相同的IoU阈值和置信度阈值）
        self.state_det_metrics = DetMetrics(save_dir=self.save_dir, on_plot=self.on_plot)
        # 存储状态检测的统计数据（类似self.stats）
        self.state_det_stats = {
            "conf": [],
            "pred_cls": [],
            "tp": [],
            "target_cls": [],
            "target_img": [],
        }

    @smart_inference_mode()
    def __call__(self, trainer=None, model=None):
        """Performs validation on the model and sets the epoch and best attributes."""
        if trainer is None and model is not None : #￥#评估阶段用
            self.model_path = model if (isinstance(model, str) or isinstance(model, PosixPath)) else model.pt_path
            if hasattr(model, 'model') and hasattr(model.model[-1], 'state_classes'):
                self.model = model #$#添加人员状态预测评估指标
        elif trainer is not None: #$#训练验证阶段用
            self.epoch = trainer.epoch + 1
            self.best = trainer.best
            self.trainer = trainer #￥#
            if model is None and hasattr(trainer, 'model'): #￥#
                self.model = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model
                #self.model = trainer.model.module #￥#
            elif model is not None:
                self.model_path = model if (isinstance(model, str) or isinstance(model, PosixPath)) else model.pt_path
                self.model = model #$#添加人员状态预测评估指标        
        
        # 确保在这里完成 state_metrics 的初始化 #$$$$$$$$$$$$$#  #&#初始化不能访问model替换
        self._ensure_state_metrics_initialized()#$$$$$$$$$$$$$# #&#初始化不能访问model替换

        stats = super().__call__(trainer, model)
        return stats
    def _ensure_state_metrics_initialized(self): #$$$$$$$$$$$$$# #&#初始化不能访问model替换
        if self.state_metrics is not None: #&#初始化不能访问model替换
            return  # 已经初始化过 #&#初始化不能访问model替换

        # 1) 优先从 args 里拿（可选，若你在外部配置过）
        num_states = self._num_states_hint #&#初始化不能访问model替换
        
        # 3) 从模型检测头读取state_classes
        if num_states is None and self.model is not None:
            head = getattr(self.model, "model", None)  # 可能是 nn.Sequential/ModuleList/自定义容器
            last = None
            if head is not None:
                # 常见两种结构：list-like 或再嵌套一层 .model
                if hasattr(head, "__getitem__"):
                    last = head[-1]
                else:
                    inner = getattr(head, "model", None)
                    if inner is not None and hasattr(inner, "__getitem__"):
                        last = inner[-1]
            if last is not None:
                num_states = getattr(last, "state_classes", None)

        # 4) 如果还没有拿到，使用默认值而不是报错
        if num_states is None:
            print("警告：无法获取state_classes，使用默认值166")
            num_states = 166  # 使用默认值，避免报错

        self.state_metrics = StateMetrics(num_states=int(num_states))
    def _prepare_batch(self, si, batch):
        """Prepares a batch of images and annotations for validation."""
        idx = batch["batch_idx"] == si
        cls = batch["cls"][idx].squeeze(-1)
        tags = batch["tags"][idx].squeeze(-1)
        bbox = batch["bboxes"][idx]
        ori_shape = batch["ori_shape"][si]
        imgsz = batch["img"].shape[2:]
        if len(cls):
            bbox = ops.xywh2xyxy(bbox) * torch.tensor(imgsz, device=self.device)[[1, 0, 1, 0]]  # target boxes
            ops.scale_boxes(imgsz, bbox, ori_shape)  # native-space labels
        return {"cls": cls, "bbox": bbox, "ori_shape": ori_shape, "imgsz": imgsz, "tags": tags}

    def _prepare_pred(self, pred, pbatch):
        """Prepares a batch of images and annotations for validation."""
        predn = pred.clone()
        ops.scale_boxes(pbatch["imgsz"], predn[:, :4], pbatch["ori_shape"])  # native-space pred
        return predn

    def update_metrics(self, preds, batch):
        """Metrics."""
        batch_matched_tags = []
        
        # 清空当前批次的状态数据
        self.current_batch_pred_states = [] #￥#人员状态预测评估指标
        self.current_batch_target_states = [] #$#人员状态预测评估指标
        
        for si, pred in enumerate(preds):
            self.seen += 1
            npr = len(pred)
            stat = dict(
                conf=torch.zeros(0, device=self.device),
                pred_cls=torch.zeros(0, device=self.device),
                tp=torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device),
            )
            matched_tags = torch.zeros(npr, dtype=torch.int, device=self.device)    # Initialize matched tags tensor
            pbatch = self._prepare_batch(si, batch)
            cls, bbox, tags = pbatch.pop("cls"), pbatch.pop("bbox"), pbatch.pop("tags")
            nl = len(cls)
            stat["target_cls"] = cls
            stat["target_img"] = cls.unique()
            if npr == 0:
                if nl:
                    for k in self.stats.keys():
                        self.stats[k].append(stat[k])
                    if self.args.plots:
                        self.confusion_matrix.process_batch(detections=None, gt_bboxes=bbox, gt_cls=cls)
                batch_matched_tags.append(matched_tags)#%￥￥%￥# 添加空的matched_tags，确保列表长度匹配
                continue

            # Predictions
            if self.args.single_cls:
                pred[:, 5] = 0
            predn = self._prepare_pred(pred, pbatch)
            stat["conf"] = predn[:, 4]
            stat["pred_cls"] = predn[:, 5]
            """#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            # 1) 取状态分支的概率，生成“状态=6类检测”的 conf 与 cls
            embed_dim = self.model.model[-1].embed_dim #￥#人员状态预测评估指标
            state_classes = self.model.model[-1].state_classes
            state_probs = pred[:, 6 + embed_dim : 6 + embed_dim + state_classes]          # (N, 6)
            state_conf, state_cls = state_probs.max(1, keepdim=True)   
        
            # 2) 构造“状态检测”的预测矩阵：[x1,y1,x2,y2, conf_state, cls_state]
            pred_state_det = torch.cat([predn[:, :4], state_conf, state_cls.float()], dim=1)
            print(pred_state_det)

            # 3) 用“状态id”作为GT类别（tags是1-based，改成0-based）
            # 构造 GT 状态类别，保证是一维 (M,)；无 GT 时返回 shape=(0,) 的空张量
            if tags is not None and tags.numel() > 0:
                gt_state_cls = (tags-1).to(dtype=torch.long, device=pred.device).view(-1)  # (M,)
            else:
                gt_state_cls = torch.zeros(0, dtype=torch.long, device=pred.device)          # (0,)

            # 同步构造占位的 gt_tags（一维），match_predictions 里会用到但这里不关心内容
            gt_dummy_tags = torch.zeros_like(gt_state_cls, dtype=torch.long, device=pred.device)  # (M,)

            # 再调用匹配与统计
            if pred_state_det.numel() > 0 or gt_state_cls.numel() > 0:
                stat["tp"], matched_tags = self._process_batch(pred_state_det, bbox, gt_state_cls, gt_dummy_tags)
            else:
                # 两者皆空时，按空统计处理
                stat["tp"] = torch.zeros(0, self.niou, dtype=torch.bool, device=self.device)
                matched_tags = torch.zeros(0, dtype=torch.long, device=self.device)"""
 #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            # Evaluate #$#$#$#$ 
            if nl:
                stat["tp"], matched_tags = self._process_batch(predn, bbox, cls, tags)  
                if self.args.plots:
                    self.confusion_matrix.process_batch(predn, bbox, cls)
            for k in self.stats.keys():
                self.stats[k].append(stat[k])
            batch_matched_tags.append(matched_tags)

            # ========== 状态检测指标计算（复用检测指标代码，完全替换类别和置信度）==========
            if hasattr(self, "state_det_stats") and hasattr(self.model.model[-1], "state_classes") and self.model.model[-1].state_classes is not None:
                embed_dim = self.model.model[-1].embed_dim
                state_classes = self.model.model[-1].state_classes
                
                # 提取状态预测概率（从NMS前的pred中提取，然后匹配到predn）
                if pred.shape[1] > 6 + embed_dim and len(predn) > 0:
                    # 从pred中提取状态预测
                    state_probs_raw = pred[:, 6 + embed_dim:6 + embed_dim + state_classes]  # (N_pred, state_classes)
                    
                    # 对状态概率进行softmax
                    state_probs_raw = torch.softmax(state_probs_raw, dim=1)
                    
                    # 找到predn中每个预测框对应的pred中的索引（通过bbox IoU匹配）
                    if len(pred) > 0:
                        # pred的bbox（NMS前）
                        pred_bboxes = pred[:, :4]  # (N_pred, 4)
                        # predn的bbox（NMS后）
                        predn_bboxes = predn[:, :4]  # (N_predn, 4)
                        
                        # 计算IoU矩阵
                        iou_matrix = box_iou(predn_bboxes, pred_bboxes)  # (N_predn, N_pred)
                        
                        # 为每个predn找到最匹配的pred索引
                        matched_indices = iou_matrix.argmax(dim=1)  # (N_predn,)
                        
                        # 从匹配的pred中提取状态预测
                        state_probs = state_probs_raw[matched_indices]  # (N_predn, state_classes)
                        
                        # 1) 使用状态预测的最大概率作为置信度
                        state_conf_raw, state_cls = state_probs.max(1, keepdim=True)  # (N_predn, 1), (N_predn, 1)
                        #state_conf = state_conf_raw.squeeze(1)
                        # 2) 使用检测框置信度作为置信度（因为检测框已经通过了NMS和置信度阈值）
                        # 这样可以确保状态检测指标不会因为状态置信度过低而被过滤
                        det_conf = predn[:, 4]  # 检测框的原始置信度
                        state_conf = det_conf
                        
                        
                        # 2) 构造"状态检测"的预测矩阵：使用predn的bbox，替换conf和cls为状态相关的
                        pred_state_det = predn.clone()  # 复制predn（包含bbox、conf、cls）
                        pred_state_det[:, 4] = state_conf  #.squeeze(1) 替换置信度为状态置信度
                        pred_state_det[:, 5] = state_cls.squeeze(1).float()  # 替换类别为状态类别
                    else:
                        # pred为空，但predn不为空，使用predn的原始conf和cls（虽然不准确，但保证代码不报错）
                        pred_state_det = predn.clone()
                    
                    # 3) 用"状态id"作为GT类别（tags是0-based，0-5对应6个状态）
                    if tags is not None and tags.numel() > 0:
                        # 确保tags是0-based
                        gt_state_cls = tags.to(dtype=torch.long, device=pred.device).view(-1)  # (M,)
                        # 检查范围：应该在0到state_classes-1之间
                        gt_state_cls = gt_state_cls.clamp_(min=0, max=state_classes-1)
                    else:
                        gt_state_cls = torch.zeros(0, dtype=torch.long, device=pred.device)
                    
                    # 4) 构造占位的gt_tags（match_predictions需要，但这里不关心内容）
                    gt_dummy_tags = torch.zeros_like(gt_state_cls, dtype=torch.long, device=pred.device)
                    
                    # 5) 计算状态检测的TP矩阵（复用_process_batch，使用相同的IoU阈值）
                    if pred_state_det.numel() > 0 and len(gt_state_cls) > 0:
                        # 临时保存原始nc
                        original_nc = self.nc
                        # 临时设置nc为状态类别数
                        self.nc = state_classes
                        
                        state_stat = dict(
                            conf=torch.zeros(0, device=self.device),
                            pred_cls=torch.zeros(0, device=self.device),
                            tp=torch.zeros(len(pred_state_det), self.niou, dtype=torch.bool, device=self.device),
                        )
                        state_stat["target_cls"] = gt_state_cls
                        state_stat["target_img"] = gt_state_cls.unique()
                        
                        # 使用_process_batch计算状态检测的TP（使用相同的IoU阈值）
                        state_stat["tp"], _ = self._process_batch(pred_state_det, bbox, gt_state_cls, gt_dummy_tags)
                        # 使用专门的状态检测匹配函数：先匹配框（基于IoU），再判断状态类别是否正确
                        # iou = box_iou(bbox, pred_state_det[:, :4])  # (M, N)，M是GT数量，N是预测数量
                        # state_stat["tp"] = self.match_predictions_for_state_detection(
                        #     pred_state_det[:, 5].long(),  # 预测的状态类别
                        #     gt_state_cls,  # 真实的状态类别
                        #     iou  # IoU矩阵
                        # )
                        state_stat["conf"] = pred_state_det[:, 4]  # 状态置信度
                        state_stat["pred_cls"] = pred_state_det[:, 5].long()  # 状态类别
                        
                        # 恢复原始nc
                        self.nc = original_nc
                        
                        # 保存状态检测统计数据
                        for k in self.state_det_stats.keys():
                            self.state_det_stats[k].append(state_stat[k])
                    elif len(gt_state_cls) > 0:
                        # 只有GT没有预测的情况
                        state_stat = dict(
                            conf=torch.zeros(0, device=self.device),
                            pred_cls=torch.zeros(0, device=self.device),
                            tp=torch.zeros(0, self.niou, dtype=torch.bool, device=self.device),
                        )
                        state_stat["target_cls"] = gt_state_cls
                        state_stat["target_img"] = gt_state_cls.unique()
                        for k in self.state_det_stats.keys():
                            self.state_det_stats[k].append(state_stat[k])

            # 收集当前图像的状态预测和目标（基于GT统计）
            if npr > 0 and nl > 0: #￥#人员状态预测评估指标
                # 传入TP信息，确保只统计TP
                tp_mask = stat["tp"][:, 0] if stat["tp"].numel() > 0 else torch.zeros(npr, dtype=torch.bool, device=self.device)  # 使用IoU=0.5的TP
                self._collect_state_data_for_image(si, pred, predn, bbox, tags, matched_tags, tp_mask) #￥#人员状态预测评估指标

            # Save
            if self.args.save_json:
                self.pred_to_json(predn, batch["im_file"][si])
            if self.args.save_txt:
                # 获取图像文件的完整路径
                im_file_path = Path(batch["im_file"][si])
                
                # 尝试从self.data获取数据集根目录
                dataset_root = None
                if hasattr(self, 'data') and self.data is not None:
                    dataset_root = self.data.get("path")
                    if dataset_root:
                        dataset_root = Path(dataset_root)
                
                # 计算相对路径并构建保存路径
                if dataset_root and dataset_root.exists():
                    try:
                        # 获取相对于数据集根目录的相对路径
                        relative_path = im_file_path.relative_to(dataset_root)
                        # 移除文件名，保留目录结构
                        relative_dir = relative_path.parent
                        
                        # 如果相对路径的第一部分是images，则跳过它
                        relative_parts = list(relative_dir.parts)
                        if relative_parts and relative_parts[0] == 'images':
                            relative_parts = relative_parts[1:]  # 跳过images目录
                            if relative_parts:
                                relative_dir = Path(*relative_parts)
                            else:
                                relative_dir = Path()  # 空路径
                        
                        # 构建保存路径：save_dir/labels/相对目录/文件名.txt
                        if relative_dir.parts:
                            save_path = self.save_dir / "labels" / relative_dir / f'{im_file_path.stem}.txt'
                        else:
                            save_path = self.save_dir / "labels" / f'{im_file_path.stem}.txt'
                    except (ValueError, AttributeError):
                        # 如果无法计算相对路径，使用路径哈希区分
                        path_str = str(im_file_path.parent).replace(os.sep, '_').replace('/', '_')
                        # 只保留最后几级路径避免文件名过长
                        path_parts = path_str.split('_')
                        path_suffix = '_'.join(path_parts[-3:]) if len(path_parts) > 3 else path_str[-50:]
                        save_path = self.save_dir / "labels" / f'{im_file_path.stem}_{path_suffix}.txt'
                else:
                    # 如果找不到数据集根目录，从图像路径中提取子目录结构
                    # 查找images、train、val、test等关键目录
                    path_parts = list(im_file_path.parts)
                    subdir_parts = []
                    start_idx = None
                    
                    # 找到images、train、val、test等目录的索引
                    for i, part in enumerate(path_parts):
                        if part in ['images', 'train', 'val', 'test']:
                            start_idx = i + 1  # 从该目录之后开始（跳过images等目录）
                            break
                    
                    if start_idx is not None and start_idx < len(path_parts) - 1:
                        # 提取子目录部分（不包括文件名）
                        subdir_parts = path_parts[start_idx:-1]
                        if subdir_parts:
                            # 构建保存路径：save_dir/labels/子目录/文件名.txt
                            save_path = self.save_dir / "labels" / Path(*subdir_parts) / f'{im_file_path.stem}.txt'
                        else:
                            save_path = self.save_dir / "labels" / f'{im_file_path.stem}.txt'
                    else:
                        # 如果找不到关键目录，使用父目录名+文件名（但跳过images）
                        parent_name = im_file_path.parent.name
                        if parent_name and parent_name not in ['images', 'train', 'val', 'test']:
                            save_path = self.save_dir / "labels" / parent_name / f'{im_file_path.stem}.txt'
                        else:
                            save_path = self.save_dir / "labels" / f'{im_file_path.stem}.txt'
                
                self.save_one_txt(
                    predn,
                    self.args.save_conf,
                    pbatch["ori_shape"],
                    save_path,
                    pred=pred,  # 传入原始pred以提取状态信息   
                )
        
        # 在调用reid_metrics之前检查batch_matched_tags是否为空 #%￥￥%￥#
        if batch_matched_tags and any(len(tags) > 0 for tags in batch_matched_tags): #%￥￥%￥#
            # Process batch for reid metrics
            self.reid_metrics.process_batch(preds, batch_matched_tags) #往后处理%￥￥%￥#
        
        self._process_batch_state_metrics() #￥#人员状态预测评估指标# 处理当前批次的状态预测评估

    def _collect_state_data_for_image(self, image_idx, pred, predn, gt_bboxes, gt_tags, matched_tags, tp_mask):
        """
        收集单张图像的状态预测数据（基于所有GT统计）
        注意：状态索引直接从类别索引获取，因为类别索引0-5对应状态索引0-5
        """
        if not hasattr(self.model.model[-1], 'state_classes') or self.model.model[-1].state_classes is None:
            return
            
        embed_dim = self.model.model[-1].embed_dim
        state_classes = self.model.model[-1].state_classes
        
        # 从预测结果中提取状态预测
        if pred.shape[1] > 6 + embed_dim:
            state_preds = pred[:, 6+embed_dim:6+embed_dim+state_classes]  # (N_detections, state_classes)
            
            # 安全地检查张量是否为空（处理0维张量的情况）
            if gt_tags.numel() == 0 or len(predn) == 0:
                return
            
            # 确保gt_tags和gt_bboxes是正确的维度
            if gt_tags.dim() == 0:
                gt_tags = gt_tags.unsqueeze(0)
            if gt_bboxes.dim() == 1:
                gt_bboxes = gt_bboxes.unsqueeze(0)
            
            
            # 确保tp_mask是正确的维度
            if tp_mask.dim() == 0:
                tp_mask = tp_mask.unsqueeze(0) if tp_mask.numel() > 0 else torch.zeros(len(predn), dtype=torch.bool, device=pred.device)
            if len(tp_mask) != len(predn):
                tp_mask = torch.zeros(len(predn), dtype=torch.bool, device=pred.device)
            
            # 计算IoU矩阵，用于找到每个GT匹配的最佳预测框
            from ultralytics.utils.metrics import box_iou
            iou_matrix = box_iou(gt_bboxes, predn[:, :4])  # (M, N)
            
            # 直接统计所有GT实例，状态索引直接从类别索引获取
            pred_states_list = []
            target_states_list = []
            
            for gt_idx, gt_tag in enumerate(gt_tags):
                gt_tag_value = gt_tag.item() if gt_tag.dim() == 0 else int(gt_tag)
                
                # 关键修复：假设gt_tags是0-based（0-5），直接使用作为状态索引
                # 如果数据集中gt_tags实际上是1-based（1-6），那么需要减1
                # 根据person_states字典的键是0-based，我们假设gt_tags也是0-based
                gt_state_0based = int(gt_tag_value)
                
                # 检查范围：应该在0到5之间（0-based，对应6个状态）
                if gt_state_0based < 0 or gt_state_0based >= 6:
                    continue
                
                # 找到匹配到该GT的预测框（如果有）- 匹配时使用原始的gt_tag_value
                matched_pred_indices = (matched_tags == gt_tag_value).nonzero(as_tuple=True)[0]
                
                if len(matched_pred_indices) > 0:
                    # 使用匹配预测的状态
                    ious = iou_matrix[gt_idx, matched_pred_indices]
                    best_pred_idx = matched_pred_indices[ious.argmax()]
                    pred_states_list.append(state_preds[best_pred_idx:best_pred_idx+1])
                else:
                    # FN情况：使用GT状态作为"预测"（用于统计一致性）
                    # 使用0-based索引
                    dummy_pred = torch.zeros(1, state_classes, device=pred.device)
                    dummy_pred[0, gt_state_0based] = 1.0
                    pred_states_list.append(dummy_pred)
                
                # 使用0-based索引用于StateMetrics
                target_states_list.append(torch.tensor([gt_state_0based], device=pred.device, dtype=torch.long))

            # 合并所有GT实例的状态预测
            if pred_states_list and target_states_list:
                batch_pred_states = torch.cat(pred_states_list, dim=0)
                batch_target_states = torch.cat(target_states_list, dim=0)
                
                self.current_batch_pred_states.append(batch_pred_states)
                self.current_batch_target_states.append(batch_target_states)

    def _process_batch_state_metrics(self): #￥#人员状态预测评估指标
        """处理当前批次的状态预测指标"""
        if not self.current_batch_pred_states or not self.current_batch_target_states: #￥#人员状态预测评估指标
            return #￥#人员状态预测评估指标
            
        if self.state_metrics is None: #￥#人员状态预测评估指标
            return #￥#人员状态预测评估指标
            
        # 合并当前批次的所有状态预测数据
        try: #￥#人员状态预测评估指标
            batch_pred_states = torch.cat(self.current_batch_pred_states, dim=0) #￥#人员状态预测评估指标
            batch_target_states = torch.cat(self.current_batch_target_states, dim=0) #￥#人员状态预测评估指标
            
            # 构建图像索引（每个样本对应的图像索引）
            image_indices = []
            for img_idx, states in enumerate(self.current_batch_target_states):
                image_indices.extend([img_idx + self.seen - len(self.current_batch_target_states)] * len(states))
            image_indices = np.array(image_indices)
            
            # 确保维度匹配
            if len(batch_pred_states) == len(batch_target_states): #￥#人员状态预测评估指标
                self.state_metrics.process(batch_pred_states, batch_target_states, image_indices) #￥#人员状态预测评估指标
                #print(f"✅ 成功处理 {len(batch_pred_states)} 个状态预测样本")
                #print(f"📈 当前状态准确率: {self.state_metrics.state_accuracy:.4f}")
            else:
                print(f"警告：状态预测维度不匹配 - pred: {len(batch_pred_states)}, target: {len(batch_target_states)}") #￥#人员状态预测评估指标
                 
        except Exception as e: #￥#人员状态预测评估指标
            print(f"状态指标处理错误: {e}") #￥#人员状态预测评估指标

    def get_stats(self):
        """Returns metrics statistics and results dictionary."""
        stats = {k: torch.cat(v, 0).cpu().numpy() for k, v in self.stats.items()}  # to numpy
        self.nt_per_class = np.bincount(stats["target_cls"].astype(int), minlength=self.nc)
        self.nt_per_image = np.bincount(stats["target_img"].astype(int), minlength=self.nc)
        stats.pop("target_img", None)
        

        # 定义完整的ReID metrics结构（无论是否有检测都返回）
        reid_metrics = {
            "val/pos_cos": 0.0,
            "val/neg_cos": 1.0,
            "val/pos_euc": 0.0,
            "val/neg_euc": 1.0,
            "val/cos_sep_ratio": 1.0,
            "val/euc_sep_ratio": 1.0,
            "val/cos_silhouette": 0.0,
            "val/euc_silhouette": 0.0,
            "val/davies_bouldin": 0.0,
            "val/calinski_harabasz": 0.0,
            "val/r1_acc": 0.0,
            "val/r5_acc": 0.0,
            "val/mean_ap": 0.0,
            "val/hota": 0.0,
            "val/mota": 0.0,
            "val/idf1": 0.0,
        }
        
        if len(stats) and stats["tp"].any():
            self.metrics.process(**stats)
            # 只有在有正确检测时才更新reid_metrics
            computed_reid_metrics = self.reid_metrics.get_metrics()
            reid_metrics.update(computed_reid_metrics)  # 用实际计算值覆盖默认值
        else:
            # 当没有正确检测时，使用默认的0值占位
            print("⚠️ 当前批次没有正确检测，跳过ReID metrics计算")
            
        detector_results = self.metrics.results_dict
        detector_results.update(reid_metrics)
        
        # ========== 修复：确保状态预测指标始终返回11个键 ==========
        # 定义完整的状态预测指标结构（与StateMetrics.results_dict返回的键完全一致）
        default_state_metrics = {
            "metrics/state_accuracy": 0.0,
            "metrics/state_macro_accuracy": 0.0,
            "metrics/state_macro_precision": 0.0,
            "metrics/state_macro_recall": 0.0,
            "metrics/state_macro_f1": 0.0,
            "metrics/state_micro_precision": 0.0,
            "metrics/state_micro_recall": 0.0,
            "metrics/state_micro_f1": 0.0,
            "metrics/state_total_tp": 0,
            "metrics/state_total_fp": 0,
            "metrics/state_total_fn": 0,
        }
        
        if self.state_metrics is not None:
            state_results = self.state_metrics.results_dict
            # 用实际计算值更新默认值（确保键完全匹配）
            default_state_metrics.update(state_results)
        
        # 无论是否有state_metrics，都使用相同的键结构
        detector_results.update(default_state_metrics)

        # ========== 修复：确保状态检测指标始终返回6个键 ==========
        # 定义完整的状态检测指标结构（与DetMetrics.results_dict返回的键完全一致）
        default_state_det_metrics = {
            "state_det/metrics/precision(B)": 0.0,
            "state_det/metrics/recall(B)": 0.0,
            "state_det/metrics/mAP50(B)": 0.0,
            "state_det/metrics/mAP75(B)": 0.0,
            "state_det/metrics/mAP50-95(B)": 0.0,
            "state_det/fitness": 0.0,
        }
        
        if hasattr(self, "state_det_stats") and len(self.state_det_stats) > 0:
            state_det_stats = {k: torch.cat(v, 0).cpu().numpy() if v else np.array([]) 
                          for k, v in self.state_det_stats.items()}
            
            if len(state_det_stats) > 0 and state_det_stats["tp"].size > 0 and state_det_stats["tp"].any():
                # 临时保存原始nc和nt_per_class
                original_nc = self.nc
                original_nt_per_class = self.nt_per_class
                
                # 设置状态类别数
                state_classes = getattr(self.model.model[-1], "state_classes", 6) if hasattr(self, "model") and self.model is not None else 6
                self.nc = state_classes
                
                # 计算状态检测的每类统计信息（在pop target_img之前）
                self.state_nt_per_class = np.bincount(
                    state_det_stats["target_cls"].astype(int), 
                    minlength=state_classes
                )
                # 计算每类出现的图像数
                if "target_img" in state_det_stats and state_det_stats["target_img"].size > 0:
                    self.state_nt_per_image = np.bincount(
                        state_det_stats["target_img"].astype(int), 
                        minlength=state_classes
                    )
                else:
                    self.state_nt_per_image = np.zeros(state_classes, dtype=np.int64)
                
                state_det_stats.pop("target_img", None)
                
                # 使用DetMetrics计算状态检测指标
                self.state_det_metrics.process(**state_det_stats)
                state_det_results = self.state_det_metrics.results_dict
                
                # 恢复原始nc
                self.nc = original_nc
                self.nt_per_class = original_nt_per_class
                
                # 重命名状态检测指标（添加前缀）
                state_det_results_renamed = {
                    f"state_det/{k}": v for k, v in state_det_results.items()
                }
                # 用实际计算值更新默认值（确保键完全匹配）
                default_state_det_metrics.update(state_det_results_renamed)
        
            # 初始化空的统计信息（如果需要）
            if not hasattr(self, 'state_nt_per_class'):
                state_classes = getattr(self.model.model[-1], "state_classes", 6) if hasattr(self, "model") and self.model is not None else 6
                self.state_nt_per_class = np.zeros(state_classes, dtype=np.int64)
                self.state_nt_per_image = np.zeros(state_classes, dtype=np.int64)
        
        # 无论是否有state_det_stats，都使用相同的键结构
        detector_results.update(default_state_det_metrics)
        
        return detector_results

    def preprocess(self, batch):
        """Preprocesses batch by converting masks to float and sending to device."""
        batch = super().preprocess(batch)
        batch["tags"] = batch["tags"].to(self.device).float()
        return batch

    def postprocess(self, preds):
        """Apply Non-maximum suppression to prediction outputs."""
        
        
        # 应用NMS（只对检测部分）
        preds = ops.non_max_suppression(
            preds[0],
            self.args.conf,
            self.args.iou,
            labels=self.lb,
            multi_label=True,
            agnostic=self.args.single_cls or self.args.agnostic_nms,
            max_det=self.args.max_det,
            nc=self.nc,
        )
        return preds

    def _process_batch(self, detections, gt_bboxes, gt_cls, gt_tags):
        """
        Return correct prediction matrix.

        Args:
            detections (torch.Tensor): Tensor of shape (N, 6) representing detections where each detection is
                (x1, y1, x2, y2, conf, class).
            gt_bboxes (torch.Tensor): Tensor of shape (M, 4) representing ground-truth bounding box coordinates. Each
                bounding box is of the format: (x1, y1, x2, y2).
            gt_cls (torch.Tensor): Tensor of shape (M,) representing target class indices.
            gt_tags (torch.Tensor): Tensor of shape (M,) representing target tags.

        Returns:
            (torch.Tensor): Correct prediction matrix of shape (N, 10) for 10 IoU levels.

        Note:
            The function does not return any value directly usable for metrics calculation. Instead, it provides an
            intermediate representation used for evaluating predictions against ground truth.
        """
        iou = box_iou(gt_bboxes, detections[:, :4])
        return self.match_predictions(detections[:, 5], gt_cls, gt_tags, iou)

    def match_predictions(self, pred_classes, true_classes, true_tags, iou, use_scipy=False):
        """
        Matches predictions to ground truth objects (pred_classes, true_classes) using IoU.

        Args:
            pred_classes (torch.Tensor): Predicted class indices of shape(N,).
            true_classes (torch.Tensor): Target class indices of shape(M,).
            true_tags (torch.Tensor): Target tags of shape(M,).
            iou (torch.Tensor): An NxM tensor containing the pairwise IoU values for predictions and ground of truth
            use_scipy (bool): Whether to use scipy for matching (more precise).

        Returns:
            (torch.Tensor): Correct tensor of shape(N,10) for 10 IoU thresholds.
        """
        # Initialize the list for storing matched tags using IoU threshold of 0.5
        matched_tags = [False] * pred_classes.shape[0]  # Default to None if no match

        # Dx10 matrix, where D - detections, 10 - IoU thresholds
        correct = np.zeros((pred_classes.shape[0], self.iouv.shape[0])).astype(bool)
        # LxD matrix where L - labels (rows), D - detections (columns)
        correct_class = true_classes[:, None] == pred_classes
        iou = iou * correct_class  # zero out the wrong classes
        iou = iou.cpu().numpy()
        for i, threshold in enumerate(self.iouv.cpu().tolist()):
            if use_scipy:
                # WARNING: known issue that reduces mAP in https://github.com/ultralytics/ultralytics/pull/4708
                import scipy  # scope import to avoid importing for all commands

                cost_matrix = iou * (iou >= threshold)
                if cost_matrix.any():
                    labels_idx, detections_idx = scipy.optimize.linear_sum_assignment(cost_matrix, maximize=True)
                    valid = cost_matrix[labels_idx, detections_idx] > 0
                    if valid.any():
                        correct[detections_idx[valid], i] = True
                        # Assign tags to matched predictions
                        if threshold == self.state_iou:  #￥#￥#￥#￥#￥#￥#
                            for gt_idx, pred_idx in zip(labels_idx[valid], detections_idx[valid]):
                                matched_tags[pred_idx] = true_tags[gt_idx].item()
            else:
                matches = np.nonzero(iou >= threshold)  # IoU > threshold and classes match
                matches = np.array(matches).T
                if matches.shape[0]:
                    if matches.shape[0] > 1:
                        matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                        matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                        # matches = matches[matches[:, 2].argsort()[::-1]]
                        matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                    correct[matches[:, 1].astype(int), i] = True
                    # Assign tags to matched predictions
                    if threshold == self.state_iou:  #￥#￥#￥#￥#￥#￥#
                        for gt_idx, pred_idx in matches:
                            if true_tags.dim() > 0: #$#$#
                                matched_tags[pred_idx] = true_tags[gt_idx].item()
        return torch.tensor(correct, dtype=torch.bool, device=pred_classes.device), torch.tensor(matched_tags, dtype=torch.int, device=pred_classes.device)

    def match_predictions_for_state_detection(self, pred_classes, true_classes, iou, use_scipy=False):
        """
        专门用于状态检测的匹配函数：先匹配框（基于IoU），再判断状态类别是否正确
        
        与match_predictions的区别：
        - match_predictions: 先检查类别匹配，再匹配框（用于常规检测）
        - match_predictions_for_state_detection: 先匹配框，再检查状态类别（用于状态检测）
        
        Args:
            pred_classes (torch.Tensor): 预测的状态类别 (N,)
            true_classes (torch.Tensor): 真实的状态类别 (M,)
            iou (torch.Tensor): IoU矩阵 (M, N)，M是GT数量，N是预测数量
            use_scipy (bool): 是否使用scipy进行匹配（更精确）
        
        Returns:
            (torch.Tensor): TP矩阵 (N, 10) 对于10个IoU阈值，dtype=bool
        """
        # Dx10 matrix, where D - detections, 10 - IoU thresholds
        correct = np.zeros((pred_classes.shape[0], self.iouv.shape[0])).astype(bool)
        iou_np = iou.cpu().numpy()  # 转换为numpy用于计算
        
        for i, threshold in enumerate(self.iouv.cpu().tolist()):
            if use_scipy:
                # WARNING: known issue that reduces mAP in https://github.com/ultralytics/ultralytics/pull/4708
                import scipy.optimize  # scope import to avoid importing for all commands
                
                # 先基于IoU匹配框（不考虑类别）
                cost_matrix = iou_np * (iou_np >= threshold)
                if cost_matrix.any():
                    labels_idx, detections_idx = scipy.optimize.linear_sum_assignment(
                        cost_matrix, maximize=True
                    )
                    valid = cost_matrix[labels_idx, detections_idx] > 0
                    if valid.any():
                        # 匹配到框后，再检查状态类别是否匹配
                        for gt_idx, pred_idx in zip(labels_idx[valid], detections_idx[valid]):
                            if true_classes[gt_idx].item() == pred_classes[pred_idx].item():
                                correct[pred_idx, i] = True
            else:
                # 先找到所有IoU >= threshold的匹配（不考虑类别）
                matches = np.nonzero(iou_np >= threshold)  # IoU >= threshold
                matches = np.array(matches).T  # (num_matches, 2)，每行是[gt_idx, pred_idx]
                if matches.shape[0]:
                    if matches.shape[0] > 1:
                        # 按IoU降序排序
                        matches = matches[iou_np[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                        # 每个预测框只保留IoU最高的GT（去重预测框）
                        matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                        # 每个GT只保留IoU最高的预测框（去重GT）
                        matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                    
                    # 检查匹配的框的状态类别是否也匹配
                    for gt_idx, pred_idx in matches:
                        if true_classes[gt_idx].item() == pred_classes[pred_idx].item():
                            correct[pred_idx, i] = True
        
        return torch.tensor(correct, dtype=torch.bool, device=pred_classes.device)

    def print_results(self): #￥#添加人员状态预测评估指标的打印
        """Prints training/validation set metrics per class and state metrics."""
        # 调用父类方法显示检测指标
        super().print_results() #￥#添加人员状态预测评估指标的打印
        
        # 显示状态预测指标
        if self.state_metrics is not None and self.state_metrics.total > 0: #￥#添加人员状态预测评估指标的打印
            # 更新格式化指标
            self.state_metrics.update_formatted_metrics()
            
            # 计算每个状态出现在多少张图像中
            state_image_counts = np.array([len(s) for s in self.state_metrics.state_images_set])
            
            # 格式化输出，类似检测指标
            pf = "%22s" + "%11i" * 2 + "%11.3g" * 5  # print format: Class + Images + Instances + 5个指标
            pf2 = "%22s" + "%11s" * 2 + "%11.3s" * 5
            
            # 打印每个状态类别的详细统计
            if self.args.verbose and not self.training:
                # 尝试获取状态名称映射
                state_names = {}
                num_defined_states = 6  # 默认6个状态
                if hasattr(self, 'model') and hasattr(self.model, 'person_states') and isinstance(self.model.person_states, dict):
                    for k, v in self.model.person_states.items():
                        if isinstance(k, int):
                            state_names[k] = str(v)
                            num_defined_states = max(num_defined_states, k + 1)
                        elif isinstance(v, int):
                            state_names[v] = str(k)
                            num_defined_states = max(num_defined_states, v + 1)
                
                # 按照状态索引顺序（0到num_defined_states-1）打印所有状态
                for state_idx in range(num_defined_states):
                    state_name = state_names.get(state_idx, f"State_{state_idx}")
                    images_count = int(state_image_counts[state_idx]) if state_idx < len(state_image_counts) else 0
                    instances_count = int(self.state_metrics.state_total_counts[state_idx]) if state_idx < len(self.state_metrics.state_total_counts) else 0
                    
                    # 使用state_result方法根据状态索引获取指标
                    metrics = self.state_metrics.state_result(state_idx)
                    
                    LOGGER.info(
                        pf % (state_name, images_count, instances_count, *metrics)
                    )
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%            
            # ========== 以下为详细统计信息输出，可以注释掉不打印 ==========
            # 获取 TP、FP、FN
            tp, fp, fn = self.state_metrics.get_tp_fp_fn()
            precision, recall, f1 = self.state_metrics.get_precision_recall_f1()
            
            # 总体统计
            total_tp = int(tp.sum())
            total_fp = int(fp.sum())
            total_fn = int(fn.sum())
            
            print(
                f"✅State Prediction Results (IoU={self.state_iou}): 🔄Total samples: {self.state_metrics.total}, "
                f"📈Accuracy: {self.state_metrics.state_accuracy:.4f}, "
                f"📊Macro Accuracy: {self.state_metrics.per_state_accuracy.mean():.4f}"
            )
                       
        else:    #￥#添加人员状态预测评估指标的打印
            LOGGER.info("No state prediction data available for evaluation") #￥#添加人员状态预测评估指标的打印

        # ========== 在方法末尾添加保存Excel的逻辑 ==========
        # 检查是否启用Excel保存功能（使用实例变量）
        save_excel = getattr(self, '_save_excel', False)
        if save_excel and not self.training:
            LOGGER.info(f"✅ 开始保存评估结果到Excel (save_excel={save_excel})")
            self._save_results_to_excel()

    def save_one_txt(self, predn, save_conf, shape, file, pred=None):
        """
        Save YOLO detections to a txt file in normalized coordinates.
        If save_tag_to_txt is True, adds state information as the last column.
        
        Args:
            predn: NMS后的预测结果，格式为 [x1, y1, x2, y2, conf, cls]
            save_conf: 是否保存置信度
            shape: 图像形状
            file: 输出文件路径
            pred: NMS前的原始预测结果（可选，用于提取状态信息）
        """
        from ultralytics.engine.results import Results
        from ultralytics.utils.metrics import box_iou
        
        # 检查是否需要保存状态信息
        save_tag_to_txt = getattr(self, '_save_tag_to_txt', False)
        
        # 提取状态信息（如果需要保存且pred提供且模型支持状态预测）
        state_cls_list = None
        if save_tag_to_txt and pred is not None:
            if hasattr(self.model.model[-1], "state_classes") and self.model.model[-1].state_classes is not None:
                embed_dim = self.model.model[-1].embed_dim
                state_classes = self.model.model[-1].state_classes
                
                # 检查pred是否包含状态信息
                if pred.shape[1] > 6 + embed_dim and len(predn) > 0 and len(pred) > 0:
                    # 从pred中提取状态预测
                    state_probs_raw = pred[:, 6 + embed_dim:6 + embed_dim + state_classes]  # (N_pred, state_classes)
                    
                    # 对状态概率进行softmax
                    state_probs_raw = torch.softmax(state_probs_raw, dim=1)
                    
                    # 找到predn中每个预测框对应的pred中的索引（通过bbox IoU匹配）
                    pred_bboxes = pred[:, :4]  # (N_pred, 4)
                    predn_bboxes = predn[:, :4]  # (N_predn, 4)
                    
                    # 计算IoU矩阵
                    iou_matrix = box_iou(predn_bboxes, pred_bboxes)  # (N_predn, N_pred)
                    
                    # 为每个predn找到最匹配的pred索引
                    matched_indices = iou_matrix.argmax(dim=1)  # (N_predn,)
                    
                    # 从匹配的pred中提取状态预测
                    state_probs = state_probs_raw[matched_indices]  # (N_predn, state_classes)
                    
                    # 获取状态类别（最大概率对应的类别）
                    _, state_cls = state_probs.max(1)  # (N_predn,)
                    state_cls_list = state_cls.cpu().numpy().tolist()
        
        # 如果提取到了状态信息，手动写入文件
        if state_cls_list is not None:
            Path(file).parent.mkdir(parents=True, exist_ok=True)
            with open(file, "a") as f:
                # 创建Results对象用于获取归一化坐标
                results = Results(
                    np.zeros((shape[0], shape[1]), dtype=np.uint8),
                    path=None,
                    names=self.names,
                    boxes=predn[:, :6],
                )
                
                for i, box in enumerate(results.boxes):
                    c = int(box.cls)
                    conf = float(box.conf)
                    # 转换为归一化的xywh格式
                    xywhn = box.xywhn[0].cpu().numpy()
                    
                    # 构建输出行：class x_center y_center width height [conf] state
                    line = [c, xywhn[0], xywhn[1], xywhn[2], xywhn[3]]
                    if save_conf:
                        line.append(conf)
                    # 添加状态类别 - 确保每个框都使用正确的状态类别
                    state_cls_val = state_cls_list[i] if i < len(state_cls_list) else 0
                    line.append(state_cls_val)
                    
                    # 写入文件
                    f.write(("%g " * len(line)).rstrip() % tuple(line) + "\n")
        else:
            # 如果没有状态信息，使用父类的默认保存方式
            super().save_one_txt(predn, save_conf, shape, file)

    def _save_results_to_excel(self):
        """保存评估结果到Excel文件"""
        # 先尝试导入必要的库
        try:
            import pandas as pd
        except ImportError as e:
            LOGGER.warning(f"⚠️ 无法导入pandas: {e}")
            LOGGER.warning(f"⚠️ 请安装pandas: pip install pandas")
            return
        
        try:
            import openpyxl
        except ImportError as e:
            LOGGER.warning(f"⚠️ 无法导入openpyxl: {e}")
            LOGGER.warning(f"⚠️ 请安装openpyxl: pip install openpyxl")
            return
        
        try:
            from pathlib import Path
            
            # 获取模型名称（使用实例变量）
            model_name = getattr(self, '_model_name', 'unknown')
            
            # 获取Excel保存目录（使用实例变量）
            excel_save_dir = getattr(self, '_excel_save_dir', None)
            excel_name = getattr(self, '_excel_name', 'result_all.xlsx')
            if excel_save_dir:
                excel_path = Path(excel_save_dir) / excel_name
            else:
                # 如果没有指定，使用项目目录的父目录
                excel_path = Path(self.save_dir).parent / excel_name
            
            LOGGER.info(f"📝 Excel文件路径: {excel_path}")
            
            # 准备数据字典
            row_data = {}
            
            # 第1列：模型名称
            row_data['Model'] = model_name
            
            # 第2-8列：检测指标 (Images, Instances, P, R, mAP50, mAP75, mAP50-95)
            det_results = self.metrics.mean_results()  # [P, R, mAP50, mAP75, mAP50-95]
            row_data['Images'] = self.seen
            row_data['Instances'] = int(self.nt_per_class.sum())
            row_data['Box_P'] = round(float(det_results[0]), 3) if len(det_results) > 0 else 0.0
            row_data['Box_R'] = round(float(det_results[1]), 3) if len(det_results) > 1 else 0.0
            row_data['Box_mAP50'] = round(float(det_results[2]), 3) if len(det_results) > 2 else 0.0
            row_data['Box_mAP75'] = round(float(det_results[3]), 3) if len(det_results) > 3 else 0.0
            row_data['Box_mAP50-95'] = round(float(det_results[4]), 3) if len(det_results) > 4 else 0.0
            
            # 第9-10列：State Prediction Results (Accuracy, Macro Accuracy)
            if self.state_metrics is not None and self.state_metrics.total > 0:
                row_data['State_Accuracy'] = round(float(self.state_metrics.state_accuracy), 4)
                support = self.state_metrics.state_total_counts > 0
                row_data['State_Macro_Accuracy'] = round(float(self.state_metrics.per_state_accuracy[support].mean()), 4) if support.any() else 0.0
            else:
                row_data['State_Accuracy'] = 0.0
                row_data['State_Macro_Accuracy'] = 0.0
            
            # 第11-13列：State Prediction Metrics (P, R, F1)
            if self.state_metrics is not None and self.state_metrics.total > 0:
                self.state_metrics.update_formatted_metrics()
                state_pred_results = self.state_metrics.mean_results()  # [mp, mr, mf1, 0, 0]
                row_data['State_Pred_P'] = round(float(state_pred_results[0]), 3) if len(state_pred_results) > 0 else 0.0
                row_data['State_Pred_R'] = round(float(state_pred_results[1]), 3) if len(state_pred_results) > 1 else 0.0
                row_data['State_Pred_F1'] = round(float(state_pred_results[2]), 3) if len(state_pred_results) > 2 else 0.0
            else:
                row_data['State_Pred_P'] = 0.0
                row_data['State_Pred_R'] = 0.0
                row_data['State_Pred_F1'] = 0.0
            
            # 第14-18列：State Detection Metrics (pre, rec, mAP50, mAP75, mAP50-95)
            if hasattr(self, "state_det_metrics") and hasattr(self, "state_det_stats") and len(self.state_det_stats) > 0:
                state_det_stats = {k: torch.cat(v, 0).cpu().numpy() if v else np.array([]) 
                                      for k, v in self.state_det_stats.items()}
                if len(state_det_stats) > 0 and state_det_stats["tp"].size > 0 and state_det_stats["tp"].any():
                    state_det_stats.pop("target_img", None)
                    self.state_det_metrics.process(**state_det_stats)
                    state_det_results = self.state_det_metrics.mean_results()  # [P, R, mAP50, mAP75, mAP50-95]
                    row_data['State_Det_P'] = round(float(state_det_results[0]), 3) if len(state_det_results) > 0 else 0.0
                    row_data['State_Det_R'] = round(float(state_det_results[1]), 3) if len(state_det_results) > 1 else 0.0
                    row_data['State_Det_mAP50'] = round(float(state_det_results[2]), 3) if len(state_det_results) > 2 else 0.0
                    row_data['State_Det_mAP75'] = round(float(state_det_results[3]), 3) if len(state_det_results) > 3 else 0.0
                    row_data['State_Det_mAP50-95'] = round(float(state_det_results[4]), 3) if len(state_det_results) > 4 else 0.0
                else:
                    row_data['State_Det_P'] = 0.0
                    row_data['State_Det_R'] = 0.0
                    row_data['State_Det_mAP50'] = 0.0
                    row_data['State_Det_mAP75'] = 0.0
                    row_data['State_Det_mAP50-95'] = 0.0
            else:
                row_data['State_Det_P'] = 0.0
                row_data['State_Det_R'] = 0.0
                row_data['State_Det_mAP50'] = 0.0
                row_data['State_Det_mAP75'] = 0.0
                row_data['State_Det_mAP50-95'] = 0.0
            
            # 读取或创建Excel文件
            if excel_path.exists():
                try:
                    # 读取现有Excel文件
                    df = pd.read_excel(excel_path, sheet_name=0, engine='openpyxl')
                    
                    # 检查是否已存在该模型名称的行
                    if 'Model' in df.columns:
                        model_idx = df[df['Model'] == model_name].index
                        if len(model_idx) > 0:
                            # 更新现有行
                            for col, val in row_data.items():
                                df.at[model_idx[0], col] = val
                            LOGGER.info(f"📝 更新Excel中模型 '{model_name}' 的数据")
                        else:
                            # 添加新行
                            df = pd.concat([df, pd.DataFrame([row_data])], ignore_index=True)
                            LOGGER.info(f"📝 在Excel中添加新模型 '{model_name}' 的数据")
                    else:
                        # 如果Model列不存在，添加新行
                        df = pd.concat([df, pd.DataFrame([row_data])], ignore_index=True)
                        LOGGER.info(f"📝 Excel文件缺少Model列，添加新行")
                except Exception as e:
                    LOGGER.warning(f"⚠️ 读取Excel文件时出错: {e}")
                    # 如果读取失败，创建新的DataFrame
                    df = pd.DataFrame([row_data])
                    LOGGER.info(f"📝 创建新的Excel文件")
            else:
                # 创建新的DataFrame
                df = pd.DataFrame([row_data])
                LOGGER.info(f"📝 创建新的Excel文件")
            
            # 保存到Excel
            df.to_excel(excel_path, index=False, engine='openpyxl')
            LOGGER.info(f"✅ 评估结果已成功保存到Excel: {excel_path}")
            
        except ImportError as e:
            LOGGER.warning(f"⚠️ 导入库时出错: {e}")
            import traceback
            traceback.print_exc()
        except Exception as e:
            LOGGER.warning(f"⚠️ 保存Excel时出错: {e}")
            import traceback
            traceback.print_exc()
