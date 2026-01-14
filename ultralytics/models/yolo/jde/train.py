# Ultralytics YOLO 🚀, AGPL-3.0 license

from copy import copy

from ultralytics.models import yolo
from ultralytics.nn.tasks import JDEModel
from ultralytics.utils import DEFAULT_CFG, RANK
from ultralytics.utils.plotting import plot_images, plot_results


class JDETrainer(yolo.detect.DetectionTrainer):
    """
    A class extending the DetectionTrainer class for training based on a joint detection and embedding model.

    Example:
        ```python
        from ultralytics.models.yolo.jde import JDETrainer

        args = dict(model="yolov8n-jde.pt", data="coco8-seg.yaml", epochs=3)
        trainer = JDETrainer(overrides=args)
        trainer.train()
        ```
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize a SegmentationTrainer object with given arguments."""
        if overrides is None:
            overrides = {}
        #$#overrides["task"] = "jde"
        super().__init__(cfg, overrides, _callbacks)
        #self.model.person_states = self.data.get("person_states", {})

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return SegmentationModel initialized with specified config and weights."""
        model = JDEModel(cfg, ch=3, nc=self.data["nc"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)

        return model

    def get_validator(self):
        """Return an instance of SegmentationValidator for validation of YOLO model."""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss", "emb_loss", "state_loss"  # 添加state_loss #￥#添加人员状态预测评估指标
        return yolo.jde.JDEValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )
    
    def label_loss_items(self, loss_items=None, prefix="train"): #￥#添加人员状态预测评估指标
        """返回带标签的损失项字典，包括状态损失"""
        keys = [f"{prefix}/{x}" for x in self.loss_names] #￥#添加人员状态预测评估指标
        if loss_items is not None: #￥#添加人员状态预测评估指标
            loss_items = [round(float(x), 5) for x in loss_items] #￥#添加人员状态预测评估指标
            return dict(zip(keys, loss_items)) #￥#添加人员状态预测评估指标
        else: #￥#添加人员状态预测评估指标
            return keys #￥#添加人员状态预测评估指标

    def plot_training_samples(self, batch, ni):
        """Plot training samples with annotations."""
        tags_or_cls = batch.get("tags", batch["cls"]).squeeze(-1) # 使用get获取tags，如果不存在就使用cls
        plot_images(
            images=batch["img"],
            batch_idx=batch["batch_idx"],
            cls=tags_or_cls,  # batch["tags"].squeeze(-1),
            bboxes=batch["bboxes"],
            paths=batch["im_file"],
            fname=self.save_dir / f"train_batch{ni}.jpg",
            on_plot=self.on_plot,
        )

    def set_model_attributes(self):
        """设置JDE模型属性，包括names和person_states"""
        super().set_model_attributes()  # 调用父类方法
        
        # 确保person_states被正确设置到模型
        if hasattr(self, 'data') and self.data and 'person_states' in self.data:
            person_states = self.data["person_states"]
            self.model.person_states = person_states
            #print(f"JDETrainer: 成功设置person_states到模型: {person_states}")
        else:
            self.model.person_states = {}
            print("JDETrainer: 未找到person_states数据，设置为空字典")
        
        # 如果是DDP包装的模型，也设置到module中
        if hasattr(self.model, 'module'):
            self.model.module.person_states = getattr(self.model, 'person_states', {})
            #print(f"JDETrainer: 也设置person_states到DDP模块: {getattr(self.model, 'person_states', {})}")
