from mmcv.runner import HOOKS
from mmcv.runner.hooks import Hook


@HOOKS.register_module()
class CorrectionScaleAnnealHook(Hook):
    """Hook to update the correction_scale in TopoLogicSGNNDecoder at each epoch.

    This enables cosine annealing of the correction scale from correction_scale_min
    (tight constraint, forces reliance on priors) to correction_scale_max
    (looser constraint, allows larger deviations) over the course of training.

    The tight constraint early in training forces the model to learn residuals
    relative to the geometric priors. As training progresses, the constraint
    relaxes to allow fine-grained predictions for edge cases.
    """

    def after_train_epoch(self, runner):
        epoch = runner.epoch + 1  # runner.epoch is 0-indexed, so +1 after completion
        total_epochs = runner.max_epochs

        model = runner.model
        # Handle DistributedDataParallel wrapper
        if hasattr(model, 'module'):
            model = model.module

        # Navigate: model -> lane_head -> transformer -> decoder
        lane_head = getattr(model, 'lane_head', None)
        if lane_head is None:
            return

        transformer = getattr(lane_head, 'transformer', None)
        if transformer is None:
            return

        decoder = getattr(transformer, 'decoder', None)
        if decoder is None:
            return

        if hasattr(decoder, 'set_epoch'):
            decoder.set_epoch(epoch, total_epochs)
            current_scale = decoder.get_correction_scale()
            runner.logger.info(
                f'[CorrectionScaleAnnealHook] Epoch {epoch}/{total_epochs}: '
                f'correction_scale = {current_scale:.4f}'
            )
