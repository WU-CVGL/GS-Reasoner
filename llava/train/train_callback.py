from transformers import TrainerCallback


class ConvertToFP32Callback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        model = kwargs["model"]
        spatial_encoder = model.get_model().spatial_encoder


        for param in spatial_encoder.parameters():
            param.data = param.data.float()

        model.get_model().set_spatial_encoder(spatial_encoder)        