import segmentation_models_pytorch as smp
import re


def get_model(model_type, input_dim, output_dim, device='cuda:0'):
    return get_smp_model(model_type, output_dim, device)


def get_smp_model(model_type, output_dim, device):
    rgx = r'smp_(.*)'
    match = re.match(rgx, model_type)
    assert bool(match), f'no match found for smp net: {model_type}'

    model_type = match.group(1)
    model = smp.Unet(
        model_type, classes=output_dim,
        encoder_weights='imagenet',
        activation='identity'
    )
    return model.to(device)
