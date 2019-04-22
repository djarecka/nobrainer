from nobrainer.models.highresnet import highresnet
from nobrainer.models.meshnet import meshnet
from nobrainer.models.meshnet import variational_meshnet
from nobrainer.models.unet import unet


def get(name):
    """Return callable that creates a particular `tf.keras.Model`.

    Parameters
    ----------
    name: str, the name of the model (case-insensitive).

    Returns
    -------
    Callable, which instantiates a `tf.keras.Model` object.
    """
    if not isinstance(name, str):
        raise ValueError("Model name must be a string.")

    models = {
        'highresnet': highresnet,
        'meshnet': meshnet,
        'unet': unet,
        'variational_meshnet': variational_meshnet,
    }

    try:
        return models[name.lower()]
    except KeyError:
        avail = ', '.join(models.keys())
        raise ValueError(
            "Uknown model: '{}'. Available models are {}.".format(name, avail))
