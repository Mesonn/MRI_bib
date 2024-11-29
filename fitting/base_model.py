
# Fitting Interface-like 



class BaseModel:
    """
    Base class for MRI fitting models.

    Methods
    -------
    fit(images, mask=None)
        Fit the model to the provided images.
    """
    def fit(self, images, mask=None):
        raise NotImplementedError("Must override fit method")