from torchvision import transforms


class NIHChestXrayTransforms:
    """Class to apply Image Transformations.
        # TODO: Add more transformations.
    """
    def __init__(self, image_size: int = 256):
        self.image_size = image_size
        self.transformations = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ])

    def __call__(self, image):
        return self.transformations(image)
