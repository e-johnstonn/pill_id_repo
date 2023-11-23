# pill_id_repo
- augment.py - uses albumentations to transform images in a variety of ways, intended to cover cases where lighting/angle of the image might be different
- segmenting.py - uses FastSAM to segment images, resulting in images where the object (pill) is isolated from the background (pill.jpg is an example of the output)
- utils.py - ocr's the pill and tries to match based on the text
- testing.py - storing images in deep lake vector store
