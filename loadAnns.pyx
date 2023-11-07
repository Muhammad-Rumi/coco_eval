from pycocotools.coco import COCO

def loadAnns(annFile):
    """
    Load COCO annotations.

    Args:
        annFile (str): Path to the COCO annotations file.

    Returns:
        list[dict]: A list of COCO annotations.
    """

    coco = COCO(annFile)
    anns = coco.loadAnns(coco.getAnnIds())
    return anns