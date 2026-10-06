from app.schemas.schemas import ElementDetection
from PIL import Image
from itertools import zip_longest
import logging
import zxingcpp


logger = logging.getLogger(__name__)
PDF417_FIELDS = (
    "tramite_number", "surname", "name", "gender",
    "document_number", "category", "birth_date", "issue_date",
)

class PDF417Reader:

    def __init__(self) -> None:
        pass

    def read(self, element: ElementDetection) -> dict | None:
        """Read PDF417 barcode from the specified element in the image using zxing."""
        if element is None:
            return None
        logger.info("Reading PDF417 from element in image of shape: %s", element.crop.shape)
        cropped_img = Image.fromarray(element.crop)
        barcode = zxingcpp.read_barcodes(cropped_img, 
                                         formats=zxingcpp.BarcodeFormat.PDF417)
        if len(barcode) == 1:
            raw = barcode[0].text 
            raw = raw.split("@")
            return dict(zip_longest(PDF417_FIELDS, raw[:len(PDF417_FIELDS)]))
        logger.warning(
            "Failed to read PDF417 from element in image of shape: %s", element.crop.shape
        )
        return None
