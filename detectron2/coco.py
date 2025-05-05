"""COCO utilities.

Usage: 
    coco_helper = CocoHelper(coco_path)
"""

import json
import logging

# Get module-level logger
logger = logging.getLogger(__name__)


class CocoHelper:
    def __init__(self, coco_path):
        self.coco_path = coco_path
        self._load()

    def validate_segmentations(self):
        errors = []
        for i, ann in enumerate(self.data.get("annotations", [])):
            seg = ann.get("segmentation", None)
            if not isinstance(seg, list):
                errors.append(
                    f"Annotation {i} has invalid segmentation type: {type(seg)}: {ann}")
                continue

            for j, poly in enumerate(seg):
                if not isinstance(poly, list):
                    errors.append(
                        f"Annotation {i}, Polygon {j} is not a list: {ann}")
                    continue
                if any(isinstance(p, list) for p in poly):
                    errors.append(
                        f"Annotation {i}, Polygon {j} is nested (should be flat list): {ann}")
                if len(poly) % 2 != 0:
                    errors.append(
                        f"Annotation {i}, Polygon {j} does not have an even number of coordinates: {ann}")

        return errors

    def _load(self):
        with open(self.coco_path, 'r') as f:
            self.data = json.load(f)

        # Add debug logging
        logger.debug(f"Total images in dataset: {len(self.data['images'])}")
        logger.debug(f"Total annotations: {len(self.data['annotations'])}")

        # Count annotations per category
        category_counts = {}
        for ann in self.data['annotations']:
            cat_id = ann['category_id']
            category_counts[cat_id] = category_counts.get(cat_id, 0) + 1

        logger.debug(f"Annotations per category ID: {category_counts}")

        # Basic validation
        assert 'images' in self.data, "COCO file missing 'images'"
        assert 'annotations' in self.data, "COCO file missing 'annotations'"
        assert 'categories' in self.data, "COCO file missing 'categories'"

        errors = self.validate_segmentations()
        if errors:
            raise ValueError(
                f"Segmentation validation failed with {len(errors)} errors:\n" + "\n".join(errors[:5]))

        # Extract classes in order
        self.categories = sorted(
            self.data['categories'], key=lambda x: x['id'])

        expected_ids = list(range(len(self.categories)))
        actual_ids = [cat['id'] for cat in self.categories]
        assert actual_ids == expected_ids, (
            f"Category IDs must be 0-based and continuous. Found IDs: {actual_ids}"
        )

        self.class_names = [cat['name'] for cat in self.categories]
        self.class_id_map = {cat['id']: i for i,
                             cat in enumerate(self.categories)}

    def get_class_names(self):
        return self.class_names

    def get_class_id_map(self):
        return self.class_id_map

    def get_num_classes(self):
        return len(self.class_names)
