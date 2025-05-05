"""COCO utilities.

Usage: 
    coco_helper = CocoHelper(coco_path)
"""

import json
import logging

# Get module-level logger
logger = logging.getLogger(__name__)


class CocoHelper:
    def __init__(self, coco_path, focus_label=None):
        self.coco_path = coco_path
        self.focus_label = focus_label
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

    def _validate_annotations(self):
        anns_by_image = {}
        for ann in self.data['annotations']:
            anns_by_image.setdefault(ann["image_id"], []).append(ann)

        cat_ids = set(cat['id'] for cat in self.categories)
        focus_cat_id = self._cat_name_to_id.get(
            self.focus_label) if self.focus_label else None

        images_missing_anns = []
        images_with_invalid_cats = []
        images_with_malformed_anns = []
        images_missing_focus = []

        for img in self.data['images']:
            img_id = img['id']
            anns = anns_by_image.get(img_id, [])

            if len(anns) == 0:
                images_missing_anns.append(img_id)
                continue

            has_focus = False
            for ann in anns:
                # Bad category ID
                if ann["category_id"] not in cat_ids:
                    images_with_invalid_cats.append(img_id)
                    break

                # Empty bbox of malformed segmentation
                if (not ann.get("bbox") or
                    len(ann["bbox"]) != 4 or
                    ann["bbox"][2] <= 0 or
                        ann["bbox"][3] <= 0):

                    images_with_malformed_anns.append(img_id)
                    break

                seg = ann.get("segmentation")
                if (not isinstance(seg, list) or
                    not seg or not isinstance(seg[0], list) or
                        len(seg[0]) < 6):

                    images_with_malformed_anns.append(img_id)
                    break

                if focus_cat_id is not None and ann["category_id"] == focus_cat_id:
                    has_focus = True

            if focus_cat_id is not None and not has_focus:
                images_missing_focus.append(img_id)

        # These are not errors per se, just that there are images in the
        # input dataset that don't have any annotations whatsoever and images
        # that don't have the focus label (so they effectively won't be used).
        # NB: when using focus_label to generate annotations, this list turns
        # into the focus_label list because we just have a bunch of images that
        # don't have the focus (or any other) label.
        if len(images_missing_anns) > 0:
            logger.warning(
                f"{len(images_missing_anns)} image(s) have no annotations and will be ignored.")
            for img_id in images_missing_anns:
                img_data = next(
                    (img for img in self.data['images'] if img['id'] == img_id), None)
                filename = img_data.get(
                    'file_name', 'unknown') if img_data else 'unknown'
                logger.debug(f"Image ID: {img_id}, File: {filename}")

        if self.focus_label and len(images_missing_focus) > 0:
            # This is not an error per se, just that there are images in the
            # input dataset that don't have the focus label
            logger.warning(
                f"{len(images_missing_focus)} image(s) do not have any annotation with class '{self.focus_label}'.")
            for img_id in images_missing_focus:
                img_data = next(
                    (img for img in self.data['images'] if img['id'] == img_id), None)
                filename = img_data.get(
                    'file_name', 'unknown') if img_data else 'unknown'
                logger.debug(f"Image ID: {img_id}, File: {filename}")

        errors = []
        if len(images_with_invalid_cats) > 0:
            errors.append(
                f"{len(images_with_invalid_cats)} image(s) have annotations with unknown category IDs.")
            for img_id in images_with_invalid_cats:
                img_data = next(
                    (img for img in self.data['images'] if img['id'] == img_id), None)
                filename = img_data.get(
                    'file_name', 'unknown') if img_data else 'unknown'
                errors.append(f"Image ID: {img_id}, File: {filename}")

        if len(images_with_malformed_anns) > 0:
            errors.append(
                f"{len(images_with_malformed_anns)} image(s) have malformed segmentations or empty bounding boxes.")
            for img_id in images_with_malformed_anns:
                img_data = next(
                    (img for img in self.data['images'] if img['id'] == img_id), None)
                filename = img_data.get(
                    'file_name', 'unknown') if img_data else 'unknown'
                errors.append(f"Image ID: {img_id}, File: {filename}")

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
        self._cat_name_to_id = {
            cat['name']: cat['id'] for cat in self.categories}

        errors = self.validate_segmentations()
        if errors:
            raise ValueError(
                f"Segmentation validation failed with {len(errors)} errors:\n" + "\n".join(errors[:5]))

        errors = self._validate_annotations()
        if errors:
            raise ValueError(
                f"Annotation validation failed with {len(errors)} errors:\n" + "\n".join(errors[:5]))

    def get_class_names(self):
        return self.class_names

    def get_class_id_map(self):
        return self.class_id_map

    def get_num_classes(self):
        return len(self.class_names)
