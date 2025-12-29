"""
QA Validation Engine for annotations.

Validates annotations for common issues:
- Polygon token count valid
- All polygon points in [0,1] range
- Minimum area threshold
- Bounding box validity
"""

from dataclasses import dataclass
from core.models import Annotation


@dataclass
class ValidationWarning:
    """A validation warning."""
    annotation_id: int
    severity: str  # 'error', 'warning', 'info'
    code: str
    message: str


def validate_annotation(
    annotation: Annotation,
    image_width: int,
    image_height: int,
    min_area_ratio: float = 0.0001,  # 0.01% of image area
    max_polygon_points: int = 10000,
) -> list[ValidationWarning]:
    """
    Validate a single annotation.
    
    Args:
        annotation: The annotation to validate
        image_width: Width of the image
        image_height: Height of the image
        min_area_ratio: Minimum annotation area as ratio of image area
        max_polygon_points: Maximum number of polygon points
        
    Returns:
        List of validation warnings
    """
    warnings = []
    
    # Check bounding box
    if annotation.bbox_xyxy:
        x1, y1, x2, y2 = annotation.bbox_xyxy
        
        # Check bbox validity
        if x1 >= x2 or y1 >= y2:
            warnings.append(ValidationWarning(
                annotation_id=annotation.id,
                severity='error',
                code='INVALID_BBOX',
                message=f'Invalid bounding box: ({x1}, {y1}, {x2}, {y2}) - width or height is zero/negative'
            ))
        
        # Check bbox is within image bounds
        if x1 < 0 or y1 < 0 or x2 > image_width or y2 > image_height:
            warnings.append(ValidationWarning(
                annotation_id=annotation.id,
                severity='warning',
                code='BBOX_OUT_OF_BOUNDS',
                message='Bounding box extends outside image bounds'
            ))
        
        # Check minimum area
        bbox_area = (x2 - x1) * (y2 - y1)
        image_area = image_width * image_height
        if bbox_area / image_area < min_area_ratio:
            warnings.append(ValidationWarning(
                annotation_id=annotation.id,
                severity='warning',
                code='BBOX_TOO_SMALL',
                message=f'Bounding box area ({bbox_area:.0f}px) is very small ({bbox_area/image_area*100:.4f}% of image)'
            ))
    else:
        warnings.append(ValidationWarning(
            annotation_id=annotation.id,
            severity='error',
            code='NO_BBOX',
            message='Annotation has no bounding box'
        ))
    
    # Check polygon (normalized coordinates)
    if annotation.polygon_norm:
        polygon = annotation.polygon_norm
        num_points = len(polygon) // 2
        
        # Check point count
        if num_points < 3:
            warnings.append(ValidationWarning(
                annotation_id=annotation.id,
                severity='error',
                code='POLYGON_TOO_FEW_POINTS',
                message=f'Polygon has only {num_points} points (minimum 3 required)'
            ))
        
        if num_points > max_polygon_points:
            warnings.append(ValidationWarning(
                annotation_id=annotation.id,
                severity='warning',
                code='POLYGON_TOO_MANY_POINTS',
                message=f'Polygon has {num_points} points (max recommended: {max_polygon_points})'
            ))
        
        # Check all points are in [0, 1] range
        for i in range(num_points):
            x = polygon[i * 2]
            y = polygon[i * 2 + 1]
            if not (0 <= x <= 1 and 0 <= y <= 1):
                warnings.append(ValidationWarning(
                    annotation_id=annotation.id,
                    severity='warning',
                    code='POLYGON_POINT_OUT_OF_RANGE',
                    message=f'Polygon point {i} ({x:.4f}, {y:.4f}) is outside [0,1] range'
                ))
                break  # Only report first out-of-range point
        
        # Check polygon area (using shoelace formula on normalized coords)
        polygon_area = 0.0
        for i in range(num_points):
            j = (i + 1) % num_points
            x1, y1 = polygon[i * 2], polygon[i * 2 + 1]
            x2, y2 = polygon[j * 2], polygon[j * 2 + 1]
            polygon_area += x1 * y2 - x2 * y1
        polygon_area = abs(polygon_area) / 2.0
        
        # Normalized area ratio (polygon in normalized coords, so area is 0-1)
        if polygon_area < min_area_ratio:
            warnings.append(ValidationWarning(
                annotation_id=annotation.id,
                severity='warning',
                code='POLYGON_TOO_SMALL',
                message=f'Polygon area ({polygon_area*100:.4f}% of image) is very small'
            ))
    
    # Check mask
    if annotation.mask_rle is None and annotation.polygon_norm is None:
        warnings.append(ValidationWarning(
            annotation_id=annotation.id,
            severity='info',
            code='NO_SEGMENTATION',
            message='Annotation has no mask or polygon (only bounding box)'
        ))
    
    # Check status
    if annotation.status == 'rejected':
        warnings.append(ValidationWarning(
            annotation_id=annotation.id,
            severity='info',
            code='REJECTED_ANNOTATION',
            message='Annotation is marked as rejected'
        ))
    
    return warnings


def validate_annotations(
    annotations: list[Annotation],
    image_width: int,
    image_height: int,
) -> list[ValidationWarning]:
    """Validate multiple annotations."""
    all_warnings = []
    for ann in annotations:
        all_warnings.extend(validate_annotation(ann, image_width, image_height))
    return all_warnings


@dataclass
class ProjectValidationReport:
    """Validation report for entire project."""
    total_annotations: int
    total_images: int
    errors: list[ValidationWarning]
    warnings: list[ValidationWarning]
    info: list[ValidationWarning]
    
    @property
    def is_valid(self) -> bool:
        """Project is valid if there are no errors."""
        return len(self.errors) == 0
    
    @property
    def error_count(self) -> int:
        return len(self.errors)
    
    @property
    def warning_count(self) -> int:
        return len(self.warnings)
    
    def summary(self) -> str:
        """Get a summary string."""
        return (
            f"Validation Report:\n"
            f"  Images: {self.total_images}\n"
            f"  Annotations: {self.total_annotations}\n"
            f"  Errors: {self.error_count}\n"
            f"  Warnings: {self.warning_count}\n"
            f"  Valid: {'Yes' if self.is_valid else 'No'}"
        )


def validate_project(store, project_id: int) -> ProjectValidationReport:
    """
    Validate all annotations in a project.
    
    Args:
        store: The AnnotationStore instance
        project_id: ID of the project to validate
        
    Returns:
        ProjectValidationReport with all issues found
    """
    images = store.list_images(project_id)
    
    all_warnings = []
    total_annotations = 0
    
    for image in images:
        annotations = store.list_annotations(image.id)
        total_annotations += len(annotations)
        
        for ann in annotations:
            warnings = validate_annotation(ann, image.width, image.height)
            all_warnings.extend(warnings)
    
    # Separate by severity
    errors = [w for w in all_warnings if w.severity == 'error']
    warnings = [w for w in all_warnings if w.severity == 'warning']
    info = [w for w in all_warnings if w.severity == 'info']
    
    return ProjectValidationReport(
        total_annotations=total_annotations,
        total_images=len(images),
        errors=errors,
        warnings=warnings,
        info=info,
    )
