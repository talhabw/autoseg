"""
Tests for ProjectStore CRUD operations.
"""

import os
import tempfile
import pytest
from pathlib import Path

from core.store import ProjectStore
from core.models import Project, Label, ImageRecord, Annotation


@pytest.fixture
def temp_project_dir():
    """Create a temporary directory for project files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def temp_image_dir(temp_project_dir):
    """Create a temporary directory with test images."""
    img_dir = os.path.join(temp_project_dir, "images")
    os.makedirs(img_dir)
    
    # Create some dummy image files (1x1 pixel PNGs)
    from PIL import Image
    
    for i in range(5):
        img = Image.new('RGB', (100 + i * 10, 100 + i * 10), color='red')
        img.save(os.path.join(img_dir, f"image_{i:02d}.png"))
    
    yield img_dir


class TestProjectStore:
    """Tests for ProjectStore operations."""
    
    def test_create_project(self, temp_project_dir, temp_image_dir):
        """Test creating a new project."""
        project = ProjectStore.create_project(
            project_dir=temp_project_dir,
            image_dir=temp_image_dir,
            name="Test Project"
        )
        
        assert project.id == 1
        assert project.name == "Test Project"
        assert os.path.exists(project.db_path)
    
    def test_load_project(self, temp_project_dir, temp_image_dir):
        """Test loading an existing project."""
        # Create project
        ProjectStore.create_project(
            project_dir=temp_project_dir,
            image_dir=temp_image_dir,
            name="Test Project"
        )
        
        # Load project
        project = ProjectStore.load_project(temp_project_dir)
        
        assert project.name == "Test Project"
    
    def test_list_images(self, temp_project_dir, temp_image_dir):
        """Test listing images in a project."""
        project = ProjectStore.create_project(
            project_dir=temp_project_dir,
            image_dir=temp_image_dir,
            name="Test Project"
        )
        
        store = ProjectStore(project.db_path)
        try:
            images = store.list_images(project.id)
            
            assert len(images) == 5
            assert images[0].order_index == 0
            assert images[4].order_index == 4
            # Check natural sort order
            assert "image_00" in images[0].path
            assert "image_04" in images[4].path
        finally:
            store.close()
    
    def test_get_image_by_index(self, temp_project_dir, temp_image_dir):
        """Test getting image by index."""
        project = ProjectStore.create_project(
            project_dir=temp_project_dir,
            image_dir=temp_image_dir,
            name="Test Project"
        )
        
        store = ProjectStore(project.db_path)
        try:
            image = store.get_image_by_index(project.id, 2)
            
            assert image is not None
            assert image.order_index == 2
            assert "image_02" in image.path
        finally:
            store.close()
    
    def test_upsert_label(self, temp_project_dir, temp_image_dir):
        """Test creating and updating labels."""
        project = ProjectStore.create_project(
            project_dir=temp_project_dir,
            image_dir=temp_image_dir,
            name="Test Project"
        )
        
        store = ProjectStore(project.db_path)
        try:
            # Create label
            label1 = store.upsert_label(project.id, "car")
            assert label1.name == "car"
            assert label1.color_hex is not None
            
            # Create another label
            label2 = store.upsert_label(project.id, "person", "#00FF00")
            assert label2.name == "person"
            assert label2.color_hex == "#00FF00"
            
            # Upsert existing label (should return same)
            label1_again = store.upsert_label(project.id, "car")
            assert label1_again.id == label1.id
        finally:
            store.close()
    
    def test_list_labels(self, temp_project_dir, temp_image_dir):
        """Test listing labels."""
        project = ProjectStore.create_project(
            project_dir=temp_project_dir,
            image_dir=temp_image_dir,
            name="Test Project"
        )
        
        store = ProjectStore(project.db_path)
        try:
            store.upsert_label(project.id, "car")
            store.upsert_label(project.id, "person")
            store.upsert_label(project.id, "bike")
            
            labels = store.list_labels(project.id)
            
            assert len(labels) == 3
            names = [l.name for l in labels]
            assert "car" in names
            assert "person" in names
            assert "bike" in names
        finally:
            store.close()
    
    def test_create_annotation(self, temp_project_dir, temp_image_dir):
        """Test creating an annotation."""
        project = ProjectStore.create_project(
            project_dir=temp_project_dir,
            image_dir=temp_image_dir,
            name="Test Project"
        )
        
        store = ProjectStore(project.db_path)
        try:
            # Get an image
            image = store.get_image_by_index(project.id, 0)
            
            # Create a label
            label = store.upsert_label(project.id, "car")
            
            # Create annotation
            annotation = store.create_annotation(
                image_id=image.id,
                label_id=label.id,
                bbox_xyxy=[10.0, 20.0, 100.0, 150.0],
                source="manual",
                status="approved"
            )
            
            assert annotation.id == 1
            assert annotation.image_id == image.id
            assert annotation.label_id == label.id
            assert annotation.bbox_xyxy == [10.0, 20.0, 100.0, 150.0]
            assert annotation.source == "manual"
            assert annotation.status == "approved"
        finally:
            store.close()
    
    def test_update_annotation(self, temp_project_dir, temp_image_dir):
        """Test updating an annotation."""
        project = ProjectStore.create_project(
            project_dir=temp_project_dir,
            image_dir=temp_image_dir,
            name="Test Project"
        )
        
        store = ProjectStore(project.db_path)
        try:
            image = store.get_image_by_index(project.id, 0)
            label = store.upsert_label(project.id, "car")
            
            annotation = store.create_annotation(
                image_id=image.id,
                label_id=label.id,
                bbox_xyxy=[10.0, 20.0, 100.0, 150.0]
            )
            
            # Update bbox
            updated = store.update_annotation(
                annotation.id,
                bbox_xyxy=[15.0, 25.0, 110.0, 160.0]
            )
            
            assert updated.bbox_xyxy == [15.0, 25.0, 110.0, 160.0]
            
            # Update status
            updated = store.update_annotation(
                annotation.id,
                status="needs_review"
            )
            
            assert updated.status == "needs_review"
        finally:
            store.close()
    
    def test_delete_annotation(self, temp_project_dir, temp_image_dir):
        """Test deleting an annotation."""
        project = ProjectStore.create_project(
            project_dir=temp_project_dir,
            image_dir=temp_image_dir,
            name="Test Project"
        )
        
        store = ProjectStore(project.db_path)
        try:
            image = store.get_image_by_index(project.id, 0)
            label = store.upsert_label(project.id, "car")
            
            annotation = store.create_annotation(
                image_id=image.id,
                label_id=label.id,
                bbox_xyxy=[10.0, 20.0, 100.0, 150.0]
            )
            
            # Delete
            store.delete_annotation(annotation.id)
            
            # Verify deleted
            result = store.get_annotation_by_id(annotation.id)
            assert result is None
        finally:
            store.close()
    
    def test_list_annotations(self, temp_project_dir, temp_image_dir):
        """Test listing annotations for an image."""
        project = ProjectStore.create_project(
            project_dir=temp_project_dir,
            image_dir=temp_image_dir,
            name="Test Project"
        )
        
        store = ProjectStore(project.db_path)
        try:
            image = store.get_image_by_index(project.id, 0)
            label1 = store.upsert_label(project.id, "car")
            label2 = store.upsert_label(project.id, "person")
            
            store.create_annotation(
                image_id=image.id,
                label_id=label1.id,
                bbox_xyxy=[10.0, 20.0, 100.0, 150.0]
            )
            store.create_annotation(
                image_id=image.id,
                label_id=label2.id,
                bbox_xyxy=[200.0, 50.0, 300.0, 200.0]
            )
            
            annotations = store.list_annotations(image.id)
            
            assert len(annotations) == 2
        finally:
            store.close()
    
    def test_roundtrip_persistence(self, temp_project_dir, temp_image_dir):
        """Test that data persists across store instances."""
        # Create project and add data
        project = ProjectStore.create_project(
            project_dir=temp_project_dir,
            image_dir=temp_image_dir,
            name="Test Project"
        )
        
        store = ProjectStore(project.db_path)
        image = store.get_image_by_index(project.id, 0)
        label = store.upsert_label(project.id, "test_label")
        annotation = store.create_annotation(
            image_id=image.id,
            label_id=label.id,
            bbox_xyxy=[10.0, 20.0, 100.0, 150.0]
        )
        store.close()
        
        # Reload and verify
        loaded_project = ProjectStore.load_project(temp_project_dir)
        store2 = ProjectStore(loaded_project.db_path)
        
        try:
            labels = store2.list_labels(loaded_project.id)
            assert len(labels) == 1
            assert labels[0].name == "test_label"
            
            images = store2.list_images(loaded_project.id)
            assert len(images) == 5
            
            annotations = store2.list_annotations(image.id)
            assert len(annotations) == 1
            assert annotations[0].bbox_xyxy == [10.0, 20.0, 100.0, 150.0]
        finally:
            store2.close()
