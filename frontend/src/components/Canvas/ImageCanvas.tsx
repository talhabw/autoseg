import { useEffect, useRef, useState, useCallback, useMemo } from 'react';
import { Stage, Layer, Image as KonvaImage, Rect, Circle, Transformer } from 'react-konva';
import Konva from 'konva';
import { useProjectStore } from '../../stores/projectStore';
import { useAnnotationStore } from '../../stores/annotationStore';
import { useUIStore } from '../../stores/uiStore';
import { getOptimizedImageUrl } from '../../api/client';
import { maskToCanvas, type RLEMask } from '../../utils/rle';
import type { Annotation, DrawingBbox } from '../../types';

export function ImageCanvas() {
  const containerRef = useRef<HTMLDivElement>(null);
  const stageRef = useRef<Konva.Stage>(null);
  const transformerRef = useRef<Konva.Transformer>(null);
  const imageRef = useRef<Konva.Image>(null);

  const [containerSize, setContainerSize] = useState({ width: 800, height: 600 });
  const [image, setImage] = useState<HTMLImageElement | null>(null);
  const [stageScale, setStageScale] = useState(1);
  const [stagePos, setStagePos] = useState({ x: 0, y: 0 });
  const [drawingBbox, setDrawingBbox] = useState<DrawingBbox | null>(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [isImageLoading, setIsImageLoading] = useState(true);

  // Cache for decoded mask canvases (previous ref for cleanup)
  const prevMaskCanvasesRef = useRef<Map<number, HTMLCanvasElement>>(new Map());

  const { images, currentImageIndex } = useProjectStore();
  const {
    annotations,
    selectedAnnotationId,
    selectAnnotation,
    selectedLabelId,
    createAnnotation,
    updateAnnotation,
    labels,
    refinePoints,
    addRefinePoint,
  } = useAnnotationStore();
  const { mode, maskOpacity, performanceProgress } = useUIStore();

  const currentImage = images[currentImageIndex];
  // Skip heavy rendering when performance propagation is running (no failed image = loop active)
  const isPerformanceLoopActive = performanceProgress !== null && performanceProgress.failedImageIndex === null;

  // Get label color
  const getLabelColor = useCallback((labelId: number): string => {
    const label = labels.find((l) => l.id === labelId);
    return label?.color || '#888888';
  }, [labels]);

  // Decode masks when annotations change (skip during performance loop)
  const maskCanvases = useMemo(() => {
    if (isPerformanceLoopActive) return prevMaskCanvasesRef.current;

    // Don't mutate previous canvases - Konva may still be rendering them.
    // Just replace the reference and let GC handle cleanup.
    const newCanvases = new Map<number, HTMLCanvasElement>();

    for (const ann of annotations) {
      if (ann.mask_rle) {
        try {
          const rle = ann.mask_rle as RLEMask;
          const color = getLabelColor(ann.label_id);
          const canvas = maskToCanvas(rle, color, maskOpacity);
          // Only add valid canvases (non-zero dimensions)
          if (canvas.width > 0 && canvas.height > 0) {
            newCanvases.set(ann.id, canvas);
          }
        } catch (err) {
          console.error('Failed to decode mask for annotation', ann.id, err);
        }
      }
    }

    prevMaskCanvasesRef.current = newCanvases;
    return newCanvases;
  }, [annotations, maskOpacity, getLabelColor, isPerformanceLoopActive]);

  // Get project for cache busting
  const project = useProjectStore((s) => s.project);

  const fitToView = useCallback((img: HTMLImageElement) => {
    if (!containerRef.current) return;

    const padding = 40;
    const scaleX = (containerSize.width - padding) / img.width;
    const scaleY = (containerSize.height - padding) / img.height;
    const scale = Math.min(scaleX, scaleY, 1);

    const x = (containerSize.width - img.width * scale) / 2;
    const y = (containerSize.height - img.height * scale) / 2;

    setStageScale(scale);
    setStagePos({ x, y });
  }, [containerSize]);

  // Load image when current image changes (skip during performance loop)
  useEffect(() => {
    if (!currentImage || isPerformanceLoopActive) {
      if (!currentImage) {
        setImage(null);
        setIsImageLoading(false);
      }
      return;
    }

    // Set loading state - this will hide stale annotations
    setIsImageLoading(true);
    let cancelled = false;

    const img = new window.Image();
    img.crossOrigin = 'anonymous';
    // Use optimized image endpoint for faster loading
    img.src = getOptimizedImageUrl(currentImage.id, 2048, 85, project?.id);

    img.onload = () => {
      if (!cancelled) {
        setImage(img);
        fitToView(img);
        setIsImageLoading(false);
      }
    };

    img.onerror = () => {
      console.error('Failed to load image:', currentImage.id);
      if (!cancelled) {
        setImage(null);
        setIsImageLoading(false);
      }
    };

    // Cleanup: abort image loading on unmount/change
    return () => {
      cancelled = true;
      img.onload = null;
      img.onerror = null;
      img.src = ''; // Abort loading
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps -- depend on currentImage?.id not the full object to avoid unnecessary reloads
  }, [currentImage?.id, project?.id, isPerformanceLoopActive, fitToView]);

  // Resize handler
  useEffect(() => {
    const updateSize = () => {
      if (containerRef.current) {
        setContainerSize({
          width: containerRef.current.offsetWidth,
          height: containerRef.current.offsetHeight,
        });
      }
    };

    updateSize();

    const resizeObserver = new ResizeObserver(() => {
      updateSize();
    });

    if (containerRef.current) {
      resizeObserver.observe(containerRef.current);
    }

    return () => {
      resizeObserver.disconnect();
    };
  }, []);

  // Cleanup Konva Stage on unmount to release WebGL/Canvas resources
  useEffect(() => {
    const stage = stageRef.current;
    return () => {
      stage?.destroy();
    };
  }, []);

  // Update transformer when selection changes
  useEffect(() => {
    if (!transformerRef.current || !stageRef.current) return;

    const selectedNode = stageRef.current.findOne(`#annotation-${selectedAnnotationId}`);
    if (selectedNode && mode !== 'view') {
      transformerRef.current.nodes([selectedNode]);
    } else {
      transformerRef.current.nodes([]);
    }
    transformerRef.current.getLayer()?.batchDraw();
  }, [selectedAnnotationId, mode]);

  // Wheel zoom
  const handleWheel = (e: Konva.KonvaEventObject<WheelEvent>) => {
    e.evt.preventDefault();

    const stage = stageRef.current;
    if (!stage) return;

    const oldScale = stageScale;
    const pointer = stage.getPointerPosition();
    if (!pointer) return;

    const mousePointTo = {
      x: (pointer.x - stagePos.x) / oldScale,
      y: (pointer.y - stagePos.y) / oldScale,
    };

    const scaleBy = 1.1;
    const newScale = e.evt.deltaY < 0 ? oldScale * scaleBy : oldScale / scaleBy;
    const clampedScale = Math.max(0.1, Math.min(10, newScale));

    setStageScale(clampedScale);
    setStagePos({
      x: pointer.x - mousePointTo.x * clampedScale,
      y: pointer.y - mousePointTo.y * clampedScale,
    });
  };

  // Get pointer position in image coordinates
  const getImagePointer = (): { x: number; y: number } | null => {
    const stage = stageRef.current;
    if (!stage || !image) return null;

    const pointer = stage.getPointerPosition();
    if (!pointer) return null;

    return {
      x: (pointer.x - stagePos.x) / stageScale,
      y: (pointer.y - stagePos.y) / stageScale,
    };
  };

  // Mouse down
  const handleMouseDown = (e: Konva.KonvaEventObject<MouseEvent>) => {
    const pos = getImagePointer();
    if (!pos || !currentImage) return;

    // In refine mode, allow clicking on the selected annotation's bbox
    if (mode === 'refine' && selectedAnnotationId) {
      // Get the selected annotation's bbox
      const selectedAnn = annotations.find((a) => a.id === selectedAnnotationId);
      if (!selectedAnn?.bbox) return;

      const [bx1, by1, bx2, by2] = selectedAnn.bbox;

      // Check if click is within the bbox
      const isInsideBbox = pos.x >= bx1 && pos.x <= bx2 && pos.y >= by1 && pos.y <= by2;
      if (!isInsideBbox) return;  // Ignore clicks outside the bbox

      // Check if clicking near an existing point (to remove it)
      const CLICK_THRESHOLD = 15 / stageScale;  // 15 pixels at current zoom
      const nearbyPointIndex = refinePoints.findIndex((pt) => {
        const dist = Math.sqrt((pt.x - pos.x) ** 2 + (pt.y - pos.y) ** 2);
        return dist < CLICK_THRESHOLD;
      });

      if (nearbyPointIndex >= 0) {
        // Remove the point
        const newPoints = refinePoints.filter((_, i) => i !== nearbyPointIndex);
        // Use direct store update since we need to filter
        useAnnotationStore.setState({ refinePoints: newPoints });
      } else {
        // Add new refine point
        addRefinePoint({
          x: pos.x,
          y: pos.y,
          type: e.evt.button === 2 ? 'negative' : 'positive',
        });
      }
      return;
    }

    // In view mode, ignore if clicking on an annotation (allow selection to handle it)
    // In draw mode, we want to draw through existing annotations
    if (mode === 'view' && e.target !== stageRef.current && e.target !== imageRef.current) return;

    if (mode === 'draw') {
      // If clicking on the currently selected annotation's bbox border, don't start drawing
      // This allows dragging/resizing the selected bbox
      const targetId = (e.target as Konva.Node).id?.();
      if (targetId === `annotation-${selectedAnnotationId}`) {
        // User is clicking on the selected bbox to drag/resize it
        return;
      }
      
      setIsDrawing(true);
      setDrawingBbox({ x1: pos.x, y1: pos.y, x2: pos.x, y2: pos.y });
      selectAnnotation(null);
    } else if (mode === 'view') {
      // Will be handled by stage dragging
    }
  };

  // Mouse move
  const handleMouseMove = () => {
    const pos = getImagePointer();
    
    if (!isDrawing || mode !== 'draw' || !pos) return;

    setDrawingBbox((prev) => prev ? { ...prev, x2: pos.x, y2: pos.y } : null);
  };

  // Mouse up
  const handleMouseUp = async () => {
    if (!isDrawing || mode !== 'draw' || !drawingBbox || !currentImage) {
      setIsDrawing(false);
      setDrawingBbox(null);
      return;
    }

    setIsDrawing(false);

    // Normalize bbox
    const x1 = Math.min(drawingBbox.x1, drawingBbox.x2);
    const y1 = Math.min(drawingBbox.y1, drawingBbox.y2);
    const x2 = Math.max(drawingBbox.x1, drawingBbox.x2);
    const y2 = Math.max(drawingBbox.y1, drawingBbox.y2);

    // Minimum size check
    if (x2 - x1 < 5 || y2 - y1 < 5) {
      setDrawingBbox(null);
      return;
    }

    // Create annotation
    if (selectedLabelId) {
      try {
        await createAnnotation({
          image_id: currentImage.id,
          label_id: selectedLabelId,
          bbox: [x1, y1, x2, y2],
          source: 'manual',
          status: 'approved',
        });
      } catch (err) {
        console.error('Failed to create annotation:', err);
      }
    }

    setDrawingBbox(null);
  };

  // Handle bbox transform (resize/move)
  const handleBboxTransformEnd = async (ann: Annotation, node: Konva.Rect) => {
    const scaleX = node.scaleX();
    const scaleY = node.scaleY();

    // Reset scale
    node.scaleX(1);
    node.scaleY(1);

    const bbox = ann.bbox;
    if (!bbox) return;

    const width = (bbox[2] - bbox[0]) * scaleX;
    const height = (bbox[3] - bbox[1]) * scaleY;

    const newBbox: [number, number, number, number] = [
      node.x(),
      node.y(),
      node.x() + width,
      node.y() + height,
    ];

    try {
      await updateAnnotation(ann.id, { bbox: newBbox });
    } catch (err) {
      console.error('Failed to update annotation:', err);
    }
  };

  // Right-click context menu prevention
  const handleContextMenu = (e: Konva.KonvaEventObject<PointerEvent>) => {
    e.evt.preventDefault();
  };

  if (!currentImage) {
    return (
      <div
        ref={containerRef}
        className="flex-1 bg-background flex items-center justify-center"
      >
        <p className="text-muted-foreground">No image selected</p>
      </div>
    );
  }

  return (
    <div
      ref={containerRef}
      className="flex-1 min-w-0 bg-background overflow-hidden cursor-crosshair"
      style={{ cursor: mode === 'view' ? 'grab' : mode === 'refine' ? 'crosshair' : 'crosshair' }}
    >
      <Stage
        ref={stageRef}
        width={containerSize.width}
        height={containerSize.height}
        scaleX={stageScale}
        scaleY={stageScale}
        x={stagePos.x}
        y={stagePos.y}
        draggable={mode === 'view'}
        onWheel={handleWheel}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onContextMenu={handleContextMenu}
        onDragEnd={(e) => {
          if (e.target === stageRef.current) {
            setStagePos({ x: e.target.x(), y: e.target.y() });
          }
        }}
      >
        {/* Image layer */}
        <Layer>
          {image && (
            <KonvaImage
              ref={imageRef}
              image={image}
              x={0}
              y={0}
              width={currentImage.width}
              height={currentImage.height}
            />
          )}
        </Layer>

        {/* Mask overlay layer - only show when image is loaded and annotations match current image */}
        <Layer>
          {!isImageLoading && annotations
            .filter((ann) => ann.image_id === currentImage?.id)
            .map((ann) => {
              const maskCanvas = maskCanvases.get(ann.id);
              // Skip invalid canvases (null or 0 dimensions) to prevent Konva drawImage errors
              if (!maskCanvas || maskCanvas.width === 0 || maskCanvas.height === 0) return null;

              return (
                <KonvaImage
                  key={`mask-${ann.id}`}
                  image={maskCanvas}
                  x={0}
                  y={0}
                  listening={false}
                />
              );
            })}
        </Layer>

        {/* Annotations layer (bboxes) - only show when image is loaded and annotations match current image */}
        <Layer>
          {/* Existing annotations */}
          {!isImageLoading && annotations
            .filter((ann) => ann.image_id === currentImage?.id)
            .map((ann) => {
              if (!ann.bbox) return null;
              const [x1, y1, x2, y2] = ann.bbox;
              const isSelected = ann.id === selectedAnnotationId;
              const color = getLabelColor(ann.label_id);

              return (
                <Rect
                  key={ann.id}
                  id={`annotation-${ann.id}`}
                  x={x1}
                  y={y1}
                  width={x2 - x1}
                  height={y2 - y1}
                  stroke={color}
                  strokeWidth={isSelected ? 3 : 2}
                  fill={isSelected && !maskCanvases.has(ann.id) ? `${color}20` : 'transparent'}
                  fillEnabled={mode !== 'refine' || !isSelected}
                  hitStrokeWidth={mode === 'refine' && isSelected ? 10 : 0}
                  draggable={mode === 'refine' || (mode === 'draw' && isSelected)}
                  onClick={() => selectAnnotation(ann.id)}
                  onTap={() => selectAnnotation(ann.id)}
                  onDragEnd={(e) => {
                    const node = e.target as Konva.Rect;
                    const newBbox: [number, number, number, number] = [
                      node.x(),
                      node.y(),
                      node.x() + (x2 - x1),
                      node.y() + (y2 - y1),
                    ];
                    updateAnnotation(ann.id, { bbox: newBbox });
                  }}
                  onTransformEnd={(e) => handleBboxTransformEnd(ann, e.target as Konva.Rect)}
                />
              );
            })}

          {/* Drawing bbox */}
          {drawingBbox && (
            <Rect
              x={Math.min(drawingBbox.x1, drawingBbox.x2)}
              y={Math.min(drawingBbox.y1, drawingBbox.y2)}
              width={Math.abs(drawingBbox.x2 - drawingBbox.x1)}
              height={Math.abs(drawingBbox.y2 - drawingBbox.y1)}
              stroke={selectedLabelId ? getLabelColor(selectedLabelId) : '#888888'}
              strokeWidth={2}
              dash={[5, 5]}
            />
          )}

          {/* Transformer */}
          <Transformer
            ref={transformerRef}
            boundBoxFunc={(oldBox, newBox) => {
              // Limit min size
              if (newBox.width < 10 || newBox.height < 10) {
                return oldBox;
              }
              return newBox;
            }}
          />
        </Layer>

        {/* Refine points layer */}
        <Layer>
          {refinePoints.map((point, idx) => (
            <Circle
              key={idx}
              x={point.x}
              y={point.y}
              radius={6 / stageScale}
              fill={point.type === 'positive' ? '#22c55e' : '#ef4444'}
              stroke="white"
              strokeWidth={2 / stageScale}
            />
          ))}
        </Layer>
      </Stage>
    </div>
  );
}
