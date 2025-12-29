import { useProjectStore } from '../../stores/projectStore';
import { LabelPicker } from '../LabelPicker/LabelPicker';
import { AnnotationList } from '../AnnotationList/AnnotationList';
import { Button } from '@/components/ui/button';
import { Separator } from '@/components/ui/separator';
import { ChevronLeft, ChevronRight } from 'lucide-react';

export function Sidebar() {
  const {
    images,
    currentImageIndex,
    nextImage,
    prevImage,
    project,
  } = useProjectStore();

  const currentImage = images[currentImageIndex];

  if (!project) {
    return (
      <aside className="w-80 bg-background border-l flex flex-col">
        <div className="flex-1 flex items-center justify-center p-4">
          <p className="text-muted-foreground text-sm text-center">
            Open or create a project to get started
          </p>
        </div>
      </aside>
    );
  }

  return (
    <aside className="w-80 shrink-0 bg-background border-l flex flex-col shadow-xl z-20">
      {/* Navigation */}
      <div className="p-4 border-b">
        <div className="flex items-center gap-2 mb-2">
          <Button
            variant="outline"
            size="sm"
            className="flex-1 h-8"
            onClick={prevImage}
            disabled={currentImageIndex <= 0}
            title="Previous Image (A / Left Arrow)"
          >
            <ChevronLeft className="w-4 h-4 mr-1" /> Prev
          </Button>

          <div className="flex-1 text-center font-mono text-sm">
            {currentImageIndex + 1} / {images.length}
          </div>

          <Button
            variant="outline"
            size="sm"
            className="flex-1 h-8"
            onClick={nextImage}
            disabled={currentImageIndex >= images.length - 1}
            title="Next Image (D / Right Arrow)"
          >
            Next <ChevronRight className="w-4 h-4 ml-1" />
          </Button>
        </div>

        {currentImage && (
          <p className="text-xs text-muted-foreground text-center truncate px-2" title={currentImage.path}>
            {currentImage.path.split('/').pop()}
          </p>
        )}
      </div>

      {/* Label picker */}
      <div className="p-4 border-b bg-muted/10">
        <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-3 flex items-center gap-2">
          <span>Active Label</span>
          <Separator className="flex-1" />
        </h3>
        <LabelPicker />
      </div>

      {/* Annotation list */}
      <div className="flex-1 overflow-hidden flex flex-col">
        <div className="p-4 pb-2">
          <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider flex items-center gap-2">
            <span>Annotations</span>
            {/* Dynamic count would be nice here but requires store access if we want total */}
            <Separator className="flex-1" />
          </h3>
        </div>
        <div className="flex-1 overflow-y-auto px-4 pb-4">
          <AnnotationList />
        </div>
      </div>
    </aside>
  );
}
