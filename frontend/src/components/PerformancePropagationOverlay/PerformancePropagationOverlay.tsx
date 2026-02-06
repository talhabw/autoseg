import { useUIStore } from '../../stores/uiStore';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { stopAutoTracking } from '../../App';

export function PerformancePropagationOverlay() {
  const performanceProgress = useUIStore((s) => s.performanceProgress);

  if (!performanceProgress) return null;

  const {
    current,
    total,
    successCount,
    failedCount,
    skippedCount,
    startTime,
    failedImageIndex,
    failedLabels,
  } = performanceProgress;

  const elapsed = (Date.now() - startTime) / 1000;
  const rate = current > 0 ? elapsed / current : 0;
  const remaining = rate > 0 ? Math.ceil((total - current) * rate) : 0;
  const percent = total > 0 ? Math.round((current / total) * 100) : 0;

  const isStopped = failedImageIndex !== null;

  const handleStop = () => {
    stopAutoTracking();
    useUIStore.getState().setAutoNext(false);
    useUIStore.getState().setPerformanceProgress(null);
  };

  const handleResume = () => {
    // Clear the failed state and resume - App.tsx handles the actual propagation
    useUIStore.getState().updatePerformanceProgress({
      failedImageIndex: null,
      failedLabels: [],
    });
  };

  return (
    <div className="fixed inset-0 z-[100] bg-black/60 backdrop-blur-sm flex items-center justify-center">
      <Card className="w-[480px] shadow-2xl">
        <CardContent className="pt-6 space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold">
              {isStopped ? 'Propagation Paused' : 'Performance Propagation'}
            </h2>
            <span className="text-sm text-muted-foreground">
              {percent}%
            </span>
          </div>

          {/* Progress bar */}
          <div className="w-full bg-muted rounded-full h-3 overflow-hidden">
            <div
              className={`h-full rounded-full transition-all duration-300 ${
                isStopped ? 'bg-yellow-500' : 'bg-primary'
              }`}
              style={{ width: `${percent}%` }}
            />
          </div>

          {/* Stats */}
          <div className="grid grid-cols-3 gap-2 text-sm">
            <div className="text-center">
              <div className="font-mono text-lg">{current}/{total}</div>
              <div className="text-muted-foreground">Images</div>
            </div>
            <div className="text-center">
              <div className="font-mono text-lg text-green-500">{successCount}</div>
              <div className="text-muted-foreground">Success</div>
            </div>
            <div className="text-center">
              <div className="font-mono text-lg text-red-500">{failedCount}</div>
              <div className="text-muted-foreground">Failed</div>
            </div>
          </div>

          {/* Time stats */}
          <div className="flex justify-between text-xs text-muted-foreground">
            <span>Elapsed: {formatTime(elapsed)}</span>
            <span>Rate: {rate > 0 ? `${rate.toFixed(1)}s/img` : '...'}</span>
            <span>ETA: {remaining > 0 ? formatTime(remaining) : '...'}</span>
          </div>

          {/* Failure info */}
          {isStopped && failedLabels.length > 0 && (
            <div className="bg-yellow-500/10 border border-yellow-500/30 rounded-md p-3 text-sm">
              <div className="font-medium text-yellow-600 dark:text-yellow-400 mb-1">
                Failed to track at image {(failedImageIndex ?? 0) + 1}:
              </div>
              <div className="text-muted-foreground">
                {failedLabels.join(', ')}
              </div>
              <div className="text-xs text-muted-foreground mt-1">
                Fix annotations on this image, then resume.
              </div>
            </div>
          )}

          {/* Actions */}
          <div className="flex gap-2 pt-2">
            {isStopped ? (
              <>
                <Button variant="outline" className="flex-1" onClick={handleStop}>
                  Stop
                </Button>
                <Button className="flex-1" onClick={handleResume}>
                  Resume
                </Button>
              </>
            ) : (
              <Button variant="destructive" className="w-full" onClick={handleStop}>
                Stop Propagation
              </Button>
            )}
          </div>

          {skippedCount > 0 && (
            <div className="text-xs text-muted-foreground text-center">
              {skippedCount} images skipped (duplicates)
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

function formatTime(seconds: number): string {
  if (seconds < 60) return `${Math.round(seconds)}s`;
  const mins = Math.floor(seconds / 60);
  const secs = Math.round(seconds % 60);
  return `${mins}m ${secs}s`;
}
