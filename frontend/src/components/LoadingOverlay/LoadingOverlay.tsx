import { useUIStore } from '../../stores/uiStore';
import { Card, CardContent } from "@/components/ui/card";

export function LoadingOverlay() {
  const { isLoadingModel, statusMessage } = useUIStore();

  if (!isLoadingModel) return null;

  return (
    <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-[100] animate-in fade-in duration-200">
      <Card className="w-[300px] shadow-2xl border-primary/20 bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <CardContent className="flex flex-col items-center justify-center p-8 gap-6">
          {/* Spinner */}
          <div className="relative">
            <div className="w-12 h-12 rounded-full border-4 border-muted"></div>
            <div className="w-12 h-12 rounded-full border-4 border-primary border-t-transparent animate-spin absolute top-0 left-0"></div>
          </div>

          {/* Message */}
          <div className="text-center space-y-2">
            <p className="text-lg font-semibold tracking-tight">{statusMessage}</p>
            <p className="text-sm text-muted-foreground animate-pulse">
              Processing...
            </p>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
