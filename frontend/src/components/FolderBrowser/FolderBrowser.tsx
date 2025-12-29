import { useState, useEffect, useCallback } from 'react';
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Folder,
  FolderOpen,
  ChevronUp,
  Home,
  RefreshCw,
  Check,
  X,
} from "lucide-react";
import { cn } from "@/lib/utils";
import * as api from '../../api/client';

interface FolderBrowserProps {
  value: string;
  onChange: (path: string) => void;
  onSelect?: (path: string) => void;
  placeholder?: string;
  className?: string;
}

export function FolderBrowser({
  value,
  onChange,
  onSelect,
  placeholder = "/path/to/folder",
  className,
}: FolderBrowserProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [currentPath, setCurrentPath] = useState('/');
  const [entries, setEntries] = useState<api.DirectoryEntry[]>([]);
  const [parentPath, setParentPath] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Load directory contents
  const loadDirectory = useCallback(async (path: string) => {
    setIsLoading(true);
    setError(null);
    try {
      const result = await api.listDirectory(path);
      setCurrentPath(result.path);
      setParentPath(result.parent);
      setEntries(result.entries);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to load directory');
      setEntries([]);
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Load home directory when first opened
  const loadHome = useCallback(async () => {
    try {
      const result = await api.getHomeDirectory();
      await loadDirectory(result.path);
    } catch {
      await loadDirectory('/');
    }
  }, [loadDirectory]);

  // When browser opens, navigate to current value or home
  useEffect(() => {
    if (isOpen) {
      if (value.trim()) {
        loadDirectory(value.trim());
      } else {
        loadHome();
      }
    }
  }, [isOpen, value, loadDirectory, loadHome]);

  const handleEntryClick = (entry: api.DirectoryEntry) => {
    if (entry.is_dir) {
      loadDirectory(entry.path);
    }
  };

  const handleSelect = () => {
    onChange(currentPath);
    onSelect?.(currentPath);
    setIsOpen(false);
  };

  const handleCancel = () => {
    setIsOpen(false);
  };

  if (!isOpen) {
    return (
      <div className={cn("flex gap-2", className)}>
        <Input
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder={placeholder}
          className="flex-1"
        />
        <Button
          type="button"
          variant="outline"
          size="icon"
          onClick={() => setIsOpen(true)}
          title="Browse folders"
        >
          <FolderOpen className="w-4 h-4" />
        </Button>
      </div>
    );
  }

  return (
    <div className={cn("border rounded-lg p-3 space-y-3 bg-background", className)}>
      {/* Current path bar */}
      <div className="flex items-center gap-2">
        <Input
          value={currentPath}
          onChange={(e) => setCurrentPath(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter') {
              loadDirectory(currentPath);
            }
          }}
          className="flex-1 text-xs h-8"
        />
        <Button
          type="button"
          variant="ghost"
          size="icon"
          className="h-8 w-8"
          onClick={() => loadDirectory(currentPath)}
          title="Refresh"
        >
          <RefreshCw className={cn("w-4 h-4", isLoading && "animate-spin")} />
        </Button>
      </div>

      {/* Navigation buttons */}
      <div className="flex items-center gap-1">
        <Button
          type="button"
          variant="ghost"
          size="sm"
          className="h-7 px-2"
          onClick={loadHome}
          title="Home"
        >
          <Home className="w-4 h-4" />
        </Button>
        <Button
          type="button"
          variant="ghost"
          size="sm"
          className="h-7 px-2"
          onClick={() => parentPath && loadDirectory(parentPath)}
          disabled={!parentPath}
          title="Go up"
        >
          <ChevronUp className="w-4 h-4" />
        </Button>
        <span className="text-xs text-muted-foreground ml-2 truncate flex-1">
          {currentPath}
        </span>
      </div>

      {/* Directory listing */}
      <ScrollArea className="h-48 border rounded-md">
        {error ? (
          <div className="p-4 text-center text-sm text-red-500">
            {error}
          </div>
        ) : entries.length === 0 ? (
          <div className="p-4 text-center text-sm text-muted-foreground">
            {isLoading ? 'Loading...' : 'Empty folder'}
          </div>
        ) : (
          <div className="p-1">
            {entries.map((entry) => (
              <button
                key={entry.path}
                type="button"
                className={cn(
                  "w-full flex items-center gap-2 px-2 py-1.5 rounded text-sm text-left hover:bg-accent transition-colors",
                  entry.is_dir && "cursor-pointer"
                )}
                onClick={() => handleEntryClick(entry)}
                onDoubleClick={() => {
                  if (entry.is_dir) {
                    loadDirectory(entry.path);
                  }
                }}
              >
                <Folder className="w-4 h-4 text-muted-foreground flex-shrink-0" />
                <span className="truncate">{entry.name}</span>
              </button>
            ))}
          </div>
        )}
      </ScrollArea>

      {/* Action buttons */}
      <div className="flex justify-end gap-2">
        <Button
          type="button"
          variant="outline"
          size="sm"
          onClick={handleCancel}
        >
          <X className="w-4 h-4 mr-1" />
          Cancel
        </Button>
        <Button
          type="button"
          size="sm"
          onClick={handleSelect}
        >
          <Check className="w-4 h-4 mr-1" />
          Select
        </Button>
      </div>
    </div>
  );
}
