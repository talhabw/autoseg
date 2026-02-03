import { useState, useCallback, useRef } from 'react';
import { useAnnotationStore } from '../../stores/annotationStore';
import { Button } from "@/components/ui/button";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
  CommandSeparator,
} from "@/components/ui/command";
import { Plus, Check, ChevronsUpDown } from "lucide-react";
import { cn } from "@/lib/utils";
import type { Label } from '../../types';

// Color swatch component for editing label colors
function ColorSwatch({ label, onColorChange }: { label: Label; onColorChange: (labelId: number, color: string) => void }) {
  const inputRef = useRef<HTMLInputElement>(null);

  const handleClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    inputRef.current?.click();
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    onColorChange(label.id, e.target.value);
  };

  return (
    <div className="relative">
      <button
        type="button"
        onClick={handleClick}
        className="w-3 h-3 rounded-full flex-shrink-0 mr-2 cursor-pointer hover:ring-2 hover:ring-offset-1 hover:ring-primary transition-all"
        style={{ backgroundColor: label.color }}
        title="Click to change color"
      />
      <input
        ref={inputRef}
        type="color"
        value={label.color}
        onChange={handleChange}
        className="absolute opacity-0 w-0 h-0"
        onClick={(e) => e.stopPropagation()}
      />
    </div>
  );
}

export function LabelPicker() {
  const { labels, selectedLabelId, selectLabel, createLabel, updateLabel } = useAnnotationStore();
  const [open, setOpen] = useState(false);
  const [searchValue, setSearchValue] = useState("");
  const [isCreating, setIsCreating] = useState(false);

  const selectedLabel = labels.find((l) => l.id === selectedLabelId);

  // Check if there's an exact match (case-insensitive)
  const hasExactMatch = labels.some(
    (l) => l.name.toLowerCase() === searchValue.trim().toLowerCase()
  );

  const handleCreateLabel = useCallback(async () => {
    if (!searchValue.trim() || isCreating) return;

    setIsCreating(true);
    try {
      await createLabel(searchValue.trim());
      setSearchValue("");
      setOpen(false);  // Close the popover after creating
    } catch (err) {
      console.error('Failed to create label:', err);
    } finally {
      setIsCreating(false);
    }
  }, [searchValue, isCreating, createLabel]);

  const handleColorChange = useCallback(async (labelId: number, color: string) => {
    try {
      await updateLabel(labelId, { color });
    } catch (err) {
      console.error('Failed to update label color:', err);
    }
  }, [updateLabel]);

  // Handle key down on the command input
  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    // Create label on Enter if there's search text and no exact match
    if (e.key === 'Enter' && searchValue.trim() && !hasExactMatch) {
      e.preventDefault();
      e.stopPropagation();
      handleCreateLabel();
    }
  }, [searchValue, hasExactMatch, handleCreateLabel]);

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button
          variant="outline"
          role="combobox"
          aria-expanded={open}
          className="w-full justify-between"
        >
          {selectedLabel ? (
            <div className="flex items-center gap-2 truncate">
              <span
                className="w-3 h-3 rounded-full flex-shrink-0"
                style={{ backgroundColor: selectedLabel.color }}
              />
              {selectedLabel.name}
            </div>
          ) : (
            "Select label..."
          )}
          <ChevronsUpDown className="ml-2 h-4 w-4 shrink-0 opacity-50" />
        </Button>
      </PopoverTrigger>
      <PopoverContent className="w-[var(--radix-popover-trigger-width)] p-0">
        <Command>
          <CommandInput
            placeholder="Search or create label..."
            value={searchValue}
            onValueChange={setSearchValue}
            onKeyDown={handleKeyDown}
          />
          <CommandList>
            <CommandEmpty>
              <div className="p-2 text-center text-sm">
                <p className="text-muted-foreground mb-2">No label found.</p>
                <Button
                  size="sm"
                  className="w-full"
                  onClick={handleCreateLabel}
                  disabled={isCreating}
                >
                  <Plus className="mr-2 h-4 w-4" />
                  {isCreating ? 'Creating...' : `Create "${searchValue}"`}
                </Button>
                <p className="text-muted-foreground text-xs mt-2">
                  Press Enter to create
                </p>
              </div>
            </CommandEmpty>
            <CommandGroup heading="Labels">
              {labels.map((label) => (
                <CommandItem
                  key={label.id}
                  value={label.name}
                  onSelect={() => {
                    selectLabel(label.id);
                    setOpen(false);
                  }}
                >
                  <Check
                    className={cn(
                      "mr-2 h-4 w-4",
                      selectedLabelId === label.id ? "opacity-100" : "opacity-0"
                    )}
                  />
                  <ColorSwatch label={label} onColorChange={handleColorChange} />
                  {label.name}
                </CommandItem>
              ))}
            </CommandGroup>
            {/* Always show create option when there's search text and no exact match */}
            {searchValue.trim() && !hasExactMatch && labels.length > 0 && (
              <>
                <CommandSeparator />
                <CommandGroup>
                  <CommandItem
                    value={`__create_${searchValue}`}
                    onSelect={() => handleCreateLabel()}
                    className="text-primary"
                  >
                    <Plus className="mr-2 h-4 w-4" />
                    Create "{searchValue.trim()}"
                    <span className="ml-auto text-xs text-muted-foreground">Enter</span>
                  </CommandItem>
                </CommandGroup>
              </>
            )}
          </CommandList>
        </Command>
      </PopoverContent>
    </Popover>
  );
}
