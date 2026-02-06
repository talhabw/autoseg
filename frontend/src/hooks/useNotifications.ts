import { useCallback, useRef } from 'react';

/**
 * Hook for managing browser notifications.
 * Used to notify the user when propagation stops due to failures.
 */
export function useNotifications() {
  const permissionRef = useRef<NotificationPermission>(
    typeof window !== 'undefined' && 'Notification' in window
      ? Notification.permission
      : 'default'
  );

  /**
   * Request notification permission from the user.
   * Should be called when propagation starts.
   * @returns true if permission was granted
   */
  const requestPermission = useCallback(async (): Promise<boolean> => {
    if (typeof window === 'undefined' || !('Notification' in window)) {
      console.warn('Browser does not support notifications');
      return false;
    }

    if (Notification.permission === 'granted') {
      permissionRef.current = 'granted';
      return true;
    }

    if (Notification.permission === 'denied') {
      permissionRef.current = 'denied';
      return false;
    }

    try {
      const permission = await Notification.requestPermission();
      permissionRef.current = permission;
      return permission === 'granted';
    } catch (err) {
      console.error('Failed to request notification permission:', err);
      return false;
    }
  }, []);

  /**
   * Show a browser notification.
   * @param title The notification title
   * @param options Standard notification options
   * @returns The Notification object or null if not supported/permitted
   */
  const notify = useCallback((title: string, options?: NotificationOptions): Notification | null => {
    if (typeof window === 'undefined' || !('Notification' in window)) {
      return null;
    }

    if (Notification.permission !== 'granted') {
      console.warn('Notification permission not granted');
      return null;
    }

    try {
      const notification = new Notification(title, {
        icon: '/favicon.ico',
        ...options,
      });

      // Auto-close after 10 seconds
      setTimeout(() => notification.close(), 10000);

      return notification;
    } catch (err) {
      console.error('Failed to show notification:', err);
      return null;
    }
  }, []);

  /**
   * Show a notification specifically for propagation failures.
   * @param failedLabels Array of label names that failed to propagate
   */
  const notifyPropagationFailure = useCallback((failedLabels: string[]) => {
    const labelList = failedLabels.slice(0, 3).join(', ');
    const extra = failedLabels.length > 3 ? ` and ${failedLabels.length - 3} more` : '';
    
    notify('Propagation Stopped', {
      body: `Failed to track: ${labelList}${extra}. Manual annotation required.`,
      tag: 'propagation-failure',
      requireInteraction: true, // Keep visible until dismissed
    });
  }, [notify]);

  return {
    requestPermission,
    notify,
    notifyPropagationFailure,
    isSupported: typeof window !== 'undefined' && 'Notification' in window,
    getPermission: () => permissionRef.current,
  };
}
