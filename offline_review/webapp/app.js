(function () {
  const data = window.REVIEW_DATA;
  const INITIAL_PRELOAD_COUNT = 100;
  if (!data || !Array.isArray(data.images)) {
    document.body.innerHTML = '<main style="padding:24px;color:#fff;background:#0b1020;font-family:system-ui">Missing review manifest.</main>';
    return;
  }

  const elements = {
    appShell: document.getElementById('app-shell'),
    datasetName: document.getElementById('dataset-name'),
    preloadStatus: document.getElementById('preload-status'),
    loadingOverlay: document.getElementById('loading-overlay'),
    loadingProgress: document.getElementById('loading-progress'),
    activeImage: document.getElementById('active-image'),
    focusExitButton: document.getElementById('focus-exit-button'),
    imagePath: document.getElementById('image-path'),
    labelSummary: document.getElementById('label-summary'),
    positionSummary: document.getElementById('position-summary'),
    markedCount: document.getElementById('marked-count'),
    markedList: document.getElementById('marked-list'),
    emptyState: document.getElementById('empty-state'),
    prevButton: document.getElementById('prev-button'),
    nextButton: document.getElementById('next-button'),
    markButton: document.getElementById('mark-button'),
    focusButton: document.getElementById('focus-button'),
    copyButton: document.getElementById('copy-button'),
    clearButton: document.getElementById('clear-button'),
  };

  const state = {
    currentIndex: 0,
    cache: new Map(),
    markedIds: new Set(),
    imageFocusMode: false,
    ready: false,
  };

  const imageById = new Map(data.images.map((item) => [item.id, item]));
  const storageKey = `offline-review:${data.datasetKey}`;

  function assetUrl(path) {
    return encodeURI(path);
  }

  function loadStoredMarkedIds() {
    try {
      const rawValue = localStorage.getItem(storageKey);
      const parsed = rawValue ? JSON.parse(rawValue) : [];
      if (!Array.isArray(parsed)) {
        return new Set();
      }
      const filtered = parsed.filter((id) => imageById.has(id));
      if (filtered.length !== parsed.length) {
        localStorage.setItem(storageKey, JSON.stringify(filtered));
      }
      return new Set(filtered);
    } catch (error) {
      console.warn('Failed to read stored removal list:', error);
      return new Set();
    }
  }

  function saveMarkedIds() {
    localStorage.setItem(storageKey, JSON.stringify(Array.from(state.markedIds)));
  }

  function markedItems() {
    return data.images.filter((item) => state.markedIds.has(item.id));
  }

  function currentItem() {
    return data.images[state.currentIndex] || null;
  }

  function renderCurrentImage() {
    const item = currentItem();
    if (!item) {
      return;
    }

    const cachedImage = state.cache.get(item.id);
    if (cachedImage) {
      elements.activeImage.src = cachedImage.src;
    } else {
      elements.activeImage.src = assetUrl(item.previewPath);
    }

    elements.imagePath.textContent = item.sourcePath;
    const kindSummary = item.annotationKinds && item.annotationKinds.length > 0
      ? item.annotationKinds.join(', ')
      : 'no annotations';
    const classSummary = item.classLabels && item.classLabels.length > 0
      ? item.classLabels.join(', ')
      : 'no class labels';
    elements.labelSummary.textContent = `${item.annotationCount} annotations • ${kindSummary} • ${classSummary}`;
    elements.positionSummary.textContent = `${state.currentIndex + 1} / ${data.images.length}`;
    const isMarked = state.markedIds.has(item.id);
    elements.markButton.textContent = isMarked ? 'Unmark removal' : 'Mark for removal';
  }

  function renderMarkedList() {
    const items = markedItems();
    elements.markedCount.textContent = `${items.length} marked`;
    elements.emptyState.classList.toggle('is-hidden', items.length > 0);
    elements.markedList.innerHTML = '';

    for (const item of items) {
      const listItem = document.createElement('li');
      listItem.className = 'marked-item';
      if (currentItem() && currentItem().id === item.id) {
        listItem.classList.add('is-active');
      }

      const jumpButton = document.createElement('button');
      jumpButton.type = 'button';
      jumpButton.className = 'marked-jump';
      jumpButton.innerHTML = `<strong>${item.sourcePath.split('/').pop()}</strong><span>${item.sourcePath}</span>`;
      jumpButton.addEventListener('click', () => {
        state.currentIndex = item.index;
        render();
      });

      const unmarkButton = document.createElement('button');
      unmarkButton.type = 'button';
      unmarkButton.className = 'unmark-button';
      unmarkButton.textContent = 'Unmark';
      unmarkButton.addEventListener('click', () => {
        state.markedIds.delete(item.id);
        saveMarkedIds();
        render();
      });

      listItem.appendChild(jumpButton);
      listItem.appendChild(unmarkButton);
      elements.markedList.appendChild(listItem);
    }
  }

  function render() {
    renderCurrentImage();
    renderMarkedList();
  }

  function updateFocusButton() {
    elements.focusButton.textContent = state.imageFocusMode ? 'Exit focus' : 'Focus image';
  }

  function setImageFocusMode(enabled) {
    state.imageFocusMode = enabled;
    elements.appShell.classList.toggle('image-focus-mode', enabled);
    updateFocusButton();
  }

  function toggleImageFocusMode() {
    setImageFocusMode(!state.imageFocusMode);
  }

  function move(delta) {
    if (!state.ready || data.images.length === 0) {
      return;
    }
    state.currentIndex = (state.currentIndex + delta + data.images.length) % data.images.length;
    render();
  }

  function toggleCurrentMarked() {
    const item = currentItem();
    if (!item || !state.ready) {
      return;
    }

    if (state.markedIds.has(item.id)) {
      state.markedIds.delete(item.id);
    } else {
      state.markedIds.add(item.id);
    }
    saveMarkedIds();
    render();
  }

  async function copyMarkedPaths() {
    const text = markedItems().map((item) => item.sourcePath).join('\n');
    if (!text) {
      window.alert('No images are marked for removal.');
      return;
    }

    try {
      await navigator.clipboard.writeText(text);
    } catch (error) {
      const helper = document.createElement('textarea');
      helper.value = text;
      helper.style.position = 'fixed';
      helper.style.opacity = '0';
      document.body.appendChild(helper);
      helper.select();
      document.execCommand('copy');
      helper.remove();
    }
    elements.preloadStatus.textContent = `Copied ${markedItems().length} marked paths`;
  }

  function clearMarkedPaths() {
    if (state.markedIds.size === 0) {
      return;
    }
    state.markedIds.clear();
    saveMarkedIds();
    render();
  }

  function handleKeydown(event) {
    if (!state.ready) {
      return;
    }

    const tagName = document.activeElement && document.activeElement.tagName;
    if (tagName === 'INPUT' || tagName === 'TEXTAREA') {
      return;
    }

    switch (event.key) {
      case 'ArrowLeft':
      case 'a':
      case 'A':
        event.preventDefault();
        move(-1);
        break;
      case 'ArrowRight':
      case 'd':
      case 'D':
        event.preventDefault();
        move(1);
        break;
      case 'q':
      case 'Q':
        event.preventDefault();
        toggleCurrentMarked();
        break;
      case 'Escape':
        if (state.imageFocusMode) {
          event.preventDefault();
          setImageFocusMode(false);
        }
        break;
      default:
        break;
    }
  }

  function preloadImages(items, onProgress) {
    const total = items.length;
    let completed = 0;

    const tasks = items.map((item) => new Promise((resolve) => {
      const image = new Image();
      image.onload = () => {
        state.cache.set(item.id, image);
        completed += 1;
        if (onProgress) {
          onProgress(completed, total);
        }
        resolve();
      };
      image.onerror = () => {
        completed += 1;
        if (onProgress) {
          onProgress(completed, total);
        }
        resolve();
      };
      image.src = assetUrl(item.previewPath);
    }));

    return Promise.all(tasks);
  }

  function bindEvents() {
    elements.prevButton.addEventListener('click', () => move(-1));
    elements.nextButton.addEventListener('click', () => move(1));
    elements.markButton.addEventListener('click', toggleCurrentMarked);
    elements.focusButton.addEventListener('click', toggleImageFocusMode);
    elements.focusExitButton.addEventListener('click', () => setImageFocusMode(false));
    elements.copyButton.addEventListener('click', copyMarkedPaths);
    elements.clearButton.addEventListener('click', clearMarkedPaths);
    window.addEventListener('keydown', handleKeydown);
  }

  async function init() {
    elements.datasetName.textContent = data.datasetName;
    state.markedIds = loadStoredMarkedIds();
    bindEvents();
    updateFocusButton();
    renderMarkedList();

    const initialItems = data.images.slice(0, INITIAL_PRELOAD_COUNT);
    const remainingItems = data.images.slice(initialItems.length);

    elements.loadingProgress.textContent = `0 / ${initialItems.length}`;
    await preloadImages(initialItems, (completed, total) => {
      elements.loadingProgress.textContent = `${completed} / ${total}`;
    });

    state.ready = true;
    elements.loadingOverlay.classList.add('is-hidden');
    if (remainingItems.length === 0) {
      elements.preloadStatus.textContent = `Loaded ${data.images.length} preview images`;
    } else {
      elements.preloadStatus.textContent = `Loaded ${initialItems.length}/${data.images.length}, preloading rest...`;
    }
    render();

    if (remainingItems.length > 0) {
      void preloadImages(remainingItems, (completed, total) => {
        const loadedCount = initialItems.length + completed;
        if (completed === total) {
          elements.preloadStatus.textContent = `Loaded ${data.images.length} preview images`;
        } else {
          elements.preloadStatus.textContent = `Loaded ${loadedCount}/${data.images.length}, preloading rest...`;
        }
      });
    }
  }

  init();
})();
