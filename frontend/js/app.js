/**
 * Lens // Caption — Minimalist Frontend Logic
 * Vanilla JavaScript implementation for neural image captioning.
 */

(function () {
  'use strict';

  // --- State ---
  const state = {
    selectedFile: null,
    selectedSample: null,
    previewUrl: null,
    isProcessing: false,
    history: [],
    samples: [],
    modelMeta: null,
  };

  // --- DOM Elements ---
  const dom = {
    html: document.documentElement,
    themeToggle: document.getElementById('themeToggle'),
    statusIndicator: document.getElementById('statusIndicator'),
    statusLabel: document.getElementById('statusLabel'),
    architectureBtn: document.getElementById('architectureBtn'),
    specsModal: document.getElementById('specsModal'),
    closeSpecsModal: document.getElementById('closeSpecsModal'),

    dropzone: document.getElementById('dropzone'),
    fileInput: document.getElementById('fileInput'),
    dropzoneEmpty: document.getElementById('dropzoneEmpty'),
    dropzonePreview: document.getElementById('dropzonePreview'),
    imagePreview: document.getElementById('imagePreview'),
    metaFilename: document.getElementById('metaFilename'),
    metaDimensions: document.getElementById('metaDimensions'),
    metaSize: document.getElementById('metaSize'),
    metaSource: document.getElementById('metaSource'),
    removeImageBtn: document.getElementById('removeImageBtn'),

    generateBtn: document.getElementById('generateBtn'),
    btnContent: document.querySelector('#generateBtn .btn-content'),
    btnSpinner: document.querySelector('#generateBtn .spinner'),

    sampleGrid: document.getElementById('sampleGrid'),

    resultPlaceholder: document.getElementById('resultPlaceholder'),
    resultLoading: document.getElementById('resultLoading'),
    resultActive: document.getElementById('resultActive'),
    captionText: document.getElementById('captionText'),
    latencyPill: document.getElementById('latencyPill'),
    latencyText: document.getElementById('latencyText'),
    copyCaptionBtn: document.getElementById('copyCaptionBtn'),
    copyBtnLabel: document.getElementById('copyBtnLabel'),
    speakCaptionBtn: document.getElementById('speakCaptionBtn'),

    tokensSection: document.getElementById('tokensSection'),
    tokensList: document.getElementById('tokensList'),

    statMaxLen: document.getElementById('statMaxLen'),
    statVocab: document.getElementById('statVocab'),

    historyList: document.getElementById('historyList'),
    historyCount: document.getElementById('historyCount'),
    clearHistoryBtn: document.getElementById('clearHistoryBtn'),
  };

  // -------------------------------------------------------------------------
  // 1. Initialization
  // -------------------------------------------------------------------------
  function init() {
    initTheme();
    setupEventListeners();
    fetchHealth();
    fetchSamples();
  }

  // -------------------------------------------------------------------------
  // 2. Theme Management
  // -------------------------------------------------------------------------
  function initTheme() {
    const savedTheme = localStorage.getItem('lens_theme') || 'dark';
    setTheme(savedTheme);

    dom.themeToggle.addEventListener('click', () => {
      const current = dom.html.getAttribute('data-theme');
      const next = current === 'dark' ? 'light' : 'dark';
      setTheme(next);
    });
  }

  function setTheme(theme) {
    dom.html.setAttribute('data-theme', theme);
    localStorage.setItem('lens_theme', theme);
  }

  // -------------------------------------------------------------------------
  // 3. API Health & Metadata
  // -------------------------------------------------------------------------
  async function fetchHealth() {
    try {
      const res = await fetch('/api/health');
      if (!res.ok) throw new Error('Health check failed');
      const data = await res.json();
      state.modelMeta = data;

      if (data.model_loaded) {
        dom.statusLabel.textContent = 'Model Ready';
        if (data.max_length) dom.statMaxLen.textContent = `${data.max_length} Tokens`;
        if (data.vocab_size) dom.statVocab.textContent = `${data.vocab_size.toLocaleString()} Tokens`;
      } else {
        dom.statusLabel.textContent = 'Initializing...';
      }
    } catch (err) {
      dom.statusLabel.textContent = 'Offline';
      console.warn('API Health check warning:', err);
    }
  }

  // -------------------------------------------------------------------------
  // 4. Sample Images
  // -------------------------------------------------------------------------
  async function fetchSamples() {
    try {
      const res = await fetch('/api/samples');
      if (!res.ok) return;
      const data = await res.json();
      state.samples = data.samples || [];
      renderSamples();
    } catch (err) {
      console.warn('Could not load samples:', err);
    }
  }

  function renderSamples() {
    if (!state.samples.length) {
      dom.sampleGrid.innerHTML = '<p class="subheader-tip">No sample images found</p>';
      return;
    }

    dom.sampleGrid.innerHTML = state.samples
      .map(
        (s) => `
      <div class="sample-card" data-filename="${s.filename}" title="${s.title}: ${s.description}">
        <img class="sample-thumb" src="${s.url}" alt="${s.title}" loading="lazy">
        <div class="sample-info">
          <span class="sample-title">${s.title}</span>
          <span class="sample-tag">${s.category}</span>
        </div>
      </div>
    `
      )
      .join('');

    // Attach click listeners to sample cards
    dom.sampleGrid.querySelectorAll('.sample-card').forEach((card) => {
      card.addEventListener('click', () => {
        const filename = card.getAttribute('data-filename');
        selectSample(filename);
      });
    });
  }

  function selectSample(filename) {
    const sample = state.samples.find((s) => s.filename === filename);
    if (!sample) return;

    // Update active highlight
    dom.sampleGrid.querySelectorAll('.sample-card').forEach((c) => {
      c.classList.toggle('active', c.getAttribute('data-filename') === filename);
    });

    state.selectedFile = null;
    state.selectedSample = filename;
    state.previewUrl = sample.url;

    // Load preview
    dom.dropzoneEmpty.style.display = 'none';
    dom.dropzonePreview.style.display = 'flex';
    dom.imagePreview.src = sample.url;
    dom.metaFilename.textContent = sample.title;
    dom.metaSource.textContent = 'Sample';
    dom.metaSize.textContent = 'Preset';

    dom.imagePreview.onload = () => {
      dom.metaDimensions.textContent = `${dom.imagePreview.naturalWidth} × ${dom.imagePreview.naturalHeight}`;
    };

    dom.generateBtn.disabled = false;
    runInference();
  }

  // -------------------------------------------------------------------------
  // 5. File Upload & Clipboard Paste
  // -------------------------------------------------------------------------
  function setupEventListeners() {
    // Dropzone click
    dom.dropzone.addEventListener('click', (e) => {
      if (e.target.closest('#removeImageBtn')) return;
      if (!state.previewUrl) {
        dom.fileInput.click();
      }
    });

    // File input change
    dom.fileInput.addEventListener('change', (e) => {
      const file = e.target.files[0];
      if (file) handleImageFile(file);
    });

    // Drag and drop
    dom.dropzone.addEventListener('dragover', (e) => {
      e.preventDefault();
      dom.dropzone.classList.add('dragover');
    });

    ['dragleave', 'dragend'].forEach((ev) => {
      dom.dropzone.addEventListener(ev, () => {
        dom.dropzone.classList.remove('dragover');
      });
    });

    dom.dropzone.addEventListener('drop', (e) => {
      e.preventDefault();
      dom.dropzone.classList.remove('dragover');
      const file = e.dataTransfer.files[0];
      if (file && file.type.startsWith('image/')) {
        handleImageFile(file);
      }
    });

    // Clipboard Paste (Ctrl+V) anywhere
    window.addEventListener('paste', (e) => {
      const items = e.clipboardData?.items;
      if (!items) return;
      for (const item of items) {
        if (item.type.startsWith('image/')) {
          const file = item.getAsFile();
          if (file) {
            handleImageFile(file);
            break;
          }
        }
      }
    });

    // Remove / change image
    dom.removeImageBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      resetImageSelection();
    });

    // Generate Button
    dom.generateBtn.addEventListener('click', () => {
      if (!state.isProcessing) runInference();
    });

    // Copy caption
    dom.copyCaptionBtn.addEventListener('click', copyCaptionToClipboard);

    // Speak caption (Text-to-Speech)
    dom.speakCaptionBtn.addEventListener('click', speakCaption);

    // Specs Modal
    dom.architectureBtn.addEventListener('click', () => {
      dom.specsModal.style.display = 'flex';
    });
    dom.closeSpecsModal.addEventListener('click', () => {
      dom.specsModal.style.display = 'none';
    });
    dom.specsModal.addEventListener('click', (e) => {
      if (e.target === dom.specsModal) dom.specsModal.style.display = 'none';
    });

    // Clear history
    dom.clearHistoryBtn.addEventListener('click', clearHistory);

    // Keyboard shortcuts
    window.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') {
        dom.specsModal.style.display = 'none';
      }
    });
  }

  function handleImageFile(file) {
    if (!file || !file.type.startsWith('image/')) return;

    state.selectedFile = file;
    state.selectedSample = null;

    // Deselect sample cards
    dom.sampleGrid.querySelectorAll('.sample-card').forEach((c) => c.classList.remove('active'));

    const reader = new FileReader();
    reader.onload = (e) => {
      state.previewUrl = e.target.result;
      dom.dropzoneEmpty.style.display = 'none';
      dom.dropzonePreview.style.display = 'flex';
      dom.imagePreview.src = state.previewUrl;
      dom.metaFilename.textContent = file.name;
      dom.metaSource.textContent = 'Upload';
      dom.metaSize.textContent = `${(file.size / 1024).toFixed(1)} KB`;

      dom.imagePreview.onload = () => {
        dom.metaDimensions.textContent = `${dom.imagePreview.naturalWidth} × ${dom.imagePreview.naturalHeight}`;
      };

      dom.generateBtn.disabled = false;
    };
    reader.readAsDataURL(file);
  }

  function resetImageSelection() {
    state.selectedFile = null;
    state.selectedSample = null;
    state.previewUrl = null;
    dom.fileInput.value = '';

    dom.dropzonePreview.style.display = 'none';
    dom.dropzoneEmpty.style.display = 'flex';
    dom.imagePreview.src = '';
    dom.generateBtn.disabled = true;

    dom.sampleGrid.querySelectorAll('.sample-card').forEach((c) => c.classList.remove('active'));
  }

  // -------------------------------------------------------------------------
  // 6. Inference Execution
  // -------------------------------------------------------------------------
  async function runInference() {
    if (state.isProcessing || (!state.selectedFile && !state.selectedSample)) return;

    setLoadingState(true);

    try {
      let data;
      if (state.selectedSample) {
        // Sample request
        const res = await fetch('/api/caption/sample', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ filename: state.selectedSample }),
        });
        if (!res.ok) {
          const errData = await res.json().catch(() => ({}));
          throw new Error(errData.detail || 'Inference error');
        }
        data = await res.json();
      } else if (state.selectedFile) {
        // File upload request
        const formData = new FormData();
        formData.append('file', state.selectedFile);

        const res = await fetch('/api/caption', {
          method: 'POST',
          body: formData,
        });
        if (!res.ok) {
          const errData = await res.json().catch(() => ({}));
          throw new Error(errData.detail || 'Inference error');
        }
        data = await res.json();
      }

      displayResult(data);
      recordHistory(data);
    } catch (err) {
      alert(`Captioning failed: ${err.message}`);
      dom.resultLoading.style.display = 'none';
      dom.resultPlaceholder.style.display = 'flex';
    } finally {
      setLoadingState(false);
    }
  }

  function setLoadingState(loading) {
    state.isProcessing = loading;
    dom.generateBtn.disabled = loading;

    if (loading) {
      dom.btnContent.style.display = 'none';
      dom.btnSpinner.style.display = 'inline-block';

      dom.resultPlaceholder.style.display = 'none';
      dom.resultActive.style.display = 'none';
      dom.tokensSection.style.display = 'none';
      dom.resultLoading.style.display = 'flex';
      dom.latencyPill.style.display = 'none';
    } else {
      dom.btnContent.style.display = 'inline-flex';
      dom.btnSpinner.style.display = 'none';
      dom.resultLoading.style.display = 'none';
    }
  }

  // -------------------------------------------------------------------------
  // 7. Results Presentation
  // -------------------------------------------------------------------------
  function displayResult(result) {
    dom.resultPlaceholder.style.display = 'none';
    dom.resultActive.style.display = 'flex';

    // Typewriter effect for caption
    animateCaption(result.caption);

    // Latency
    if (result.latency_ms) {
      dom.latencyText.textContent = `${result.latency_ms} ms`;
      dom.latencyPill.style.display = 'inline-flex';
    }

    // Tokens Breakdown
    if (result.tokens && result.tokens.length > 0) {
      dom.tokensSection.style.display = 'flex';
      dom.tokensList.innerHTML = result.tokens
        .map(
          (t) => `
        <div class="token-chip" title="Step ${t.step}: ${t.confidence}% probability">
          <span class="token-word">${t.token}</span>
          <span class="token-prob">${t.confidence}%</span>
        </div>
      `
        )
        .join('');
    } else {
      dom.tokensSection.style.display = 'none';
    }
  }

  function animateCaption(text) {
    dom.captionText.textContent = '';
    const words = text.split(' ');
    let i = 0;

    function step() {
      if (i < words.length) {
        dom.captionText.textContent += (i === 0 ? '' : ' ') + words[i];
        i++;
        setTimeout(step, 40);
      }
    }
    step();
  }

  function copyCaptionToClipboard() {
    const text = dom.captionText.textContent;
    if (!text) return;

    navigator.clipboard.writeText(text).then(() => {
      dom.copyBtnLabel.textContent = 'Copied!';
      setTimeout(() => {
        dom.copyBtnLabel.textContent = 'Copy';
      }, 1800);
    });
  }

  function speakCaption() {
    const text = dom.captionText.textContent;
    if (!text || !('speechSynthesis' in window)) return;

    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 0.95;
    utterance.pitch = 1.0;
    window.speechSynthesis.speak(utterance);
  }

  // -------------------------------------------------------------------------
  // 8. Session History Tracking
  // -------------------------------------------------------------------------
  function recordHistory(result) {
    const entry = {
      id: Date.now(),
      thumb: state.previewUrl,
      caption: result.caption,
      latency: result.latency_ms,
      tokens: result.tokens,
      time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    };

    state.history.unshift(entry);
    renderHistory();
  }

  function renderHistory() {
    dom.historyCount.textContent = state.history.length;
    dom.clearHistoryBtn.style.display = state.history.length > 0 ? 'inline-block' : 'none';

    if (state.history.length === 0) {
      dom.historyList.innerHTML = '<div class="history-empty">No captions generated in this session yet.</div>';
      return;
    }

    dom.historyList.innerHTML = state.history
      .map(
        (item) => `
      <div class="history-item" data-id="${item.id}">
        <img class="history-item-thumb" src="${item.thumb}" alt="Past run thumbnail">
        <div class="history-item-content">
          <span class="history-item-caption">${item.caption}</span>
          <div class="history-item-meta">
            <span>${item.time}</span>
            <span>•</span>
            <span>${item.latency} ms</span>
            <span>•</span>
            <span>${item.tokens?.length || 0} tokens</span>
          </div>
        </div>
      </div>
    `
      )
      .join('');

    dom.historyList.querySelectorAll('.history-item').forEach((itemEl) => {
      itemEl.addEventListener('click', () => {
        const id = parseInt(itemEl.getAttribute('data-id'), 10);
        const item = state.history.find((h) => h.id === id);
        if (item) {
          dom.imagePreview.src = item.thumb;
          state.previewUrl = item.thumb;
          dom.dropzoneEmpty.style.display = 'none';
          dom.dropzonePreview.style.display = 'flex';
          dom.metaFilename.textContent = 'History Item';
          dom.metaSource.textContent = 'Session Cache';
          displayResult(item);
        }
      });
    });
  }

  function clearHistory() {
    state.history = [];
    renderHistory();
  }

  // Start on DOM ready
  document.addEventListener('DOMContentLoaded', init);
})();
