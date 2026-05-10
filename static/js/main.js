document.addEventListener('DOMContentLoaded', () => {
  const dropZone = document.getElementById('drop-zone');
  const fileInput = document.getElementById('resume-input');
  const filePreview = document.getElementById('file-preview');
  const filePreviewName = document.getElementById('file-preview-name');
  const removeFileBtn = document.getElementById('remove-file');
  const submitBtn = document.getElementById('submit-btn');
  const form = document.getElementById('analyze-form');

  if (!dropZone) return;

  // Click to browse
  dropZone.addEventListener('click', () => fileInput.click());

  // Drag & drop
  dropZone.addEventListener('dragover', e => {
    e.preventDefault();
    dropZone.classList.add('drag-over');
    dropZone.style.borderColor = 'var(--accent)';
    dropZone.style.background = 'var(--accent-dim)';
  });

  dropZone.addEventListener('dragleave', () => resetDropZone());

  dropZone.addEventListener('drop', e => {
    e.preventDefault();
    resetDropZone();
    const file = e.dataTransfer.files[0];
    if (file) setFile(file);
  });

  fileInput.addEventListener('change', () => {
    if (fileInput.files[0]) setFile(fileInput.files[0]);
  });

  removeFileBtn && removeFileBtn.addEventListener('click', () => {
    fileInput.value = '';
    filePreview.classList.remove('visible');
    dropZone.style.display = '';
  });

  form && form.addEventListener('submit', () => {
    if (submitBtn) {
      submitBtn.textContent = 'Analyzing…';
      submitBtn.disabled = true;
    }
  });

  function setFile(file) {
    const allowed = ['application/pdf', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'text/plain'];
    const ext = file.name.split('.').pop().toLowerCase();
    if (!['pdf', 'docx', 'txt'].includes(ext)) {
      alert('Please upload a PDF, DOCX, or TXT file.');
      return;
    }
    const dt = new DataTransfer();
    dt.items.add(file);
    fileInput.files = dt.files;

    filePreviewName.textContent = file.name;
    filePreview.classList.add('visible');
    dropZone.style.display = 'none';
  }

  function resetDropZone() {
    dropZone.classList.remove('drag-over');
    dropZone.style.borderColor = '';
    dropZone.style.background = '';
  }
});
