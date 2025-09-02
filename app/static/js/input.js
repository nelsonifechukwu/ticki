// DOM Elements
const imgInput = document.getElementById("input-img-preview");
const imgInputLabel = document.getElementById("input-img-label");
const imgContainer = document.querySelector(".input-img-container");
const inputFormElement = document.querySelector(".img-input-form");
const uploadArea = document.querySelector(".upload-area");

// Toast notification system
const showToast = (message, type = 'info') => {
  const toast = document.createElement('div');
  toast.className = `toast toast-${type}`;
  toast.innerHTML = `
    <i class="fas ${getToastIcon(type)}"></i>
    <span>${message}</span>
  `;
  
  document.body.appendChild(toast);
  
  // Trigger animation
  setTimeout(() => toast.classList.add('show'), 100);
  
  // Remove after 4 seconds
  setTimeout(() => {
    toast.classList.remove('show');
    setTimeout(() => document.body.removeChild(toast), 300);
  }, 4000);
};

const getToastIcon = (type) => {
  const icons = {
    success: 'fa-check-circle',
    error: 'fa-exclamation-circle',
    warning: 'fa-exclamation-triangle',
    info: 'fa-info-circle'
  };
  return icons[type] || icons.info;
};

// Add toast CSS dynamically
const addToastStyles = () => {
  if (document.querySelector('#toast-styles')) return;
  
  const style = document.createElement('style');
  style.id = 'toast-styles';
  style.textContent = `
    .toast {
      position: fixed;
      top: 20px;
      right: 20px;
      background: var(--bg-primary);
      color: var(--text-primary);
      padding: 16px 20px;
      border-radius: var(--border-radius-small);
      box-shadow: var(--shadow-heavy);
      display: flex;
      align-items: center;
      gap: 12px;
      transform: translateX(400px);
      transition: var(--transition);
      z-index: 1000;
      min-width: 280px;
      border-left: 4px solid var(--primary-color);
    }
    
    .toast.show {
      transform: translateX(0);
    }
    
    .toast-success { border-left-color: var(--success-color); }
    .toast-error { border-left-color: var(--danger-color); }
    .toast-warning { border-left-color: var(--warning-color); }
    .toast-info { border-left-color: var(--primary-color); }
    
    .toast i {
      font-size: 18px;
    }
    
    .toast-success i { color: var(--success-color); }
    .toast-error i { color: var(--danger-color); }
    .toast-warning i { color: var(--warning-color); }
    .toast-info i { color: var(--primary-color); }
  `;
  document.head.appendChild(style);
};

// Initialize toast system
addToastStyles();

// Image preview handler
const handleImagePreview = () => {
  const file = imgInput.files[0];
  if (!file) return;

  // Validate file type
  if (!file.type.startsWith('image/')) {
    showToast('Please select a valid image file', 'error');
    return;
  }

  // Validate file size (10MB limit)
  if (file.size > 10 * 1024 * 1024) {
    showToast('File size must be less than 10MB', 'error');
    return;
  }

  const reader = new FileReader();
  reader.readAsDataURL(file);

  reader.onload = () => {
    // Update upload area
    uploadArea.classList.add('has-image');
    uploadArea.innerHTML = `<img src="${reader.result}" alt="Preview" class="preview-image" />`;
    
    showToast('Image loaded successfully!', 'success');
  };

  reader.onerror = () => {
    showToast('Error reading file. Please try again.', 'error');
  };
};

// Drag and drop functionality
const setupDragAndDrop = () => {
  let dragCounter = 0;

  const handleDragEnter = (e) => {
    e.preventDefault();
    dragCounter++;
    uploadArea.classList.add('drag-over');
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    dragCounter--;
    if (dragCounter === 0) {
      uploadArea.classList.remove('drag-over');
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    dragCounter = 0;
    uploadArea.classList.remove('drag-over');

    const files = e.dataTransfer.files;
    if (files.length > 0) {
      imgInput.files = files;
      handleImagePreview();
    }
  };

  // Add drag and drop styles
  const dragStyles = `
    .upload-area.drag-over {
      border-color: var(--success-color) !important;
      background: rgba(17, 153, 142, 0.1) !important;
      transform: scale(1.02);
    }
  `;

  if (!document.querySelector('#drag-styles')) {
    const style = document.createElement('style');
    style.id = 'drag-styles';
    style.textContent = dragStyles;
    document.head.appendChild(style);
  }

  uploadArea.addEventListener('dragenter', handleDragEnter);
  uploadArea.addEventListener('dragleave', handleDragLeave);
  uploadArea.addEventListener('dragover', handleDragOver);
  uploadArea.addEventListener('drop', handleDrop);
};

// Loading state management
const setLoadingState = (button, isLoading, originalText) => {
  if (isLoading) {
    button.disabled = true;
    button.classList.add('btn-loading');
    button.innerHTML = `<span>Processing...</span>`;
  } else {
    button.disabled = false;
    button.classList.remove('btn-loading');
    button.innerHTML = originalText;
  }
};

// Form submission handler
const handleFormSubmit = async (ev) => {
  ev.preventDefault();

  const qImgInput = document.getElementById("q-img");
  const submitButton = ev.target.querySelector('button[type="submit"]');
  const originalButtonText = submitButton.innerHTML;

  // Validation
  if (!imgInput.files || imgInput.files.length === 0) {
    const requiredElement = document.querySelector(".is-required");
    requiredElement.style.display = "block";
    showToast('Please select an image first', 'warning');
    return;
  }

  // Set loading state
  setLoadingState(submitButton, true, originalButtonText);

  try {
    // Transfer file to hidden input
    qImgInput.files = imgInput.files;
    const form = ev.target;
    const formData = new FormData(form);

    const response = await fetch(form.action, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const htmlText = await response.text();
    const parsedDoc = new DOMParser().parseFromString(htmlText, "text/html");

    // Update results section
    const newSelectedImgs = parsedDoc.querySelector(".selected-imgs");
    const currentSelectedImgs = document.querySelector(".selected-imgs");

    if (newSelectedImgs && currentSelectedImgs) {
      currentSelectedImgs.innerHTML = newSelectedImgs.innerHTML;
      currentSelectedImgs.classList.add('fade-in');
    }

    // Handle multiple faces
    const currentMultipleFacesContainer = document.querySelector(".multiple-faces-container");
    const newMultipleInputFaces = parsedDoc.querySelector(".multiple-faces-container .multiple-faces")?.querySelectorAll(".selectable-face");

    if (newMultipleInputFaces && newMultipleInputFaces.length > 0) {
      const newMultipleFacesContainer = parsedDoc.querySelector(".multiple-faces-container");
      currentMultipleFacesContainer.innerHTML = newMultipleFacesContainer.innerHTML;
      currentMultipleFacesContainer.style.display = "block";
      currentMultipleFacesContainer.classList.add('fade-in');

      // Re-bind event listener
      const multipleFacesForm = currentMultipleFacesContainer.querySelector("#multiple-faces-form");
      if (multipleFacesForm) {
        multipleFacesForm.addEventListener("submit", handleMultipleFacesSubmit);
      }

      showToast('Multiple faces detected! Please select which faces to search for.', 'info');
    } else {
      currentMultipleFacesContainer.style.display = "none";
      
      // Check if we have results
      const hasResults = newSelectedImgs && newSelectedImgs.children.length > 0;
      if (hasResults) {
        const resultsCount = newSelectedImgs.querySelectorAll('.result-item').length;
        showToast(`Found ${resultsCount} similar ${resultsCount === 1 ? 'image' : 'images'}!`, 'success');
      } else {
        showToast('No similar faces found in the repository.', 'info');
      }
    }

  } catch (error) {
    console.error("Form submission failed:", error);
    showToast('Something went wrong. Please try again.', 'error');
  } finally {
    setLoadingState(submitButton, false, originalButtonText);
  }
};

// Multiple faces form submission
const handleMultipleFacesSubmit = async (ev) => {
  ev.preventDefault();
  
  const form = ev.target;
  const submitButton = form.querySelector('button[type="submit"]');
  const originalButtonText = submitButton.innerHTML;

  // Validation
  const selectedFaces = form.querySelectorAll('input[name="selected_faces"]:checked');
  if (selectedFaces.length === 0) {
    showToast('Please select at least one face before submitting.', 'warning');
    return;
  }

  setLoadingState(submitButton, true, originalButtonText);

  try {
    const formData = new FormData(form);

    const response = await fetch(form.action, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const htmlText = await response.text();
    const parsedDoc = new DOMParser().parseFromString(htmlText, "text/html");

    // Update results
    const newSelectedImgs = parsedDoc.querySelector(".selected-imgs");
    const currentSelectedImgs = document.querySelector(".selected-imgs");
    const currentMultipleFacesContainer = document.querySelector(".multiple-faces-container");

    // Hide multiple faces container
    currentMultipleFacesContainer.style.display = "none";

    if (newSelectedImgs && currentSelectedImgs) {
      currentSelectedImgs.innerHTML = newSelectedImgs.innerHTML;
      currentSelectedImgs.classList.add('fade-in');

      const resultsCount = newSelectedImgs.querySelectorAll('.result-item').length;
      if (resultsCount > 0) {
        showToast(`Found ${resultsCount} similar ${resultsCount === 1 ? 'image' : 'images'}!`, 'success');
      } else {
        showToast('No similar faces found for the selected faces.', 'info');
      }
    }

  } catch (error) {
    console.error("Multiple faces form submission failed:", error);
    showToast('Something went wrong. Please try again.', 'error');
  } finally {
    setLoadingState(submitButton, false, originalButtonText);
  }
};

// Enhanced checkbox interactions
const enhanceCheckboxes = () => {
  document.addEventListener('change', (e) => {
    if (e.target.matches('.face-select-checkbox')) {
      const label = e.target.closest('.selectable-face');
      if (e.target.checked) {
        label.style.transform = 'scale(0.95)';
        setTimeout(() => {
          label.style.transform = '';
        }, 150);
      }
    }
  });
};

// Initialize everything when DOM is loaded
document.addEventListener("DOMContentLoaded", () => {
  // Set up event listeners
  if (imgInput) {
    imgInput.addEventListener("change", handleImagePreview);
  }
  
  if (inputFormElement) {
    inputFormElement.addEventListener("submit", handleFormSubmit);
  }

  // Set up drag and drop
  if (uploadArea) {
    setupDragAndDrop();
  }

  // Enhanced interactions
  enhanceCheckboxes();

  // Handle existing multiple faces form
  const existingMultipleFacesForm = document.getElementById("multiple-faces-form");
  if (existingMultipleFacesForm) {
    existingMultipleFacesForm.addEventListener("submit", handleMultipleFacesSubmit);
  }

  console.log("✅ Ticki UI initialized successfully");
});

// Handle visibility change to reset any stuck loading states
document.addEventListener('visibilitychange', () => {
  if (document.visibilityState === 'visible') {
    document.querySelectorAll('.btn-loading').forEach(btn => {
      btn.classList.remove('btn-loading');
      btn.disabled = false;
    });
  }
});

//only accept specific input images during upload
imgInput.addEventListener("change", function(e) {
  const file = e.target.files[0];
  if (file) {
    const allowed = ["image/jpeg", "image/png", "image/gif", "image/bmp", "image/webp"];
    if (!allowed.includes(file.type)) {
      alert("Only JPG, JPEG, PNG, GIF, BMP, or WEBP images are allowed.");
      e.target.value = ""; // reset input
    }
  }
});