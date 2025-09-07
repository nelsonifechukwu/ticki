// SPA JavaScript for Ticki Face Search
// This file handles all client-side interactions for the Single Page Application

// DOM Elements
const imgInput = document.getElementById("input-img-preview");
const imgInputLabel = document.getElementById("input-img-label");
const imgContainer = document.querySelector(".input-img-container");
const searchForm = document.getElementById("search-form");
const uploadArea = document.querySelector(".upload-area");
const thresholdSlider = document.getElementById("threshold");
const thresholdValue = document.querySelector(".threshold-value");
const resultsContainer = document.getElementById("results-container");
const multipleFacesContainer = document.querySelector(
  ".multiple-faces-container"
);
const facesGrid = document.getElementById("faces-grid");
const multipleFacesForm = document.getElementById("multiple-faces-form");

// Toast notification system
const showToast = (message, type = "info") => {
  const toast = document.createElement("div");
  toast.className = `toast toast-${type}`;
  toast.innerHTML = `
    <i class="fas ${getToastIcon(type)}"></i>
    <span>${message}</span>
  `;

  document.body.appendChild(toast);

  // Trigger animation
  setTimeout(() => toast.classList.add("show"), 100);

  // Remove after 4 seconds
  setTimeout(() => {
    toast.classList.remove("show");
    setTimeout(() => document.body.removeChild(toast), 300);
  }, 4000);
};

const getToastIcon = (type) => {
  const icons = {
    success: "fa-check-circle",
    error: "fa-exclamation-circle",
    warning: "fa-exclamation-triangle",
    info: "fa-info-circle",
  };
  return icons[type] || icons.info;
};

// Add CSS styles dynamically
const addStyles = () => {
  if (document.querySelector("#spa-styles")) return;

  const style = document.createElement("style");
  style.id = "spa-styles";
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
    
    .upload-area.drag-over {
      border-color: var(--success-color) !important;
      background: rgba(17, 153, 142, 0.1) !important;
      transform: scale(1.02);
    }
    
    .fade-in {
      animation: fadeIn 0.5s ease-in;
    }
    
    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(10px); }
      to { opacity: 1; transform: translateY(0); }
    }
    
    .btn-loading {
      opacity: 0.7;
      cursor: not-allowed;
    }
    
    .preview-image {
      max-width: 100%;
      max-height: 200px;
      border-radius: var(--border-radius-small);
    }
    
    .upload-area.has-image {
      padding: 10px;
      border-color: var(--success-color);
      background: rgba(17, 153, 142, 0.05);
    }
    
    .face-label {
      display: block;
      font-size: 12px;
      font-weight: 500;
      color: var(--text-secondary);
      margin-top: 8px;
      text-align: center;
    }
    
    .selectable-face input:checked + div .face-label {
      color: var(--primary-color);
      font-weight: 600;
    }
  `;
  document.head.appendChild(style);
};

// Initialize styles
addStyles();

// Image preview handler
const handleImagePreview = () => {
  const file = imgInput.files[0];
  if (!file) return;

  // Validate file type
  if (!file.type.startsWith("image/")) {
    showToast("Please select a valid image file", "error");
    return;
  }

  // Validate file size (10MB limit)
  if (file.size > 10 * 1024 * 1024) {
    showToast("File size must be less than 10MB", "error");
    return;
  }

  const reader = new FileReader();
  reader.readAsDataURL(file);

  reader.onload = () => {
    // Update upload area
    uploadArea.classList.add("has-image");
    uploadArea.innerHTML = `<img src="${reader.result}" alt="Preview" class="preview-image" />`;

    showToast("Image loaded successfully!", "success");
  };

  reader.onerror = () => {
    showToast("Error reading file. Please try again.", "error");
  };
};

// Drag and drop functionality
const setupDragAndDrop = () => {
  let dragCounter = 0;

  const handleDragEnter = (e) => {
    e.preventDefault();
    dragCounter++;
    uploadArea.classList.add("drag-over");
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    dragCounter--;
    if (dragCounter === 0) {
      uploadArea.classList.remove("drag-over");
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    dragCounter = 0;
    uploadArea.classList.remove("drag-over");

    const files = e.dataTransfer.files;
    if (files.length > 0) {
      imgInput.files = files;
      handleImagePreview();
    }
  };

  uploadArea.addEventListener("dragenter", handleDragEnter);
  uploadArea.addEventListener("dragleave", handleDragLeave);
  uploadArea.addEventListener("dragover", handleDragOver);
  uploadArea.addEventListener("drop", handleDrop);
};

// Loading state management
const setLoadingState = (button, isLoading, originalText) => {
  if (isLoading) {
    button.disabled = true;
    button.classList.add("btn-loading");
    button.innerHTML = `<span>Processing...</span>`;
  } else {
    button.disabled = false;
    button.classList.remove("btn-loading");
    button.innerHTML = originalText;
  }
};

// API call helper
const makeAPICall = async (url, formData) => {
  const response = await fetch(url, {
    method: "POST",
    body: formData,
  });

  data = await response.json();

  if (!response.ok) {
    const error = new Error(`HTTP error! status: ${response.status}`);
    error.status = response.status;
    error.body = data;
    throw error;
  }

  return data;
};

// Format results for display
const formatResults = (results) => {
  if (!results || results.length === 0) {
    return `
      <div class="no-results">
        <div class="no-results-icon">
          <i class="fas fa-search"></i>
        </div>
        <h4>No similar faces found</h4>
        <p>Try adjusting the similarity threshold or uploading a different image</p>
      </div>
    `;
  }

  return results
    .map(
      (result) => `
    <div class="result-item">
      <img src="${result.image_url}" alt="Found image" />
      <div class="similarity-score">${result.similarity.toFixed(2)}</div>
    </div>
  `
    )
    .join("");
};

// Format multiple faces selection
const formatMultipleFaces = (faces) => {
  return faces
    .map(
      (face) => `
    <label class="selectable-face">
      <input
        type="checkbox"
        name="selected_faces"
        value="face_${face.index}"
        class="face-select-checkbox"
      />
      <div>
        <img src="${face.url}" alt="${face.name}" />
        <span class="face-label">${face.name}</span>
      </div>
    </label>
  `
    )
    .join("");
};

// Main form submission handler
const handleFormSubmit = async (ev) => {
  ev.preventDefault();

  const submitButton = ev.target.querySelector('button[type="submit"]');
  const originalButtonText = submitButton.innerHTML;

  // Validation
  if (!imgInput.files || imgInput.files.length === 0) {
    showToast("Please select an image first", "warning");
    return;
  }

  // Set loading state
  setLoadingState(submitButton, true, originalButtonText);

  try {
    const formData = new FormData();
    formData.append("query-img", imgInput.files[0]);
    formData.append("threshold", thresholdSlider.value);

    const result = await makeAPICall("/", formData);

    if (result.status === "multiple_faces") {
      // Handle multiple faces detected
      facesGrid.innerHTML = formatMultipleFaces(result.faces);
      multipleFacesContainer.style.display = "block";
      multipleFacesContainer.classList.add("fade-in");

      // Hide results section
      resultsContainer.innerHTML = `
        <div class="no-results">
          <div class="no-results-icon">
            <i class="fas fa-user-friends"></i>
          </div>
          <h4>Multiple faces detected</h4>
          <p>Please select which faces to search for above</p>
        </div>
      `;

      showToast(result.message, "info");
    } else if (result.status === "success") {
      // Handle single face success
      multipleFacesContainer.style.display = "none";
      resultsContainer.innerHTML = formatResults(result.results);
      resultsContainer.classList.add("fade-in");

      if (result.results.length > 0) {
        showToast(
          `Found ${result.results.length} similar ${
            result.results.length === 1 ? "image" : "images"
          }!`,
          "success"
        );
      } else {
        showToast("No similar faces found in the repository.", "info");
      }
    }
  } catch (error) {
    showToast(error.body.details, "error");
    // Try to parse error response
  } finally {
    setLoadingState(submitButton, false, originalButtonText);
  }
};

// Multiple faces form submission
const handleMultipleFacesSubmit = async (ev) => {
  ev.preventDefault();

  const submitButton = ev.target.querySelector('button[type="submit"]');
  const originalButtonText = submitButton.innerHTML;

  // Validation
  const selectedFaces = ev.target.querySelectorAll(
    'input[name="selected_faces"]:checked'
  );
  if (selectedFaces.length === 0) {
    showToast("Please select at least one face before submitting.", "warning");
    return;
  }

  setLoadingState(submitButton, true, originalButtonText);

  try {
    const formData = new FormData();

    // Add selected faces
    selectedFaces.forEach((checkbox) => {
      formData.append("selected_faces", checkbox.value);
    });

    // Add threshold
    formData.append("threshold", thresholdSlider.value);

    const result = await makeAPICall("/multiple-faces", formData);

    if (result.status === "success") {
      // Hide multiple faces container
      multipleFacesContainer.style.display = "none";

      // Show results
      resultsContainer.innerHTML = formatResults(result.results);
      resultsContainer.classList.add("fade-in");

      if (result.results.length > 0) {
        showToast(
          `Found ${result.results.length} similar ${
            result.results.length === 1 ? "image" : "images"
          }!`,
          "success"
        );
      } else {
        showToast("No similar faces found for the selected faces.", "info");
      }
    }
  } catch (error) {
    showToast(error.body.details, "error");
    // Try to parse error response
  } finally {
    setLoadingState(submitButton, false, originalButtonText);
  }
};

// Threshold slider update
const updateThresholdValue = () => {
  thresholdValue.textContent = thresholdSlider.value;
};

// Enhanced checkbox interactions
const enhanceCheckboxes = () => {
  document.addEventListener("change", (e) => {
    if (e.target.matches(".face-select-checkbox")) {
      const label = e.target.closest(".selectable-face");
      if (e.target.checked) {
        label.style.transform = "scale(0.95)";
        setTimeout(() => {
          label.style.transform = "";
        }, 150);
      }
    }
  });
};

// Store original upload area content
const originalUploadAreaContent = `
  <div class="upload-icon">
    <i class="fas fa-cloud-upload-alt"></i>
  </div>
  <div class="upload-text">
    <div class="upload-text-main">Drop image here</div>
    <div class="upload-text-sub">or click to browse</div>
  </div>
`;

// File input validation
const validateFileInput = (e) => {
  const file = e.target.files[0];
  if (file) {
    const allowed = [
      "image/jpeg",
      "image/png",
      "image/gif",
      "image/bmp",
      "image/webp",
    ];
    if (!allowed.includes(file.type)) {
      showToast(
        "Only JPG, JPEG, PNG, GIF, BMP, or WEBP images are allowed.",
        "error"
      );
      e.target.value = ""; // reset input
    }
  } else if (e.target.files.length === 0) {
    // Handle case where user cancels file selection - restore original state
    uploadArea.innerHTML = originalUploadAreaContent;
    uploadArea.classList.remove("has-image");
    e.target.value = "";
  }
};

// Initialize everything when DOM is loaded
document.addEventListener("DOMContentLoaded", () => {
  // Set up event listeners
  if (imgInput) {
    imgInput.addEventListener("change", handleImagePreview);
    imgInput.addEventListener("change", validateFileInput);
  }

  if (searchForm) {
    searchForm.addEventListener("submit", handleFormSubmit);
  }

  if (multipleFacesForm) {
    multipleFacesForm.addEventListener("submit", handleMultipleFacesSubmit);
  }

  if (thresholdSlider) {
    thresholdSlider.addEventListener("input", updateThresholdValue);
  }

  // Set up drag and drop
  if (uploadArea) {
    setupDragAndDrop();
  }

  // Enhanced interactions
  enhanceCheckboxes();

  console.log("✅ Ticki SPA initialized successfully");
});

// Handle visibility change to reset any stuck loading states
document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "visible") {
    document.querySelectorAll(".btn-loading").forEach((btn) => {
      btn.classList.remove("btn-loading");
      btn.disabled = false;
    });
  }
});
