// API Configuration
const API_BASE_URL = "http://localhost:8081/api"; // Adjust this to your Python backend URL

// DOM Elements
const queryForm = document.getElementById("queryForm");
const uploadForm = document.getElementById("uploadForm");
const urlForm = document.getElementById("urlForm");
const queryBtn = document.getElementById("queryBtn");
const uploadBtn = document.getElementById("uploadBtn");
const urlUploadBtn = document.getElementById("urlUploadBtn");
const queryLoading = document.getElementById("queryLoading");
const uploadLoading = document.getElementById("uploadLoading");
const queryResults = document.getElementById("queryResults");
const uploadStatus = document.getElementById("uploadStatus");
const pdfFiles = document.getElementById("pdfFiles");
const fileLabel = document.getElementById("fileLabel");
const selectedFiles = document.getElementById("selectedFiles");
const fileMethodBtn = document.getElementById("fileMethodBtn");
const linkMethodBtn = document.getElementById("linkMethodBtn");
const fileUploadSection = document.getElementById("fileUploadSection");
const urlUploadSection = document.getElementById("urlUploadSection");

// Upload Method Switching
fileMethodBtn.addEventListener("click", function () {
  switchUploadMethod("file");
});

linkMethodBtn.addEventListener("click", function () {
  switchUploadMethod("url");
});

function switchUploadMethod(method) {
  if (method === "file") {
    fileMethodBtn.classList.add("active");
    linkMethodBtn.classList.remove("active");
    fileUploadSection.classList.add("active");
    urlUploadSection.classList.remove("active");
  } else {
    linkMethodBtn.classList.add("active");
    fileMethodBtn.classList.remove("active");
    urlUploadSection.classList.add("active");
    fileUploadSection.classList.remove("active");
  }
  clearUploadStatus();
}

// Keep track of selected files
let selectedFilesArray = [];

// File Upload Handling
pdfFiles.addEventListener("change", function (e) {
  addFiles(Array.from(e.target.files));
});

// Drag and drop functionality
fileLabel.addEventListener("dragover", function (e) {
  e.preventDefault();
  fileLabel.classList.add("dragover");
});

fileLabel.addEventListener("dragleave", function (e) {
  e.preventDefault();
  fileLabel.classList.remove("dragover");
});

fileLabel.addEventListener("drop", function (e) {
  e.preventDefault();
  fileLabel.classList.remove("dragover");
  const files = Array.from(e.dataTransfer.files);
  addFiles(files);
});

function addFiles(newFiles) {
  // Filter out non-PDF files and duplicates
  const pdfFiles = newFiles.filter((file) => file.type === "application/pdf");

  pdfFiles.forEach((file) => {
    // Check if file is already selected (by name and size)
    const isDuplicate = selectedFilesArray.some(
      (existingFile) =>
        existingFile.name === file.name && existingFile.size === file.size,
    );

    if (!isDuplicate) {
      selectedFilesArray.push(file);
    }
  });

  displaySelectedFiles();
  updateFileInput();
}

function removeFile(index) {
  selectedFilesArray.splice(index, 1);
  displaySelectedFiles();
  updateFileInput();
}

function displaySelectedFiles() {
  if (selectedFilesArray.length === 0) {
    selectedFiles.innerHTML = "";
    return;
  }

  let html =
    '<div style="font-weight: 600; margin-bottom: 10px;">Selected Files:</div>';
  selectedFilesArray.forEach((file, index) => {
    html += `
                    <div style="padding: 8px; background: #f0f2ff; border-radius: 5px; margin-bottom: 5px; display: flex; justify-content: space-between; align-items: center;">
                        <div style="flex: 1; min-width: 0;">
                            <div style="font-weight: 500; text-overflow: ellipsis; overflow: hidden; white-space: nowrap;">${file.name}</div>
                            <div style="color: #666; font-size: 12px;">${(file.size / 1024 / 1024).toFixed(2)} MB</div>
                        </div>
                        <button 
                            type="button" 
                            onclick="removeFile(${index})" 
                            style="background: #ff4757; color: white; border: none; border-radius: 50%; width: 24px; height: 24px; cursor: pointer; display: flex; align-items: center; justify-content: center; margin-left: 10px; font-size: 14px; line-height: 1;"
                            title="Remove file"
                        >×</button>
                    </div>
                `;
  });
  selectedFiles.innerHTML = html;
}

function updateFileInput() {
  // Create a new DataTransfer object to update the file input
  const dt = new DataTransfer();
  selectedFilesArray.forEach((file) => {
    dt.items.add(file);
  });
  pdfFiles.files = dt.files;
}

// Query Form Handler
queryForm.addEventListener("submit", async function (e) {
  e.preventDefault();

  const query = document.getElementById("query").value;
  const numResults = document.getElementById("numResults").value;

  showLoading(queryLoading, queryBtn, "Searching...");
  clearResults();

  try {
    const response = await fetch(`${API_BASE_URL}/query`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        query: query,
        num_results: parseInt(numResults),
      }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    displayResults(data.results);
  } catch (error) {
    console.error("Query error:", error);
    showError(
      "Failed to query database. Please check if the backend is running.",
    );
  } finally {
    hideLoading(queryLoading, queryBtn, "Search Database");
  }
});

// Upload Form Handler
uploadForm.addEventListener("submit", async function (e) {
  e.preventDefault();

  if (selectedFilesArray.length === 0) {
    showError("Please select at least one PDF file.");
    return;
  }

  showLoading(uploadLoading, uploadBtn, "Uploading...");
  clearUploadStatus();

  const formData = new FormData();
  selectedFilesArray.forEach((file) => {
    formData.append("files", file);
  });

  try {
    const response = await fetch(`${API_BASE_URL}/upload`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    showSuccess(
      `Successfully uploaded ${selectedFilesArray.length} PDF(s) to the database!`,
    );
    uploadForm.reset();
    selectedFilesArray = [];
    selectedFiles.innerHTML = "";
  } catch (error) {
    console.error("Upload error:", error);
    showError(
      "Failed to upload PDFs. Please check if the backend is running and files are valid.",
    );
  } finally {
    hideLoading(uploadLoading, uploadBtn, "Upload to Database");
  }
});

function updateFileInput() {
  // Create a new DataTransfer object to update the file input
  const dt = new DataTransfer();
  selectedFilesArray.forEach((file) => {
    dt.items.add(file);
  });
  pdfFiles.files = dt.files;
}

function removeFile(index) {
  selectedFilesArray.splice(index, 1);
  displaySelectedFiles();
  updateFileInput();
}

function displaySelectedFiles() {
  if (selectedFilesArray.length === 0) {
    selectedFiles.innerHTML = "";
    return;
  }

  let html =
    '<div style="font-weight: 600; margin-bottom: 10px;">Selected Files:</div>';
  selectedFilesArray.forEach((file, index) => {
    html += `
                    <div style="padding: 8px; background: #f0f2ff; border-radius: 5px; margin-bottom: 5px; display: flex; justify-content: space-between; align-items: center;">
                        <div style="flex: 1; min-width: 0;">
                            <div style="font-weight: 500; truncate: ellipsis; overflow: hidden; white-space: nowrap;">${file.name}</div>
                            <div style="color: #666; font-size: 12px;">${(file.size / 1024 / 1024).toFixed(2)} MB</div>
                        </div>
                        <button 
                            type="button" 
                            onclick="removeFile(${index})" 
                            style="background: #ff4757; color: white; border: none; border-radius: 50%; width: 24px; height: 24px; cursor: pointer; display: flex; align-items: center; justify-content: center; margin-left: 10px; font-size: 12px; line-height: 1; padding: 0; box-sizing: border-box; flex-shrink: 0;"
                            title="Remove file"
                        >×</button>
                    </div>
                `;
  });
  selectedFiles.innerHTML = html;
}

function updateFileInput() {
  // Create a new DataTransfer object to update the file input
  const dt = new DataTransfer();
  selectedFilesArray.forEach((file) => {
    dt.items.add(file);
  });
  pdfFiles.files = dt.files;
}

// Query Form Handler
queryForm.addEventListener("submit", async function (e) {
  e.preventDefault();

  const query = document.getElementById("query").value;
  const numResults = document.getElementById("numResults").value;

  showLoading(queryLoading, queryBtn, "Searching...");
  clearResults();

  try {
    const response = await fetch(`${API_BASE_URL}/query`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        query: query,
        num_results: parseInt(numResults),
      }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    displayResults(data.results);
  } catch (error) {
    console.error("Query error:", error);
    showError(
      "Failed to query database. Please check if the backend is running.",
    );
  } finally {
    hideLoading(queryLoading, queryBtn, "Search Database");
  }
});

// Upload Form Handler
uploadForm.addEventListener("submit", async function (e) {
  e.preventDefault();

  const files = pdfFiles.files;
  if (files.length === 0) {
    showError("Please select at least one PDF file.");
    return;
  }

  showLoading(uploadLoading, uploadBtn, "Uploading...");
  clearUploadStatus();

  const formData = new FormData();
  for (let i = 0; i < files.length; i++) {
    formData.append("files", files[i]);
  }

  try {
    const response = await fetch(`${API_BASE_URL}/upload`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    showSuccess(
      `Successfully uploaded ${files.length} PDF(s) to the database!`,
    );
    uploadForm.reset();
    selectedFiles.innerHTML = "";
  } catch (error) {
    console.error("Upload error:", error);
    showError(
      "Failed to upload PDFs. Please check if the backend is running and files are valid.",
    );
  } finally {
    hideLoading(uploadLoading, uploadBtn, "Upload to Database");
  }
});

// URL Upload Form Handler
urlForm.addEventListener("submit", async function (e) {
  e.preventDefault();

  const singleUrl = document.getElementById("documentUrl").value.trim();
  const urlListText = document.getElementById("urlList").value.trim();

  let urls = [];
  if (singleUrl) {
    urls.push(singleUrl);
  }
  if (urlListText) {
    const listUrls = urlListText
      .split("\n")
      .map((url) => url.trim())
      .filter((url) => url.length > 0);
    urls = urls.concat(listUrls);
  }

  if (urls.length === 0) {
    showError("Please enter at least one URL.");
    return;
  }

  // Validate URLs
  const invalidUrls = urls.filter((url) => !isValidUrl(url));
  if (invalidUrls.length > 0) {
    showError(`Invalid URLs found: ${invalidUrls.join(", ")}`);
    return;
  }

  showLoading(uploadLoading, urlUploadBtn, "Processing URLs...");
  clearUploadStatus();

  try {
    const response = await fetch(`${API_BASE_URL}/upload-url`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        urls: urls,
      }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    showSuccess(
      `Successfully processed ${urls.length} URL(s) and added documents to the database!`,
    );
    urlForm.reset();
  } catch (error) {
    console.error("URL upload error:", error);
    showError(
      "Failed to process URLs. Please check if the backend is running and URLs are accessible.",
    );
  } finally {
    hideLoading(uploadLoading, urlUploadBtn, "Add from URLs");
  }
});

// Utility Functions
function isValidUrl(string) {
  try {
    new URL(string);
    return true;
  } catch (_) {
    return false;
  }
}

function showLoading(loadingElement, buttonElement, buttonText) {
  loadingElement.classList.add("show");
  buttonElement.disabled = true;
  buttonElement.textContent = buttonText;
}

function hideLoading(loadingElement, buttonElement, originalText) {
  loadingElement.classList.remove("show");
  buttonElement.disabled = false;
  buttonElement.textContent = originalText;
}

function displayResults(results) {
  if (!results || results.length === 0) {
    queryResults.innerHTML =
      '<div class="result-item"><p>No results found for your query.</p></div>';
    return;
  }

  let html = "";
  results.forEach((result, index) => {
    html += `
                    <div class="result-item">
                        <h3>Result ${index + 1}</h3>
                        <p><strong>Score:</strong> ${result.score ? result.score.toFixed(4) : "N/A"}</p>
                        <p><strong>Content:</strong> ${result.content || result.text || "No content available"}</p>
                        <p><strong>Source:</strong> ${result.location || "No source available"}</p>
                        <p><strong>Document Section:</strong> ${result.section || "No section available"}</p>
                    </div>
                `;
  });
  queryResults.innerHTML = html;
}

function showSuccess(message) {
  uploadStatus.innerHTML = `<div class="success-message">${message}</div>`;
}

function showError(message) {
  const errorDiv = `<div class="error-message">${message}</div>`;
  if (uploadStatus.innerHTML.includes("error-message")) {
    uploadStatus.innerHTML = errorDiv;
  } else {
    queryResults.innerHTML = errorDiv;
  }
}

function clearResults() {
  queryResults.innerHTML = "";
}

function clearUploadStatus() {
  uploadStatus.innerHTML = "";
}

// Initialize
console.log("Vector Database Client initialized");
console.log("Backend URL:", API_BASE_URL);
