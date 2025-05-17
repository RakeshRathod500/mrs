function safeParseJSON(data, fallback, fieldName) {
  try {
    if (data === undefined || data === null || data === '' || data === '[' || data === '[]') {
      console.warn(`safeParseJSON: Data for ${fieldName} is invalid or empty. Using fallback:`, fallback);
      return fallback;
    }
    const parsed = JSON.parse(data);
    if (!Array.isArray(parsed) && fieldName !== 'showHospitals' && fieldName !== 'predictedDisease') {
      console.warn(`safeParseJSON: Parsed data for ${fieldName} is not an array. Using fallback:`, fallback);
      return fallback;
    }
    return parsed;
  } catch (error) {
    console.error(`Error parsing JSON for ${fieldName}:`, error, 'Data:', data);
    return fallback;
  }
}

// Modal Accessibility
document.querySelectorAll('.modal').forEach(modal => {
  let triggerElement = null;

  modal.addEventListener('show.bs.modal', (event) => {
    triggerElement = event.relatedTarget;
  });

  modal.addEventListener('hide.bs.modal', () => {
    if (triggerElement) {
      triggerElement.focus();
    }
  });

  modal.addEventListener('hidden.bs.modal', () => {
    if (triggerElement && document.activeElement !== triggerElement) {
      triggerElement.focus();
    }
    triggerElement = null;
  });
});

// Initialize on Page Load
document.addEventListener('DOMContentLoaded', () => {
  console.log("Page loaded successfully.");
});